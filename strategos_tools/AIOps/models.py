from strategos_tools.core.PYCONSTS import *

import pickle, torch as pt
from torch               import float32 as tf32
from torch.nn            import Linear, BatchNorm1d
from torch.nn.functional import relu
from torch.func          import vmap, stack_module_state, functional_call


# ==================================================================================================
# PUBLIC STUB MODELS
# These are drop-in replacements for our CFR advantage-estimation models so the program can run 
# end-to-end without exposing proprietary model architecture. Every entry point the pipeline touches
# is here with matching signatures, tensor shapes, device placement, and checkpoint format, so
# collection, training, and inference all run normally. The adv estimates themselves are arbitrary: 
# correctly shaped, finite, and bounded, but carrying no strategic meaning. A CFR run against these
# stubs will execute and converge to nothing.
# ==================================================================================================


# Stand-in for the real advantage estimator. Consumes the same tensors for game history, observable
# cards, and actions, returns one scalar per evaluated action. Internally just a small generic MLP
# over time-pooled history features, normalized card IDs, and action vectors.
class AdvNet( pt.nn.Module ):

	# ----- INITIALIZATION -----------------------------------------------------

	def __init__( self, modelIter=-1, modelSize=256, modelDevice="cuda:0", load_from_file="", for_training=False ):
		super().__init__()
		if modelDevice.startswith( 'cuda' ): pt.backends.cuda.enable_mem_efficient_sdp( True )

		self.ModelIter = modelIter
		self.ModelSize = modelSize
		self.__set_layer_sizes()
		self.__build_layers()
		self.__initialize_layers()
		if load_from_file and ( modelIter!=0 ):
			self.load_parameters( modelIter, load_from_file )
		self.train( mode=for_training )

	def __set_layer_sizes( self ):

		if self.ModelSize not in [64,128,256,512]:
			raise ValueError( f"Invalid AdvNet model size specified: {self.ModelSize}" )

		self.L_SIZE     = self.ModelSize // 4                  # hidden width
		self.A_SIZE     = EVEC_SIZE                            # action event vector
		self.H_SIZE     = EVEC_SIZE                            # time-pooled history vector
		self.C_SIZE     = NUM_CARD_SETS * CVEC_SIZE            # [cID,rID,sID] per card set
		self.STATE_SIZE = self.H_SIZE + self.C_SIZE            # concat( histFeats,cardFeats )
		self.FUSED_SIZE = self.L_SIZE + self.L_SIZE            # concat( stateEnc,actionEnc )

		# Event vectors are chip-denominated and unbounded, so they get squashed on the way in and
		# the estimates get squashed on the way out. Everything downstream assumes finite advs, and
		# the training loop runs under fp16 autocast.
		self.EVEC_SCALE = 0.01
		self.ADV_SCALE  = 10.0

	def __build_layers( self ):
		self.StubFeatureL = Linear( in_features=self.STATE_SIZE, out_features=self.L_SIZE )
		self.StubActionL  = Linear( in_features=self.A_SIZE,     out_features=self.L_SIZE )
		self.StubFuseL    = Linear( in_features=self.FUSED_SIZE, out_features=self.L_SIZE )
		self.StubFuseBN   = BatchNorm1d( num_features=self.L_SIZE )
		self.StubAdvOut   = Linear( in_features=self.L_SIZE,     out_features=1 )

	def __initialize_layers( self ):
		for pName,pVals in self.named_parameters():
			if   pName.endswith( '.bias' ): pVals.data.fill_( 0.0 )
			elif 'BN' in pName:             pVals.data.fill_( 1.0 )
			else:                           pt.nn.init.kaiming_normal_( pVals, nonlinearity='relu' )


	# ----- FORWARD PASS -------------------------------------------------------

	# Collapses a card set of any width down to normalized [cID,rID,sID] means. Hole & flop tensors
	# arrive rank 2, turn and river rank 1, so the reshape normalizes that away. During 
	# counterfactual inference, one board is shared across many candidate histories, so board 
	# features get broadcast out to nH to line up with the hole cards.
	def __card_features( self, nH, cIDs, rIDs, sIDs ):

		n = cIDs.shape[ 0 ]
		c = cIDs.reshape( n,-1 ).to( tf32 ).mean( dim=FEATURE_DIM,keepdim=True ) / DECK_SIZE
		r = rIDs.reshape( n,-1 ).to( tf32 ).mean( dim=FEATURE_DIM,keepdim=True ) / NUM_RANKS
		s = sIDs.reshape( n,-1 ).to( tf32 ).mean( dim=FEATURE_DIM,keepdim=True ) / NUM_SUITS

		cardFeatures = pt.cat( tensors=(c,r,s),dim=FEATURE_DIM ) # (n,CVEC_SIZE)
		return cardFeatures if n==nH else cardFeatures.expand( nH,-1 )

	# BatchNorm1d rejects single-sample batches while training, but a ragged final training batch can
	# hand us exactly one sample. Doubling the row satisfies that requirement while keeping the norm
	# parameters in the graph, which matters because DDP rejects parameters that skip the forward.
	def __norm( self, x ):
		if self.training and x.shape[ 0 ]==1:
			return self.StubFuseBN( x.repeat(( 2,1 )) )[ :1 ]
		return self.StubFuseBN( x )

	# Inputs: Game hist, observable cards, & actions to eval; hMask for var length H when training
	# Output: Tensor of advantage estimates for each evaluated action
	def forward( self, H, hCc,hCr,hCs, fCc,fCr,fCs, tCc,tCr,tCs, rCc,rCr,rCs, A, nA, hMask=None ):

		Inference_Mode = hMask is None # Transformer mask only needed for training due to nonuniform H
		nH   = H.shape[ 0 ] # Total number of UNIQUE input histories
		hLen = H.shape[ 1 ] # For nonuniform hLen, this is the longest sample history

		## TIME-POOLED HISTORY ##
		# Padded steps are zeroed rather than dropped, and the divisor is the static padded length,
		# so an all-padding row pools to zero instead of NaN.
		hSteps       = H if Inference_Mode else H * (~hMask).unsqueeze( -1 ).to( H.dtype )
		histFeatures = pt.tanh( hSteps.sum( dim=1 ) * (self.EVEC_SCALE/max( hLen,1 )) ) # (nH,H_SIZE)

		## CARD FEATURES ##
		holeFeatures  = self.__card_features( nH, hCc,hCr,hCs ) # (nH,CVEC_SIZE)
		flopFeatures  = self.__card_features( nH, fCc,fCr,fCs ) # (nH,CVEC_SIZE)
		turnFeatures  = self.__card_features( nH, tCc,tCr,tCs ) # (nH,CVEC_SIZE)
		riverFeatures = self.__card_features( nH, rCc,rCr,rCs ) # (nH,CVEC_SIZE)

		## STATE ENCODING ##
		stateFeatures = pt.cat( tensors=(histFeatures,holeFeatures,flopFeatures,turnFeatures,riverFeatures),
								dim=FEATURE_DIM )                    # (nH,STATE_SIZE)
		stateEncoding = relu( self.StubFeatureL( stateFeatures ) )   # (nH,L_SIZE)

		## ACTION ENCODING ##
		actionEncoding = relu( self.StubActionL( pt.tanh( A*self.EVEC_SCALE ) ) ) # (nA,L_SIZE)

		## INFERENCE SAMPLE ALIGNMENT ##
		# inference ⟹ many h per A or many a per H ⟹ repeat encodings for sample alignment
		if Inference_Mode:
			stateEncoding  = stateEncoding.repeat_interleave( nA,0 ) # (nS,L_SIZE)
			actionEncoding = actionEncoding.repeat(( nH,1 ))         # (nS,L_SIZE)

		## CONTEXTUAL ACTION ENCODING ##
		contextualActions = pt.cat( tensors=(stateEncoding,actionEncoding),dim=FEATURE_DIM )
		fusedEncoding     = relu( self.__norm( self.StubFuseL( contextualActions ) ) )

		## :| ##
		AdvEstimates = pt.tanh( self.StubAdvOut( fusedEncoding ) ) * self.ADV_SCALE
		return AdvEstimates


	# ----- MODEL PERSISTENCE --------------------------------------------------

	# Serializable model representation; can reconstruct AdvNet from this after loading
	def get_model_dict( self ):
		modelDict = { 'IterNum': self.ModelIter, 'state_dict': self.state_dict() }
		return modelDict

	def save_parameters( self, to_file="", parameter_override={} ):
		modelDict = self.get_model_dict() or parameter_override
		with open( to_file,'ab' ) as modelFile:
			pickle.dump( modelDict, modelFile, protocol=-1 )

	# Just regularizes parameter names...don't worry about it, ok?
	def __translate_model_keys( self, state_dict ):
		loadedKeys = list( state_dict.keys() )
		numLayers  = len( loadedKeys )
		for L in range( numLayers ):
			layerName = loadedKeys[ L ]
			if layerName.startswith( 'Ctx' ):
				layerWeights = state_dict.pop( layerName )
				correctName  = layerName.replace( 'Ctx','CTX' )
				state_dict[ correctName ] = layerWeights
		return state_dict

	def load_parameters( self, from_iter, from_file ):
		with open( from_file,'rb' ) as modelFile:
			while True:
				modelDict = pickle.load( modelFile )
				if modelDict[ 'IterNum' ] == from_iter:
					self.ModelIter = from_iter
					stateDict = self.__translate_model_keys( modelDict[ 'state_dict' ] )
					self.load_state_dict( stateDict )
					break


	# ----- UTILITIES ----------------------------------------------------------

	def ModelName( self ):
		return f"M{self.ModelSize}T{self.ModelIter}"

	def update_training_record( self, epochsTrained, trainDict ):
		self.EpochsTrained = epochsTrained
		self.TrainDict     = trainDict

	def enumerate_layers( self ):
		layerDict = {}
		for lname,layer in self.named_parameters():
			layerDict[ lname+f"_{self.ModelIter}" ] = layer
		return layerDict

	def list_parameter_shapes( self ):

		print(f"\nITER {self.ModelIter} ADVNET PARAMETER [NAMES | SHAPES]:\n")

		nameColumns = 0
		varColumns  = 10
		for lName,_ in self.named_parameters():
			nameLen = len( lName )
			if nameLen > nameColumns:
				nameColumns = nameLen

		totalTrainable = sum( p.numel() for p in self.parameters() if p.requires_grad )
		totStr1 = "TOTAL TRAINABLE PARAMETERS"
		if len( totStr1 ) > nameColumns:
			nameColumns = len( totStr1 )
		totStr2 = f"{totalTrainable:,}"
		totStr = "  ||||  ".join( [totStr1.ljust( nameColumns ),totStr2.rjust( varColumns )] )

		for lName,lVars in self.named_parameters():
			nameStr = lName.ljust( nameColumns )
			if len( lVars.size() )==1:
				varStr = f"({lVars.size()[0]},)"
			else:
				varStr = "(" + ",".join( [str(dimSize) for dimSize in lVars.size()] ) + ")"
			lStr = "  ||||  ".join( [nameStr,varStr.rjust( varColumns )] )
			print( f"\t{lStr}" )

		print( f"\t{totStr}" )


# Runs multiple iteration AdvNets in parallel using vmap. Inference only, advnets are pretrained.
# Stacking behavior is identical to the real MultiModel so that everything downstream keeps getting
# one (T,|𝓘|,|A|) tensor per call.
class MultiModel( pt.nn.Module ):

	def __init__( self, mFile="", iterSpan=-1, modelSize=128, device="cuda:0" ):
		super().__init__()

		self.IterSpan = iterSpan

		# ----- SPECIAL CASE: ITER 0, NO PRETRAINED ADVNETS --------------------
		if iterSpan == -1:
			net = AdvNet( iterSpan, modelSize, device, mFile )
			net.eval().requires_grad_( False ).to( device )
			self.single   = net
			self._stacked = False
			return

		# ----- STANDARD CASE: STACK & VMAP PRETRAINED ADVNETS -----------------

		# First build list of pretrained AdvNets from t=0 to t=T
		nets = []
		for t in range( iterSpan+1 ):
			net = AdvNet( t, modelSize, device, mFile )
			net.eval().requires_grad_( False ).to( device ) # AdvNets already trained
			nets.append( net )

		# Stack parameters/buffers – tensors inherit device from the originals
		self._params, self._buffers = stack_module_state( nets )
		self._fwd_mod = nets[ 0 ]

		# Per-model functional call using stacked params/buffers
		def _per_model( params_i, buffers_i, *inputs ):
			return functional_call( self._fwd_mod, (params_i, buffers_i), inputs )

		self._per_model = _per_model
		self._vmap_fwd  = vmap( self._per_model, in_dims=(0,0) + (None,)*ADVNET_INPUT_LEN, out_dims=0 )

		del nets # free the individual modules - weights live in _params/_buffers

		self._stacked = True

	def forward( self, mmInputs ):

		H   = mmInputs.H    # histories

		hCc = mmInputs.hC_c # hole cardIDs
		hCr = mmInputs.hC_r # hole rankIDs
		hCs = mmInputs.hC_s # hole suitIDs

		fCc = mmInputs.fC_c # flop cardIDs
		fCr = mmInputs.fC_r # flop rankIDs
		fCs = mmInputs.fC_s # flop suitIDs

		tCc = mmInputs.tC_c # turn cardIDs
		tCr = mmInputs.tC_r # turn rankIDs
		tCs = mmInputs.tC_s # turn suitIDs

		rCc = mmInputs.rC_c # river cardIDs
		rCr = mmInputs.rC_r # river rankIDs
		rCs = mmInputs.rC_s # river suitIDs

		A   = mmInputs.A    # action vectors
		nA  = mmInputs.nA   # |A(I)|
		nI  = mmInputs.nI   # |𝓘|
		T   = mmInputs.T    # iterspan

		if not self._stacked: # untrained iter 0 special case
			out = self.single( H, hCc,hCr,hCs, fCc,fCr,fCs, tCc,tCr,tCs, rCc,rCr,rCs, A,nA ).reshape(( 1,nI,nA ))
			return out

		inputs = ( H, hCc,hCr,hCs, fCc,fCr,fCs, tCc,tCr,tCs, rCc,rCr,rCs, A,nA )

		out_flat = self._vmap_fwd( self._params, self._buffers, *inputs )
		return out_flat.reshape(( T,nI,nA ))

	# Stacked VMap models inherit device from base models so can't be moved
	# Just construct MM on the correct device in the first place
	def to( self, device ):
		pass
