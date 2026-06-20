from strategos_tools.core.PYCONSTS      import *
from strategos_tools.utils.funcs        import hms, move_console_cursor_up, clear_current_line, progress_bar
from strategos_tools.utils.data_structs import DATAMACHINE, CFR_metadata
from strategos_tools.utils.data_ops     import load_nn_samples, post_iter_cleanup
from strategos_tools.AIOps.models       import AdvNet

import pickle
import torch as pt
from contextlib       import contextmanager
from time             import sleep, time as TimeNow
from torch.nn         import SyncBatchNorm
from lightning.fabric import Fabric


# ==================================================================================================
# All the tools needed to train strategos's nn models from collected data live here. Entry points
# are purely training phase; we assume data has already been collected, appropriately preprocessed,
# and is ready to be directly loaded into the training pipeline. All tools and processes here can 
# be run on any number of GPUs.
# ==================================================================================================


# Does meta-management for the nn training process. Does not orchestrate actual training loop,
# instead is called from the training loop to keep track of ongoing loss metrics, store current
# best-performing model, perform timing tasks when profiling, print training progress, save models
# and metadata, and various other ancillary tasks. 
class TrainManager:

	def __init__( self, aNet, CFRiter, totalEpochs, mData, dataDir, modelFile, metaFile ):
		self.__new_init__( aNet, CFRiter, totalEpochs, mData, dataDir, modelFile, metaFile )

		# Checkpointing system still buggy, for now we just assume we're doing a new training run
		#if not aNet.EpochsTrained: 
			#self.__new_init__( aNet, CFRiter, totalEpochs, modelFile, mData, metaFile )
		#else:                      
			#self.__res_init__( aNet, CFRiter, totalEpochs, modelFile, mData, metaFile )

	# Initializer for a new training phase, rather than resuming a pre-existing one
	def __new_init__( self, aNet, CFRiter, totalEpochs, mData, dataDir, modelFile, metaFile ):

		self.EPOCHS  = totalEpochs
		self.CFRIter = CFRiter
		self.RANK    = None # GPU ID for this TM instance. Set downstream by model trainer.
		self.IS_MASTER_PROCESS = None # Is this TM instance on GPU 0? Set by model trainer.

		self.Model     = aNet
		self.ModelSize = aNet.ModelSize
		self.ModelName = aNet.ModelName()
		self.ModelFile = modelFile
		self.BestModel = None # Keep track of the best-performing model found so far

		self.CFR_mData = mData
		self.MetaFile  = metaFile
		
		self.DataDir = dataDir

		# Ongoing training process records
		self.EpochsDone    = 0
		self.LHist         = []   # History of epoch loss values
		self.VLHist        = []   # History of epoch val loss values
		self.MinL          = None # Min observed loss 
		self.MinVL         = None # Min observed val loss
		self.MinLe         = None # Epoch of min observed loss
		self.MinVLe        = None # Epoch of min observed val loss
		self.MaxL          = None # Max observed loss
		self.MaxVL         = None # Max observed val loss
		self.MaxLe         = None # Epoch of max observed loss
		self.MaxVLe        = None # Epoch of max observed val less
		self.VLImprovement = 0.0  # By how much val loss has improved over training

		# Subphase & overall runtime trackers for profiling
		self.TrainStart = 0
		self.TrainTime  = 0
		self._eStart    = 0 # Stores the start time of the current epoch
		self.dTime      = 0 # Total accumulated time spent on data loading ops
		self.iTime      = 0 # Total accumulated time spent on inference ops
		self.bTime      = 0 # Total accumulated time spent on backprop ops
		self.oTime      = 0 # Total accumulated time spent on optimizer ops

	# Initializer for resuming an interrupted pre-existing training phase.
	# NOTE: This functionality is not currently implemented
	def __res_init__( self, aNet, CFRiter, totalEpochs, mData, dataDir, modelFile, metaFile ):

		modelDict = aNet.get_model_dict()
		trainDict = modelDict[ 'trainDict' ]

		self.EPOCHS            = totalEpochs
		self.CFRIter           = CFRiter
		self.RANK              = None
		self.IS_MASTER_PROCESS = None 

		self.Model     = aNet
		self.ModelSize = aNet.ModelSize
		self.ModelName = aNet.ModelName()
		self.ModelFile = modelFile
		self.BestModel = modelDict

		self.CFR_mData = mData
		self.MetaFile  = metaFile

		self.DataDir = dataDir

		self.EpochsDone    = aNet.EpochsTrained
		self.LHist         = trainDict[ 'LHist'  ]
		self.VLHist        = trainDict[ 'VLHist' ]
		self.MinL          = trainDict[ 'minL'   ]
		self.MinVL         = trainDict[ 'minVL'  ]
		self.MinLe         = trainDict[ 'minLe'  ]
		self.MinVLe        = trainDict[ 'minVLe' ]
		self.MaxL          = trainDict[ 'maxL'   ]
		self.MaxVL         = trainDict[ 'maxVL'  ]
		self.MaxLe         = trainDict[ 'maxLe'  ]
		self.MaxVLe        = trainDict[ 'maxVLe' ]
		self.VLImprovement = trainDict[ 'VLI'    ]

		self._eStart    = 0
		self.TrainStart = 0
		self.TrainTime  = trainDict[ 'TrainTime' ]
		self.dTime      = trainDict[ 'dTime'     ]
		self.iTime      = trainDict[ 'iTime'     ]
		self.bTime      = trainDict[ 'bTime'     ]
		self.oTime      = trainDict[ 'oTime'     ]

	# Called after Fabric distributed environment setup to assign this TM instance's process rank
	def set_rank( self, fabRank ):
		self.RANK = fabRank # basically just the GPU ID we're operating on
		self.IS_MASTER_PROCESS = self.RANK==0

	# For profiling how much time we spend on various subphases of the training loop
	@contextmanager
	def time_phase( self, phase ):
		start = TimeNow()
		yield
		elapsed = TimeNow() - start
		if   phase == 'data':      self.dTime += elapsed
		elif phase == 'inference': self.iTime += elapsed
		elif phase == 'backprop':  self.bTime += elapsed
		elif phase == 'opt':       self.oTime += elapsed

	def get_train_dict( self ):
		return {'LHist':     self.LHist,
				'VLHist':    self.VLHist,
				'minL':      self.MinL,
				'minVL':     self.MinVL,
				'minLe':     self.MinLe,
				'minVLe':    self.MinVLe,
				'maxL':      self.MaxL,
				'maxVL':     self.MaxVL,
				'maxLe':     self.MaxLe,
				'maxVLe':    self.MaxVLe,
				'VLI':       self.VLImprovement,
				'TrainTime': self.TrainTime,
				'iTime':     self.iTime,
				'bTime':     self.bTime,
				'oTime':     self.oTime,
				'dTime':     self.dTime }

	# When an epoch results in a new minimum val loss, store corresponding model weights
	def _update_best_model( self ):

		# This stuff is for the currently-unfinished checkpointing feature
		#trainDict = self.get_train_dict()
		#self.Model.update_training_record( self.EpochsDone, trainDict )

		# Just store the best model for now, only save to disk after training completes
		self.BestModel = self.Model.get_model_dict()

	def _save_best_model( self, silent=False ):

		# Prevent race conditions: only master process (rank 0) saves model
		if self.IS_MASTER_PROCESS:

			with open( self.ModelFile,'ab' ) as mFile:
				pickle.dump( self.BestModel, mFile, protocol=-1 )

			if not silent:
				print( f"\t{self.ModelName} best model state ( MinVLe={self.MinVLe} ) saved to: {self.ModelFile}" )

	# Print metrics for ongoing training progress
	def _print_progress( self, eLossAvg, eVLossAvg, currentLR ):

		# Avoid console spam: Only master process prints
		if self.IS_MASTER_PROCESS:

			eDone    = self.EpochsDone
			eLimit   = self.EPOCHS
			eRem     = eLimit - eDone
			tDone    = TimeNow()-self.TrainStart
			eTimeAvg = tDone / eDone
			tRem     = eRem * eTimeAvg

			lrStr  = f"Current learning rate:    {currentLR:.7f}"
			maxStr = f"Maximum [L|VL] [Le|VLe]:  [{self.MaxL:.7f}|{self.MaxVL:.7f}] [{self.MaxLe}|{self.MaxVLe}]       "
			minStr = f"Minimum [L|VL] [Le|VLe]:  [{self.MinL:.7f}|{self.MinVL:.7f}] [{self.MinLe}|{self.MinVLe}]       "
			Lstr   = f"Current [L|VL]:           [{eLossAvg:.7f}|{eVLossAvg:.7f}]                                      "
			Dstr   = f"BestModel VL Improvement: {self.VLImprovement:.7f}%       "
			Tstr   = f"Epoch time running avg:   {eTimeAvg:.3f}sec       "
			Rstr   = f"Training ETR:             {hms( tRem )} (elapsed: {hms( tDone )})       "

			# Move to start of previously printed lines and print over them
			move_console_cursor_up( numLines=7 )
			
			clear_current_line()
			print( lrStr )
			
			clear_current_line()
			print( Lstr )
			
			clear_current_line()
			print( maxStr )
			
			clear_current_line()
			print( minStr )
			
			clear_current_line()
			print( Dstr )
			
			clear_current_line()
			print( Tstr )
			
			clear_current_line()
			print( Rstr )
			
			print( progress_bar( eDone,eLimit ) + f" ({eDone}/{eLimit})", end='\r' )

	# Just mark epoch start time so we can track a running avg and estimate remaining train time
	def epoch_start( self ):
		self._eStart = TimeNow()

	# Do a bunch of logging and check whether this epoch found a new best model
	def epoch_complete( self, eLossAvg, eVLossAvg, currentLR ):

		# If first epoch, set initial defaults
		if self.EpochsDone==0:
			self.MinLe  = 1
			self.MinVLe = 1
			self.MaxLe  = 1
			self.MaxVLe = 1
			self.MinL   = eLossAvg
			self.MaxL   = eLossAvg
			self.MinVL  = eVLossAvg
			self.MaxVL  = eVLossAvg
			self._update_best_model()

		self.EpochsDone+=1
		self.LHist.append( eLossAvg )
		self.VLHist.append( eVLossAvg )

		if eLossAvg > self.MaxL:
			self.MaxL   = eLossAvg
			self.MaxLe  = self.EpochsDone

		if eVLossAvg > self.MaxVL:
			self.MaxVL  = eVLossAvg
			self.MaxVLe = self.EpochsDone

		if eLossAvg < self.MinL:
			self.MinL   = eLossAvg
			self.MinLe  = self.EpochsDone

		# If this is true, we've found a new best-performing model
		if eVLossAvg < self.MinVL:
			self.MinVL         = eVLossAvg
			self.MinVLe        = self.EpochsDone
			self.VLImprovement = ((self.MaxVL - self.MinVL) / self.MaxVL)*100 if (self.MaxVL>0) else 100
			self._update_best_model()

		epochTime = TimeNow() - self._eStart
		self.TrainTime += epochTime
		self._print_progress( eLossAvg, eVLossAvg, currentLR )

	# Log start time for timing metrics
	def training_start( self ):
		self.TrainStart = TimeNow()

	# Called when training loop finishes; finalizes timing metrics and does iter-end housekeeping
	def training_complete( self, Print_Subphase_Metrics=False ):

		dP = (self.dTime / self.TrainTime) * 100
		iP = (self.iTime / self.TrainTime) * 100
		bP = (self.bTime / self.TrainTime) * 100
		oP = (self.oTime / self.TrainTime) * 100

		if self.IS_MASTER_PROCESS:

			print( '\n\n'+(f"="*100) )
			print( f"CFR ITERATION {self.CFRIter} TRAINING PHASE COMPLETE".center(100) )
			print( f"="*100 )

			self._save_best_model()

			if Print_Subphase_Metrics:
				print( f"Total time taken: {self.TrainTime:.0f}sec. Subphase timing breakdown:" )
				print( f"Total time spent on data ops :     {dP:.2f}% ( {self.dTime:.0f}sec )" )
				print( f"Total time spent on inference:     {iP:.2f}% ( {self.iTime:.0f}sec )" )
				print( f"Total time spent on backprop:      {bP:.2f}% ( {self.bTime:.0f}sec )" )
				print( f"Total time spent on optimizer ops: {oP:.2f}% ( {self.oTime:.0f}sec )" )
			else:
				print( f"Total time taken: {self.TrainTime:.0f}sec." )

			cleanupDir = self.DataDir + "/segadvs"
			self.CFR_mData.CFR_iteration_completed( self.TrainTime, self.LHist, self.VLHist, self.DataDir )
			post_iter_cleanup( advDir=cleanupDir, for_iter=self.CFRIter )
			print()

	def end_summary( self ):

		iP = (self.iTime / self.TrainTime) * 100
		bP = (self.bTime / self.TrainTime) * 100
		oP = (self.oTime / self.TrainTime) * 100
		dP = (self.dTime / self.TrainTime) * 100

		sleep( self.RANK ) # this times printing so process summaries don't overlap each other
		print( "="*50 )
		print( f"POST-TRAINING SUMMARY FOR RANK {self.RANK}".center(50) )
		print( "="*50 )
		print( f"\tMinLe     = {self.MinLe}" )
		print( f"\tMinVLe    = {self.MinVLe}" )
		print( f"\tMinL      = {self.MinL:.7f}" )
		print( f"\tMinVL     = {self.MinVL:.7f}" )
		print( f"\tMaxL      = {self.MaxL:.7f}" )
		print( f"\tMaxVL     = {self.MaxVL:.7f}" )
		print( f"\tVLChange  = {self.VLImprovement:.2f}%" )
		print( f"\tTrainTime = {self.TrainTime:.3f}s" )
		print( f"\tdTime     = {self.dTime:.3f}s ( {dP:.0f}% )" )
		print( f"\tiTime     = {self.iTime:.3f}s ( {iP:.0f}% )" )
		print( f"\tbTime     = {self.bTime:.3f}s ( {bP:.0f}% )" )
		print( f"\toTime     = {self.oTime:.3f}s ( {oP:.0f}% )" )
		print()


def _get_training_parameters( dataDir, metaFile, modelSize, epoch_override, bsize_override, lRate_override ):

	nGPU        = pt.cuda.device_count()
	totalEpochs = epoch_override or TRAIN_EPOCHS
	mData       = CFR_metadata.pyload( from_file=metaFile )
	modelIter   = mData.get_current_iter()
	modelName   = f"M{modelSize}T{modelIter}"
	iterPOV     = (( modelIter+INITIAL_POV ) % NUM_PLAYERS) + 1
	trainFile   = dataDir + f"/p{iterPOV}advs_TRAIN.pickle"
	valFile     = dataDir + f"/p{iterPOV}advs_VAL.pickle"
	tsamples    = load_nn_samples( from_file=trainFile )
	vsamples    = load_nn_samples( from_file=valFile )
	ntSamples   = len( tsamples )
	nvSamples   = len( vsamples )
	nSamples    = ntSamples + nvSamples
	bsizeScale  = nGPU / MAX_GPUS
	bsizeGlobal = int( bsize_override ) or int( BASE_BATCH_SIZE * bsizeScale )
	bsizeLocal  = int( bsizeGlobal//nGPU )
	lRate       = lRate_override or BASE_LRATE

	return {
		'nGPU':        nGPU,
		'totalEpochs': totalEpochs,
		'mData':       mData,
		'modelIter':   modelIter,
		'modelName':   modelName,
		'tsamples':    tsamples,
		'vsamples':    vsamples,
		'ntSamples':   ntSamples,
		'nvSamples':   nvSamples,
		'nSamples':    nSamples,
		'bsizeGlobal': bsizeGlobal,
		'bsizeLocal':  bsizeLocal,
		'lRate':       lRate,
	}


# Executes the actual training loop. Instantiates new untrained AdvNet, sets up distributed Fabric 
# training environment, loads distributed training data, and executes epoch-by-epoch logic.
def ModelTrainer( dataDir, modelSize, epoch_override=0, bsize_override=0, lRate_override=0 ):

	# Torch flags shown to improve training performance
	pt.set_float32_matmul_precision( 'high' )
	pt.backends.cudnn.benchmark = True

	# First, get parameters and variables we need, and set up training entities
	metaFile  = dataDir + '/metadata.pickle'
	modelFile = dataDir + '/models.pickle'
	trainVars = _get_training_parameters( dataDir, metaFile, modelSize, 
										  epoch_override, bsize_override, lRate_override )
	aNet      = AdvNet( modelIter=trainVars[ 'modelIter' ], modelSize=modelSize, for_training=True )
	aNet      = SyncBatchNorm.convert_sync_batchnorm( aNet )
	huber     = pt.nn.HuberLoss( reduction='none' )
	opt       = pt.optim.Adam( aNet.parameters(), lr=trainVars[ 'lRate' ], eps=1e-08, foreach=True )
	lRate     = trainVars[ 'lRate' ]
	TM        = TrainManager( aNet, trainVars[ 'modelIter' ], trainVars[ 'totalEpochs' ], 
							  trainVars[ 'mData' ], dataDir, modelFile, metaFile )

	# Set up and launch Fabric distributed environment
	# Multi-processes get spun up here, everything after this runs independently per-process
	Fab = Fabric( accelerator='cuda', devices=trainVars[ 'nGPU' ], 
				  strategy='ddp', precision='16-mixed', callbacks=[TM] )
	Fab.launch()
	R = Fab.global_rank # this process's rank (just this process's GPU ID, basically)

	# Set up distributed AdvNet and optimizer entities
	Fab.print( f"\nFabric engine launched." )
	Fab.print( f"Configuring fab model and optimizer..." )
	aNet,opt = Fab.setup( aNet,opt )
	Fab.print( f"Fab model and optimizer configured." )

	# Display training setup details; Fab.print makes stuff only print once despite multiple processes
	Fab.print( '\n' + (f"=" * 100) )
	Fab.print( f"CONDUCTING SDCFR ITER {trainVars[ 'modelIter' ]} DISTRIBUTED TRAINING PHASE".center( 100 ))
	Fab.print( f"=" * 100 )
	Fab.print( f"EPOCHS:        {trainVars[ 'totalEpochs' ]}"   )
	Fab.print( f"Total samples: {trainVars[ 'nSamples'    ]:,}" )
	Fab.print( f"Train samples: {trainVars[ 'ntSamples'   ]:,}" )
	Fab.print( f"Val samples:   {trainVars[ 'nvSamples'   ]:,}" )
	Fab.print( f"Global bsize:  {trainVars[ 'bsizeGlobal' ]:,}" )
	Fab.print( f"Local bsize:   {trainVars[ 'bsizeLocal'  ]:,}" )
	Fab.print( f"Learning rate: {trainVars[ 'lRate'       ]}"   )

	# It looks like lr scheduling doesn't actually help us, but should experiment more with this
	#Fab.print( f"\nConfiguring lr scheduler on wrapped optimizer..." )
	#lrDecay = pt.optim.lr_scheduler.ReduceLROnPlateau( opt, factor=0.5, patience=250, threshold=0.0001, min_lr=0.0001 )
	#Fab.print( f"LR scheduling configured successfully." )

	# Construct per-process data loaders
	# Barriers are defensive to make sure things stay in sync around potentially slow operations
	Fab.barrier() 
	s = 's' if trainVars[ 'nGPU' ] > 1 else ''
	S = 'S' if trainVars[ 'nGPU' ] > 1 else ''
	Fab.print( f"\nConstructing DATAMACHINE{S} for {trainVars[ 'nGPU' ]} rank{s}..." )
	tDM_n = DATAMACHINE( trainVars[ 'tsamples' ], trainVars[ 'bsizeLocal' ], trainVars[ 'nGPU' ], R )
	vDM_n = DATAMACHINE( trainVars[ 'vsamples' ], trainVars[ 'bsizeLocal' ], trainVars[ 'nGPU' ], R )
	Fab.barrier()
	Fab.print( f"{trainVars[ 'nGPU' ]} train & val DATAMACHINE{S} constructed successfully." )
	Fab.print( f"\nSUCCESSFULLY CONFIGURED NN, OPTIMIZER, AND DATAMACHINE{S} FOR TRAINING ON {trainVars['nGPU']} GPU{s}" )

	# Tell each TrainManager its rank
	Fab.barrier()
	Fab.call( "set_rank", fabRank=R )
	Fab.barrier() # doing this many barrier calls prob unnecessary, but helps me sleep at night

	# Begin the actual training loop
	Fab.print( '\n' + ("="*50) )
	Fab.print( f"TRAINING ITERATION {aNet.ModelIter} ADVNET".center(50) )
	Fab.print( f"Total dataset size: {trainVars[ 'nSamples' ]:,}".center(50) )
	Fab.print( ("="*50) + ('\n'*7) )
	Fab.call ( "training_start" )

	for e in range( 1,trainVars[ 'totalEpochs' ]+1 ):
		Fab.call( 'epoch_start' )
		eLossAvg  = 0.0
		eVLossAvg = 0.0

		for b in range( tDM_n.nBatches ):
			opt.zero_grad( set_to_none=True )

			# Load training batch b
			#with TM.time_phase( 'data' ):
			tBatch = tDM_n.get_batch( bIdx=b )
			H  = tBatch.H
			A  = tBatch.A
			nA = tBatch.size
			M  = tBatch.M
			hCc, hCr, hCs = tBatch.hCc, tBatch.hCr, tBatch.hCs
			fCc, fCr, fCs = tBatch.fCc, tBatch.fCr, tBatch.fCs
			tCc, tCr, tCs = tBatch.tCc, tBatch.tCr, tBatch.tCs
			rCc, rCr, rCs = tBatch.rCc, tBatch.rCr, tBatch.rCs
			advTargets    = tBatch.V
			sampleWeights = tBatch.W

			# Forward pass & loss calc for batch b
			#with TM.time_phase( 'inference' ):
			advEsts   = aNet( H, hCc,hCr,hCs, fCc,fCr,fCs, tCc,tCr,tCs, rCc,rCr,rCs, A, nA, M ).squeeze( dim=1 )
			batchLoss = huber( advEsts,advTargets ) * ((sampleWeights+1)//2)
			meanLoss  = batchLoss.mean()

			# Batch b backprop; Fabric handles multiprocess sync
			#with TM.time_phase( 'backprop' ):
			Fab.backward( meanLoss )

			#with TM.time_phase( 'opt' ):
			opt.step()

			eLossAvg += meanLoss.item()

		for b in range( vDM_n.nBatches ):

			# Load validation batch b
			#with TM.time_phase('data'):
			vBatch = vDM_n.get_batch( bIdx=b )
			H  = vBatch.H
			A  = vBatch.A
			nA = vBatch.size
			M  = vBatch.M
			hCc, hCr, hCs = vBatch.hCc, vBatch.hCr, vBatch.hCs
			fCc, fCr, fCs = vBatch.fCc, vBatch.fCr, vBatch.fCs
			tCc, tCr, tCs = vBatch.tCc, vBatch.tCr, vBatch.tCs
			rCc, rCr, rCs = vBatch.rCc, vBatch.rCr, vBatch.rCs
			advTargets    = vBatch.V
			sampleWeights = vBatch.W

			# Forward pass & loss calc for batch b
			#with TM.time_phase('inference'):
			advEsts    = aNet( H, hCc,hCr,hCs, fCc,fCr,fCs, tCc,tCr,tCs, rCc,rCr,rCs, A, nA, M ).squeeze( dim=1 )
			batchVLoss = huber( advEsts,advTargets ) * ((sampleWeights+1)//2)

			meanVLoss  = batchVLoss.mean()
			eVLossAvg += meanVLoss.item()

			#lrDecay.step( meanVLoss )

		# After all epoch batches run, average out loss, record it, and do epoch-end logic
		eLossAvg  /= tDM_n.nSamples
		eVLossAvg /= vDM_n.nSamples
		#Fab.call( "epoch_complete", eLossAvg=eLossAvg, eVLossAvg=eVLossAvg, currentLR=lrDecay._last_lr[0] )
		Fab.call( "epoch_complete", eLossAvg=eLossAvg, eVLossAvg=eVLossAvg, currentLR=lRate )

	# Sync up all processes and do training-end logic
	Fab.barrier()
	Fab.call( "training_complete" )
	Fab.barrier()