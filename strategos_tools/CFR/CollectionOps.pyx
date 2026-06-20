#distutils: language = c
#cython: language_level 3
#cython: profile = False

cimport numpy as cnp
cnp.import_array()

from libc.stdlib   cimport rand as RNG, srand as SEEDRNG, RAND_MAX, malloc, realloc, free
from cpython.array cimport clone as pyarr
from cython.view   cimport array as cyarr

cimport strategos_tools.AIOps.EstimatorOps as ESTIMATOR

from strategos_tools.env.player_ops   cimport OpponentsOf, Are_Opponents
from strategos_tools.env.gamenode_ops cimport RootNode
from strategos_tools.utils.funcs      cimport ( progressbar as PB, HMS,
                                                arange, Unzero2d, Unzero3d, ArrAdd2d, ArrAdd3d, ArrSub2d, 
												ArrSub3d, ArrMult2d, ArrMult3d, ArrDiv2d, ArrDiv3d )
from strategos_tools.utils.data_ops   import get_presolved_iters, get_rank_pretravs
from strategos_tools.AIOps.models     import AdvNet, MultiModel

import os, pickle, numpy as np

from time       import sleep, time as TimeNow
from numpy      import asarray as NP, float32 as f32, expand_dims as newaxis, sum as sumaxis, intc
from pathlib    import Path
from torch.cuda import empty_cache as clear_GPU_cache


# ==================================================================================================
# This module defines all structures and operations required to actually do the CFR tree-traversal 
# process that collects game trajectory data and calculates neural net targets from it. This 
# constitutes the collection and calculation phases of our collect -> calc -> train loop. Objects
# and operations here include:
# 	- formal game tree structures like weighted connection graphs and path mappings
# 	- central orchestration of the DFS-style collection process
# 	- math that processes data in these structures into training targets
# ==================================================================================================


# Ordered map containing all endgames reachable from a node n (n.Zn) and their payoffs (n.UZn) 
cdef class ZMap:

	# All structures here are C-level, so only cinit is needed
	def __cinit__( self, ll nKey ):

		self.n    = nKey
		self.Zn   = vector_ll()
		self.UZn  = vector_int()
		self.size = 0

	cdef bint Contains( self, ll zKey ): #noexcept:
		return self.Zn.contains( zKey )

	# Looks up the endgame payoff from the terminal node specified by zKey
	cdef int  payout_from( self, ll zKey ): #noexcept:
		return self.UZn.at( self.Zn.index_of( zKey ))

	# Adds a reachable endgame to Zn along with its payoff
	cdef void append( self, ll zKey, int U_z ): #noexcept:
		self.Zn.append( zKey )
		self.UZn.append( U_z )
		self.size+=1

	# Returns mean payoff of all endgames reachable from n, unweighted by reach probabilities
	cdef int  UnweightedMeanU( self ): #noexcept:
		return NP( self.UZn.view() ).mean()

	# Just a human-readable printout of all endgame nodes reachable from n and their payoffs
	cdef void summary( self ):

		cdef:
			uint z
			ll   zKey
			int  uz

		print( '\n'+('='*35) )
		print( f"Z[n] SUMMARY".center(35) )
		print( '='*35 )

		print( f"n = {self.n}" )
		print( f"|Z[n]| = {self.size}" )
		print( '—'*35 )

		print( "zKey".center(23) + "||" + "u(z)".center(10) )
		print( '—'*35 )

		for z from 0 <= z < self.size:
			zKey = self.Zn.at( z )
			uz   = self.payout_from( zKey )
			print( str(zKey).ljust(23) + "||" + str(uz).center(10) )

		print( '—'*35 )
		print( f"Unweighted reachable payoff average: {self.UnweightedMeanU()}" )


# Ordered map specifying all weighted GameTree connections originating @ GTNode with key from_key.
# Weights represent action probabilities and are multi-iter, multi-hand.
# - Multi-iter: Weights include action probs estimated by all iteration AdvNets trained thus far
# - Multi-hand: If origin node's acting player is an opponent of the current traversal pov player,
#               weights include opp action probs for every possible hand they could be holding.
cdef class ConnectionMap:

	# All structures here are C-level, so only cinit is needed
	def __cinit__( self, ll from_key, vector_ll to_keys, MultiMat weights ):

		self.from_key = from_key # origin node ID
		self.to_keys  = to_keys  # IDs of all of the origin node's immediate successors
		self.weights  = weights  # edge weights to each node in to_keys; same ordering as to_keys

	# Returns multi-iter, multi-hand weights to one specified successor node
	cdef flt2 get_weights( self, ll to_key ): #noexcept:
		cdef uint successorIdx = self.to_keys.index_of( to_key )
		return self.weights._data_view[ successorIdx ]

	cdef void set_weights( self, ll to_key, flt2 weights ): #noexcept:
		cdef uint targetIdx = self.to_keys.index_of( to_key )
		self.weights._data_view[ targetIdx ] = weights

	# Simply redirects a connection edge; used for calc-phase ops where we want to skip dealer nodes
	cdef void replace_connection( self, ll oldKey, ll newKey ): #noexcept:
		cdef uint replacementIdx = self.to_keys.index_of( oldKey )
		self.to_keys.replace( at_idx=replacementIdx, newItem=newKey )

	# Does the origin node connect to to_key?
	cdef bint Connects_To( self, ll key ): #noexcept:
		return self.to_keys.contains( key )


# GameTree node. Basically wraps a gamenode with game path info we need for calculating nn targets
cdef class GTNode:

	# Some more complex structures here, so actually need a proper python-level init
	def __init__( self, uint traversalPOV, gamenode fromNode, bint Is_Terminal, uint estimatorRank=0 ):
		self.__INIT__( traversalPOV, fromNode, Is_Terminal, estimatorRank )

	cdef void __INIT__( self, uint traversalPOV, gamenode fromNode, bint Is_Terminal, uint estimatorRank=0 ): #noexcept:

		cdef gamenode parentNode = fromNode.ParentNode( Include_Deals=FALSE )

		self.TraversingPlayer = traversalPOV
		self.GameNode         = fromNode
		self.Key              = self.GameNode.GTKey()
		self.ActingPlayer     = self.GameNode.ActingPlayer()
		self.I_tp             = infoset( sourceNode=self.GameNode, perspective_of=self.TraversingPlayer )
		self.GTParentKey      = parentNode.GTKey() if (Is_Terminal or parentNode.hLen > 1) else 0

		if Is_Terminal:
			self.__INIT_TERMINUS__()
		else:          
			self.__INIT_NON_TERMINUS__( estimatorRank )

	# Placeholder until we decide whether we want to re-implement strat accumulation (we prob won't)
	cdef vector_int    __AInds( self, uint numActions ): #noexcept:

		cdef int a
		cdef vector_int aInds = vector_int( init_cap=numActions )
		for a from numActions > a >= 0: # backwards so .pop gets action inds in ascending order
			aInds.append( a )
		return aInds

	# GT branch weights from self node to its successors = ActingPlayer's action probs @ from_node.
	# Includes action probs estimated by all iteration AdvNets, and for all possible opp hands if
	# ActingPlayer ≠ TraversalPOVPlayer. estimatorRank specifies which GPU to use for inference.
	cdef ConnectionMap __CGraph( self, uint estimatorRank=0 ): #noexcept:

		cdef flt3     aProbs   = ESTIMATOR.MultiStrats( self.ActingPlayer, self.I_tp, estimatorRank )
		cdef MultiMat cWeights = MultiMat( from_view=aProbs ) # always (|A|,1326,T)
		return ConnectionMap( from_key=self.Key, to_keys=self.SuccessorKeys, weights=cWeights )

	# Returns GTKeys of all non-dealer nodes on the path to the current node
	cdef vector_ll       __get_path_node_keys( self ): #noexcept:

		cdef:
			uint      hLen = self.GameNode.History.shape[ 0 ], s
			gamenode  pathNode
			vector_ll pathKeys = vector_ll()

		for s from 1 <= s < hLen:
			pathNode = self.GameNode.Predecessor( with_hlen=s )
			if pathNode.ActingPlayer() != DEALER: 
				pathKeys.append( pathNode.GTKey() )

		if not pathKeys.contains( self.Key ): 
			pathKeys.append( self.Key )

		return pathKeys

	# Returns GTKeys of all possible successor nodes implied by available actions
	cdef vector_ll     __get_successor_node_keys( self ): #noexcept:
		
		cdef:
			uint     nA    = self.A.size, s
			ll1      sKeys = pyarr( ARR_TMPLT_ll, nA, zero=False )
			gamenode n     = self.A.SourceNode()

		# ∀a∈A, generate the successor node resulting from doing a, then get its key
		for a from 0 <= a < nA:
			sKeys[ a ] = n.Successor( e=self.A.at( a ) ).GTKey()

		return vector_ll( from_view=sKeys )

	# Normal initialization for non-endgame nodes
	cdef void          __INIT_NON_TERMINUS__( self, uint estimatorRank ): #noexcept:

		self.I_ap = self.I_tp if not Are_Opponents( self.TraversingPlayer, self.ActingPlayer ) else                    \
					infoset( sourceNode=self.GameNode, perspective_of=self.ActingPlayer )
		self.A    = actionset( self.I_ap )

		# For tracking current completeness of exploration
		self.Fully_Explored = FALSE # TRUE iff RemainingAInds.size = 0
		self.Solvable       = FALSE # TRUE iff self.Fully_Explored & s.Fully_Explored ∀s ∈ self.SubKeys
		self.FullAInds      = self.__AInds( self.A.size )
		self.RemainingAInds = self.FullAInds.copy()

		# Defines connection structure between this node, its direct successors, & reachable endgames
		self.SuccessorKeys   = self.__get_successor_node_keys()
		self.ConnectionGraph = self.__CGraph( estimatorRank ) # axes = (a,h,t)
		self.SubKeys         = vector_ll()
		self.Zn              = vector_ll()
		self.FwdReaches      = None # don't need to alloc mem for this until Solvable == TRUE
		self.CFReaches       = None # same here
		self.UZn             = ZMap( self.Key )

		# Terminal properties which don't matter here
		self.Is_Terminal  = FALSE
		self.uz           = 0
		self.TerminalPath = vector_ll()

	# Special case initialization for endgame nodes; stores payoff and path information
	cdef void          __INIT_TERMINUS__( self ): #noexcept:

		# Endgame data which will be passed up the tree to calculate advs for non-terminal nodes
		self.Is_Terminal  = TRUE
		self.TerminalPath = self.PathKeys()
		self.uz           = self.GameNode.GameResults()[ self.TraversingPlayer ]

		# Non-terminal properties which don't matter here, initialize to defaults.
		self.Fully_Explored = TRUE # doing this for znodes simplifies later search for solvable subgames
		self.Solvable       = TRUE # same here
		self.FullAInds      = vector_int()
		self.RemainingAInds = vector_int()
		self.SuccessorKeys  = vector_ll()
		self.SubKeys        = vector_ll()
		self.Zn             = vector_ll()
		self.UZn            = ZMap( self.Key )
		self.FwdReaches     = None
		self.CFReaches      = None

	cdef bint            Has_Direct_Connection_To( self, ll key ): #noexcept:
		return self.ConnectionGraph.Connects_To( key )

	# Replaces dealer node connection by replacing its key with the key of the next non-dealer node
	cdef void            skip_dealer_connection( self, ll skip_deal_key, ll skip_to_key ): #noexcept:
		self.ConnectionGraph.replace_connection( oldKey=skip_deal_key, newKey=skip_to_key )

	# Called when we've traversed all branches leading away from this node
	cdef void            exploration_completed( self ): #noexcept:
		self.Fully_Explored = TRUE

	# Called when we find a new endgame reachable from this node
	cdef void            append_reachable_endgame( self, ll zKey, int uz ): #noexcept:
		self.Zn.append( zKey )
		self.UZn.append( zKey, uz )

	# 0-initializes map of forward reach probabilities from current node to reachable endgames
	cdef void            initialize_fwd_reaches( self ): #noexcept:

		cdef:
			flt3     zeros, ones
			MultiMat zeroReaches, selfReaches
			uint     nZ = self.Zn.size, nH = NUM_POSSIBLE_HANDS

		if not self.Is_Terminal:
			zeros       = cyarr( (nZ,nH,T), FLTSIZE, 'f' )
			zeros[:]    = 0
			zeroReaches = MultiMat( from_view=zeros )
			self.FwdReaches = ConnectionMap( from_key=self.Key, to_keys=self.Zn, weights=zeroReaches )

		# TODO: Do you need 𝓹(z) weights to tell you which hInds are 0?
		# Prob not; π arrs above z will be 0 there anyway
		else: # useful downstream if zNodes have π(z,z)=1 
			ones        = cyarr( (nZ,nH,T), FLTSIZE, 'f' )
			ones[:]     = 1
			selfReaches = MultiMat( from_view=ones )
			self.FwdReaches = ConnectionMap( from_key=self.Key, to_keys=self.Zn, weights=selfReaches )

	# Counts num times each player (including DEALER) has acted along path to this node
	cdef uint1           count_path_action_points( self ): #noexcept:

		cdef uint1 actionPts = pyarr( ARR_TMPLT_I, self.GameNode.PLAYER_COUNT+1, zero=True )
		cdef uint  hLen      = self.GameNode.hLen,                                                                     \
				   start     = self.GameNode.BLINDS_DONE,                                                              \
				   s, stepPlayer, p

		for s from start <= s < hLen:
			stepPlayer = self.GameNode.History[ s,PLAYEDBY ]
			actionPts[ stepPlayer ] += 1

		return actionPts

	# Ok turns out we need a public interface for this too
	cdef vector_ll       PathKeys( self ): #noexcept:
		return self.__get_path_node_keys()


# Simple [mostly] C-level fast lightweight container for GTNodes
cdef class NodeVector:

	# Init a pylist where the GTNode objects will live, but we will interact with them via pointers
	def __init__ ( self, uint init_cap=0 ): 
		self.__nodeList = []

	def __cinit__( self, uint init_cap=0 ):

		self.size     = 0
		self.capacity = init_cap if init_cap > 0 else 1
		self._data    = <void **>malloc( self.capacity * PTRSIZE )

		if self._data is NULL:
			raise MemoryError( "cinit allocation for NodeVector failed, wtf are you trying to do???" )

	cdef uint __get_min_spanning_cap( self ): #noexcept:
		cdef uint n=0
		for n from 0 <= n < 99:
			if 2**n > self.size: 
				return 2**n
		raise MemoryError( "Exactly what the fuck are you doing that you need a vector > 2^99 in size for???" )

	cdef void   resize( self, uint newCap=0 ): #noexcept:
		self.capacity = newCap if newCap > 0 else self.__get_min_spanning_cap()
		cdef void **newAlloc = <void **>realloc( self._data, self.capacity * PTRSIZE )
		self._data = newAlloc

	cdef void   shrink_wrap( self ): #noexcept:
		self.resize( newCap=self.size )

	cdef void   append( self, GTNode newNode ): #noexcept:

		self.__nodeList.append( newNode )
		self.size+=1

		if self.size == self.capacity: 
			self.resize()

		self._data[ self.size-1 ] = <void *>self.__nodeList[ self.size-1 ]

	cdef GTNode at( self, uint idx ): #noexcept:
		return <GTNode>( self._data[ idx ] )


# The tree structure we build out during exploration; just a map from GTNode keys to GTNodes
cdef class GameTree:

	def __cinit__( self ):
		self.GTKeys = vector_ll()
		self.Nodes  = NodeVector()

	cdef bint   Contains( self, ll nodeKey ): #noexcept:
		return self.GTKeys.contains( nodeKey )

	cdef GTNode node_at( self, ll nodeKey ): #noexcept:
		return <GTNode>(self.Nodes.at( self.GTKeys.index_of( nodeKey ) ))

	cdef void   append( self, GTNode newNode ): #noexcept:
		self.GTKeys.append( newNode.Key )
		self.Nodes.append( newNode )


# Orchestrates the CFR process of collecting game trajectory data by exploring the game tree
cdef class CFRCollector:

	def __init__( self, str dataDir, uint parallelRank, uint serialRank, uint nPlayers, uint gameSize, uint for_iter, uint POVplayer, uint segTravs ):
		self.__INIT__( dataDir, parallelRank, serialRank, nPlayers, gameSize, for_iter, POVplayer, segTravs )

	cdef void     __INIT__( self, str dataDir, uint parallelRank, uint serialRank, uint nPlayers, uint gameSize, uint for_iter, uint POVplayer, uint segTravs ): #noexcept:

		# Game/collection run parameters
		self.RANK_P            = parallelRank # concurrent collection worker ID
		self.RANK_S            = serialRank   # number in serial sequence of collection runs
		self.nPlayers          = nPlayers
		self.GameSize          = gameSize     # average starting stack
		self.SBlind            = (50*(gameSize//1000)) or 1
		self.MinStack          = <uint>(self.GameSize*0.75)
		self.MaxStack          = <uint>(self.GameSize*1.25)
		self.StackRange        = self.MaxStack-self.MinStack
		self.StackDeviation    = self.StackRange/16
		self.CurrentStacks     = pyarr( ARR_TMPLT_I, self.nPlayers+1, zero=True )
		self.SolvingIter       = for_iter
		self.POVplayer         = POVplayer    # ID we're playing as during collection
		self.SegmentTravReq    = segTravs

		# Ongoing collection records/metadata for this iteration
		self.SegmentTravsDone  = 0
		self.RootNodeRolls     = 0
		self.K_Phase_Complete  = 0
		self.nSolvedPositions  = 0
		self.nSolvedSubgames   = 0
		self.nCollectedSamples = 0
		self.SegmentStart      = 0
		self.TraversalDuration = 0
		self.AvgTravTimeIso    = 0
		self.AdvCalcTime       = 0
		self.SegmentDuration   = 0
		self.TreeComplexity    = 0
		self.nNodesSeen        = pyarr( ARR_TMPLT_I, self.nPlayers+2, zero=True )
		self.ASizes            = pyarr( ARR_TMPLT_f, self.nPlayers+1, zero=True )
		self.ExplorationDepths = pyarr( ARR_TMPLT_I, NUM_ROUNDS, zero=True )

		cdef str advFile = f"/p{POVplayer}advs_P{self.RANK_P}S{self.RANK_S}.pickle"
		cdef str recFile = f"/segrecords_P{self.RANK_P}S{self.RANK_S}.pickle"

		self.AdvFile    = dataDir + "/segadvs" + advFile # where this worker stores computed targets
		self.RecordFile = dataDir + "/segrecs" + recFile # where this worker stores its metadata

		# GameTree information
		self.zKeys             = vector_ll()
		self.FullyExploredKeys = vector_ll()
		self.SolvableKeys      = vector_ll()
		self.SolvableSubgames  = vector_ll()
		self.GTree             = GameTree()

		# "Anyone who attempts to generate random numbers by deterministic means is, of course, living in a state of sin"
		# For posterity: This looks insane, but all I'm doing is using a slow pyRNG to seed our faster internal C RNG 
		self.pyRNG = np.random.default_rng()
		cdef uint rngseed = <uint>self.pyRNG.integers( low=0,high=4294967295 )
		SEEDRNG( rngseed )

	cdef GTNode     at( self, ll nodeKey ): #noexcept:
		return self.GTree.node_at( nodeKey )


	# ----- COLLECTION PHASE OPS -----------------------------------------------


	cdef bint       Has_Encountered( self, ll nKey ): #noexcept:
		return self.GTree.Contains( nKey )

	# Upon encountering a new node, add it to the subnodes of every path node above it in the tree
	cdef void       extend_subgames( self, GTNode new_subnode ): #noexcept:

		cdef:
			ll        subKey    = new_subnode.Key, superKey
			vector_ll superKeys = new_subnode.PathKeys()
			uint      pathLen   = superKeys.size, s

		for s from 0 <= s < pathLen:
			superKey = superKeys.at( s )
			if (superKey != subKey) and self.GTree.Contains( superKey ):
				self.at( superKey ).SubKeys.append( subKey )

	cdef bint       Node_Has_Unexplored_Actions( self, ll nKey ): #noexcept:
		return self.at( nKey ).RemainingAInds.size > 0

	# Called as soon as we explore all actions at a given node
	cdef void       node_fully_explored( self, ll nKey ): #noexcept:
		self.at( nKey ).exploration_completed()
		self.FullyExploredKeys.append( nKey )

	# Useful later when we're setting up paths to properly calculate reach probabilities
	cdef bint       Dealer_Node_Is_Parent_Of( self, GTNode N ): #noexcept:
		if N.GTParentKey == 0:
			return FALSE
		# N's GTparent lacks a direct connection to N iff ∃ ≥1 deal node between them
		else:
			return not self.at( N.GTParentKey ).Has_Direct_Connection_To( N.Key )

	# Another useful path-setup function for properly calculating reach probabilities 
	cdef ll         find_deal_sequence_start( self, ll above_key ): #noexcept:

		cdef gamenode prevNode = self.at( above_key ).GameNode.ParentNode( Include_Deals=TRUE ), dealNode

		# Step backward through hist starting @ above_key until nondeal node found
		while prevNode.ActingPlayer() == DEALER:
			dealNode = prevNode
			prevNode = prevNode.ParentNode( Include_Deals=TRUE )

		return dealNode.GTKey() # last dealnode found during reverse stepping is start of sequence

	# Edits the connection graph for pathNode's GTParent to skip a deal node between it and pathNode.
	# We want to set up paths to only include player nodes to calc reach probs excluding dealer rng.
	cdef void       remove_parent_dealer_node( self, GTNode pathNode ): #noexcept:

		cdef ll pathKey = pathNode.Key, skipKey = self.find_deal_sequence_start( above_key=pathKey )
		self.at( pathNode.GTParentKey ).skip_dealer_connection( skip_deal_key=skipKey, skip_to_key=pathKey )

	cdef flt1       NodesPerTraversal( self ): #noexcept:

		cdef uint p
		cdef flt1 nodesPerTrav = pyarr( ARR_TMPLT_f, self.nPlayers+2, zero=False )

		for p from 0 <= p <= self.nPlayers+1: 
			nodesPerTrav[ p ] = self.nNodesSeen[ p ] / self.SegmentTravsDone

		return nodesPerTrav

	cdef void     __print_collection_progress( self ): #noexcept:

		cdef:
			double kDur, kAvg
			uint   kDone  = self.SegmentTravsDone,                                                                     \
				   kReq   = self.SegmentTravReq,                                                                       \
				   nRolls = self.RootNodeRolls,                                                                        \
				   s1     = self.CurrentStacks[ 1 ],                                                                   \
				   s2     = self.CurrentStacks[ 2 ],                                                                   \
				   sb     = self.SBlind,                                                                               \
				   rDur

		if ((kDone % 10)==0) or (kDone==kReq):
			if (kDone % 100)==0:
				if (kDone % 500)==0: 
					clear_GPU_cache() # just housekeeping to reduce mem fragmentation

				kDur = TimeNow() - self.SegmentStart
				kAvg = kDur/kDone
				rDur = <uint>((kReq - kDone) * kAvg)

				print( LINE_UP*4,  end='\r' )
				print( LINE_CLEAR, end='\r' )
				print( f"Root-level collection calls: {nRolls} (current stacks: {s1}|{s2}, sb={sb})          " )

				print( LINE_CLEAR, end='\r' )
				print( f"Traversal time running avg:  {kAvg:.5f}s                                            " )

				print( LINE_CLEAR, end='\r' )
				print( f"Elapsed collection duration: {HMS( kDur )}                                          " )

				print( LINE_CLEAR, end='\r' )
				print( f"Traversal ETR:               {HMS( rDur )}                                          " )

			print( PB( kDone,kReq )+f" ({kDone}/{kReq})" + (' '*10),end='\r' )

	# Called each time a traversal completes
	cdef void       trav_completed( self ): #noexcept:
		self.SegmentTravsDone += 1
		self.__print_collection_progress()
		if self.SegmentTravsDone >= self.SegmentTravReq:
			self.K_Phase_Complete = TRUE

	cdef void     __accumulate_path_node_count( self, GTNode zNode ): #noexcept:

		cdef uint1 actingPts = zNode.count_path_action_points()
		cdef uint  p
		for p from 0 <= p <= self.nPlayers+1: 
			self.nNodesSeen[ p ] += actingPts[ p ]

	# Appends endgame z to Z[n] ∀ n∈𝓟(z) & edits GT connections along 𝓟(z) to skip dealer nodes
	cdef void     __complete_terminal_initialization( self, GTNode zNode ): #noexcept:

		cdef:
			ll     zKey    = zNode.Key, pathKey
			uint   pathLen = zNode.TerminalPath.size, s
			GTNode pathNode

		self.trav_completed()
		self.zKeys.append( zKey )
		#self.__accumulate_path_node_count( zNode )

		for s from 0 <= s < pathLen:
			pathKey  = zNode.TerminalPath.at( s )
			pathNode = self.at( pathKey )

			if self.Dealer_Node_Is_Parent_Of( pathNode ): 
				self.remove_parent_dealer_node( pathNode )

			pathNode.append_reachable_endgame( zKey, zNode.uz )

	# Just initializes a game tree entry for a newly encountered node
	cdef void       initialize_entry( self, gamenode n, bint Is_Terminal ): #noexcept:

		cdef GTNode newGTNode = GTNode( self.POVplayer, n, Is_Terminal, estimatorRank=self.RANK_P )
		cdef uint   aPlayer   = newGTNode.ActingPlayer

		self.nNodesSeen[ aPlayer ] += 1
		self.nNodesSeen[ n.PLAYER_COUNT+1 ] += 1
		self.GTree.append( newGTNode )
		self.extend_subgames( new_subnode=newGTNode )

		if Is_Terminal: 
			self.__complete_terminal_initialization( newGTNode )
		else:			
			self.ASizes[ aPlayer ] += newGTNode.A.size

	# Just randomly samples an action index according to probabilities from_strategy
	# TODO: We should really have some kind of debug assert here that σ sums to 1.0
	cdef int        ActionSample( self, flt1 from_strategy ): #noexcept:

		cdef int   nA = from_strategy.shape[0], a
		cdef float u = RNG()/(<float>RAND_MAX + 1.0) # +1.0 guarantees u ∈ [0,1) instead of [0,1]

		for a from 0 <= a < nA:
			u -= from_strategy[ a ]
			if u <= 0: 
				return a

		return nA-1 # fallback to protect against weird stuff that can happen due to float noise

	# Gets the next unexplored action index at for_key
	cdef uint     __next_unexplored_index( self, ll for_key ): #noexcept:
		return <uint>(self.at( for_key ).RemainingAInds.pop())

	# Successor node resulting from taking the next unexplored action idx at node for_key
	# Option to get a specific successor using arg aIdx
	cdef gamenode   NextSuccessor( self, ll for_key, int aIdx=-1 ): #noexcept:

		cdef:
			GTNode    currentGTNode = self.at( for_key )
			uint      actionIdx     = self.__next_unexplored_index( for_key ) if aIdx==-1 else <uint>aIdx
			gameevent action        = currentGTNode.A.at( actionIdx )

		return currentGTNode.GameNode.Successor( action )

	cdef void       Collect( self, gamenode currentNode ): #noexcept:

		cdef:
			uint      aPlayer, aIdx
			ll        nKey
			flt1      oStrat
			gameevent dealEvent
			gamenode  nextNode

		if not self.K_Phase_Complete:
			aPlayer = currentNode.ActingPlayer()

			# Endgame state: record payoff and path info for target calculation
			if currentNode.Is_Terminal():
				self.initialize_entry( currentNode, Is_Terminal=TRUE )

			# POVplayer state: explore all a∈A to find all reachable endgames
			elif aPlayer==self.POVplayer:
				nKey = currentNode.GTKey()
				# Initialize new gametree node if we haven't been here before
				if not self.Has_Encountered( nKey ): self.initialize_entry( currentNode, Is_Terminal=FALSE )

				# Continue exploring a∈A as long as they exist and we haven't hit traversal limit
				while (self.Node_Has_Unexplored_Actions( nKey ) and (not self.K_Phase_Complete)):
					nextNode = self.NextSuccessor( for_key=nKey )
					self.Collect( nextNode )

				# Once we've explored all of A, mark node exploration as complete
				if ((not self.Node_Has_Unexplored_Actions( nKey )) and (not self.K_Phase_Complete)):
					self.node_fully_explored( nKey )

			# Opponent state: sample one action according to their strategy
			elif Are_Opponents( aPlayer, self.POVplayer ):
				nKey = currentNode.GTKey()
				# Initialize new gametree node if we haven't been here before
				if not self.Has_Encountered( nKey ): self.initialize_entry( currentNode, Is_Terminal=FALSE )

				oStrat = ESTIMATOR.Strat( self.at( nKey ).I_ap, GPUrank=self.RANK_P )
				aIdx   = self.ActionSample( from_strategy=oStrat )

				nextNode = self.NextSuccessor( for_key=nKey, aIdx=aIdx )
				self.node_fully_explored( nKey ) #We only ever explore 1 action at opponent nodes
				self.Collect( nextNode )

			# Dealer state: generate deal event & continue traversal, no collector entry needed
			elif aPlayer==DEALER:
				self.nNodesSeen[ DEALER ]+=1; self.nNodesSeen[ currentNode.PLAYER_COUNT+1 ]+=1
				dealEvent = currentNode.Deal()
				nextNode  = currentNode.Successor( dealEvent )
				self.Collect( nextNode )

	# Called when traversal requirement satisfied, records metadata about traversal and performance
	cdef void       K_phase_completed( self ): #noexcept:

		cdef:
			uint   t      = self.SolvingIter, K = self.SegmentTravReq, p
			double kTime  = TimeNow()-self.SegmentStart, n
			flt1   nNodes = self.NodesPerTraversal()

		# Average out |A| per node ∀p
		for p from 1 <= p <= self.nPlayers: self.ASizes[ p ] /= self.nNodesSeen[ p ]
		# We define gametree complexity as num seen nodes * avg branching factor (i.e. avg nActions)
		for p from 1 <= p <= self.nPlayers: self.TreeComplexity += <float>(self.nNodesSeen[ p ]) * self.ASizes[ p ]
		self.TreeComplexity   /= self.SegmentTravsDone
		self.TraversalDuration = <uint>kTime
		self.AvgTravTimeIso    = kTime / self.SegmentTravsDone

		print( '\n\n'+(f"="*50) )
		print( f"ITER {t} SEGMENT TRAVERSAL PHASE COMPLETE".center(50) )
		print( f"="*50 )
		print( f"Time taken: {HMS( kTime )} ( {self.AvgTravTimeIso:.5f}s/trav iso )" )
		print( f"Nodes seen: {list( self.nNodesSeen )} ( avg {[ round( n,3 ) for n in nNodes ]}/trav )" )

	cdef void     __reroll_stacks( self ): #noexcept:
		self.CurrentStacks[ 1 ] = <uint>self.pyRNG.normal( loc=self.GameSize, scale=self.StackDeviation )
		self.CurrentStacks[ 2 ] = <uint>self.pyRNG.normal( loc=self.GameSize, scale=self.StackDeviation )
		#self.CurrentStacks[:] = self.GameSize #NONRANDOM STACKS FOR MORE CONSISTENT RUNTIME BENCHMARKING

	cdef gamenode __reroll_root_node( self ): #noexcept:

		#if self.RootNodeRolls==0: self.__reroll_stacks() #DEBUG: Get random stacks on first trav, never re-roll
		self.__reroll_stacks()

		cdef uint  bpos   = (RNG()%self.nPlayers)+1
		cdef uint1 iConds = cyarr( (NUM_ICONDS,), UINTSIZE, 'I' )
		iConds[ NPLR ]    = self.nPlayers
		iConds[ BPOS ]    = bpos
		iConds[ SBAM ]    = self.SBlind
		iConds[ STK: ]    = self.CurrentStacks

		self.RootNodeRolls+=1
		return RootNode( initialConditions=iConds )

	# Executes data collection via DFS gametree traversal.
	# Generate a root node, explore tree under it, repeat until traversal requirement satisfied.
	cdef void       Traverse( self ): #noexcept:

		print( ('\n'*2)+('='*50) )
		print( f"CONDUCTING ITER {self.SolvingIter} TREE TRAVERSAL; POV PLAYER = {self.POVplayer}".center(50) )
		print( ('='*50)+('\n'*4) )

		cdef gamenode rootNode
		self.SegmentStart = TimeNow()

		while not self.K_Phase_Complete:
			rootNode = self.__reroll_root_node()
			self.Collect( currentNode=rootNode )
			
		self.K_phase_completed()

	
	# ----- CALC PHASE OPS -----------------------------------------------------


		# ----- NOTATION -----------------------------------
		# 	𝓦(x,y) = ConnectionGraph weights from x to y, shape (1326,T)
		# 	Pl(x) = Acting player @ x
		# 	𝓟(x) = GT path to x, 𝓟(x)y+ = direct successor to y in 𝓟(x)
		# 	𝓹(x) = GTParent(x), POV𝓹(x) = first node n above x:Pl(n)=POV
		# 	πₒ(x) = Opp-only counterfactual reach prob for x, shape (1326,T)
		# 	π(x,y) = fwd reach prob from x to y, shape (1326,T)
		# 	𝓝ₑ = Set of all fully-explored nodes
		# 	𝓝ₛ = Set of all solvable nodes (⊆ 𝓝ₑ)
		# 	Solvable node := node n∈𝓝ₑ: s∈𝓝ₑ ∀ n∈S(n)
		# 	Sᵣ = Root of subgame S
		# 	𝓢ₛ = Set of all solvable subgames (partitions 𝓝ₛ)
		# 	Sₛ = Solvable subgame ∈ 𝓢ₛ
		# 	Solvable subgame := Subgame S:[(POV𝓹(Sᵣ)∉𝓝ₛ) & (n∈𝓝ₛ ∀n∈S)]

		# ----- ARR AXES -----------------------------------
		# 	N.ConnectionGraph: axes=(aIdx, handIdx, tIdx) aka (ACTIONS, HISTORIES, ITERS), shape=(|N.A|,1326,T)
		# 		if N.ActingPlayer=POV: CGraph[a,h₁,t] = CGraph[a,h₂,t] since h₁=h₂ ∀h₁,h₂ 
		# 		if N.ActingPlayer=opp: CGraph[a,h,t] = σᵗₒ(Iₒ(h),a) wherever h=idx of a possible opp hand, 0 elsewhere
		# 	N.CFReaches:  axes=(oppHandIdx, tIdx) aka (HISTORIES, ITERS), shape=(1326,T)
		# 	N.FwdReaches: axes=(oppHandIdx, tIdx) aka (HISTORIES, ITERS), shape=(1326,T)
		# 		N.[CF/Fwd]Reaches[ h,t ] = reaches derived using σᵗₒ(Iₒ(h)) wherever opp probabilities are required
		# 		As above, these will be 0 wherever h is not an index of a possible opp hand


	# When finding solvable subgames, useful to be able to get the most recent POVplayer path node
	cdef ll         PrecedingPNodeKey( self, ll nKey ): #noexcept:

		cdef GTNode currentNode = self.at( nKey )
		if currentNode.GTParentKey == 0: # case where path is only 1 node long
			return 0

		cdef GTNode prevNode = self.at( currentNode.GTParentKey )

		# Step up GT path from current node until a POVplayer node is found, then return its key
		while prevNode.ActingPlayer != self.POVplayer:
			if prevNode.GTParentKey == 0: # found a root node before a pov node
				return 0
			prevNode = self.at( prevNode.GTParentKey )

		return prevNode.Key

	# For a subgame starting at nKey to be solvable, all nodes under n must be fully explored
	cdef bint       All_SubNodes_Explored( self, ll nKey ): #noexcept:
		
		cdef:
			vector_ll S = self.at( nKey ).SubKeys # the subgame in question
			uint      s
			GTNode    subNode
			
		for s from 1 <= s <= S.size:
			subNode = self.at( S.at( s-1 ) )
			if (subNode.ActingPlayer==self.POVplayer) and (not subNode.Fully_Explored): 
				return FALSE

		return TRUE

	# Solvable node: a node at the top of a subtree whose nodes are all fully explored
	cdef void       find_solvable_nodes( self ): #noexcept:

		cdef uint nExploredNodes = self.FullyExploredKeys.size, k
		cdef ll   exploredKey

		print( f"\n\tFinding all solvable nodes in set of {nExploredNodes} fully explored nodes..." )
		
		# For every fully explored node, check whether all of its subnodes have been explored
		for k from 1 <= k <= nExploredNodes:
			exploredKey = self.FullyExploredKeys.at( k-1 )
			
			if self.All_SubNodes_Explored( exploredKey ):
				self.at( exploredKey ).Solvable = TRUE
				self.SolvableKeys.append( exploredKey )

		print( f"\t{self.SolvableKeys.size} solvable nodes found, proceeding to solvable subgame search..." )

	# Finds all solvable subgame root nodes; i.e. all solvable nodes with no solvable predecessors
	cdef void       find_solvable_subgames( self ): #noexcept:

		cdef uint nSolvableKeys = self.SolvableKeys.size, k
		cdef ll   rootKey, parentKey
		
		print( f"\n\tFinding solvable Sᵣ in set of {nSolvableKeys} solvable nodes..." )

		for k from 1 <= k <= nSolvableKeys:
			rootKey   = self.SolvableKeys.at( k-1 )
			parentKey = self.PrecedingPNodeKey( rootKey )

			# If node has no solvable parent, it's the root of a solvable subgame
			if (parentKey==0) or (not self.at( parentKey ).Solvable): 
				self.SolvableSubgames.append( rootKey )

		print( f"\t{self.SolvableSubgames.size} solvable subgames found, proceeding to set counterfactual payoffs..." )

	# Find 𝓝ₛ⊆𝓝ₑ ⟶ Find 𝓢ₛ⊆𝓝ₛ ⟶ set CFU(z) ∀ z∈Z[Sₛ] ∀ Sₛ∈𝓢ₛ ⟶ init n.ZMap ∀ n∈𝓝ₛ ⟶ path-prop CFU(z) ∀ z∈Z[Sₛ] ∀ Sₛ∈𝓢ₛ
	# Basically, finds all solvable paths and transmits along those paths the endgame data we need for calculating advs
	cdef void       build_solvable_paths( self ): #noexcept:

		print( '\n'+("="*50) )		
		print( f"FINDING SOLVABLE PATHS".center(50) )
		print( "="*50 )

		# TODO: You should prob just turn these timing ops into context managers
		cdef double pathStart = TimeNow()
		
		self.find_solvable_nodes()
		self.find_solvable_subgames()
		
		cdef double pathTime = TimeNow() - pathStart
		cdef uint K = self.SegmentTravsDone

		print( '\n'+("="*50) )		
		print( f"ALL SOLVABLE PATHS FOUND".center(50) )
		print( "="*50 )
		print( f"Time taken: {pathTime:.3f}sec" )
		print( f"|𝓢ₛ| = {self.SolvableSubgames.size}" )
		print( f"|𝓝ₛ| = {self.SolvableKeys.size}" )

	# Empty-inits forward reach maps for all nodes in all solvable subgames we've found
	cdef void       initialize_fwd_reach_maps( self ): #noexcept:

		print(f"\nInitializing πfwd maps ∀ n∈Sₛ ∀ Sₛ∈𝓢ₛ...")

		cdef:
			uint      nSS = self.SolvableSubgames.size, nNodes=0, nInit=0, s, ss
			ll        Sr, subKey
			vector_ll S
			double    initStart = TimeNow()

		for s from 1 <= s <= nSS:
			Sr = self.SolvableSubgames.at( s-1 )
			S  = self.at( Sr ).SubKeys
			nNodes += (S.size+1) # +1 because Sᵣ ∉ SubKeys

		for s from 1 <= s <= nSS:
			Sr = self.SolvableSubgames.at( s-1 )
			S  = self.at( Sr ).SubKeys
			self.at( Sr ).initialize_fwd_reaches() # again, Sᵣ ∉ SubKeys

			for ss from 1 <= ss <= S.size:
				subKey = S.at( ss-1 )
				self.at( subKey ).initialize_fwd_reaches()
				nInit += 1
				print( PB( nInit, nNodes ) + f" {nInit}/{nNodes}", end='\r' )

		cdef double initTime = TimeNow() - initStart
		print( f"\n{nNodes} πfwd maps successfully initialized, time taken: {initTime:.3f}sec" )

	# Just nice syntax for getting the parent GTNode of the node with the given key
	cdef GTNode     GTParent( self, ll of_key ): #noexcept:
		return self.at( self.at( of_key ).GTParentKey )

	# Inductive step for fwd reaches: sets π(𝓹(z),z) ∀z∈𝓩 so we can calc π(n,z) ∀ n∈𝓟(z) inductively
	# ∀z∈𝓩, π(𝓹(z),z) = 𝓦(𝓹(z),z), & since 𝓦 is calculated upon init, ∃ 𝓦(𝓹(z),z) ∀ z∈𝓩 already
	cdef void       set_base_fwd_reaches( self ): #noexcept:

		print(f"\nCalculating π(𝓹(z),z) ∀ z∈𝓩...")

		cdef:
			uint   nZ = self.zKeys.size, z
			ll     zKey
			double zTime, zStart = TimeNow()
			flt2   zReach
			GTNode pz

		# For every endgame we've encountered, set the forward reaches for its parent 
		for z from 1 <= z <= nZ:
			zKey   = self.zKeys.at( z-1 )
			pz     = self.GTParent( of_key=zKey )
			zReach = pz.ConnectionGraph.get_weights( to_key=zKey ) # (1326,T)

			# This happens if pz.ActingPlayer is opp, since we only did this for POV nodes above
			if pz.FwdReaches is None: 
				pz.initialize_fwd_reaches()
				
			pz.FwdReaches.set_weights( to_key=zKey, weights=zReach )
			print( PB( z,nZ ) + f" {z}/{nZ}", end='\r' )

		zTime = TimeNow()-zStart
		print( f"\nπ(𝓹(z),z) calculated ∀ z∈𝓩, time taken: {zTime:.3f}sec" )

	# Just counts number of nodes along specified path where the specified player is acting
	cdef uint     __count_steps( self, uint by_player, vector_ll along_path ): #noexcept:
		
		cdef uint pathLen = along_path.size, pSteps=0, s
		cdef ll   stepKey
		
		for s from 1 <= s <= pathLen:
			stepKey = along_path.at( s-1 )
			if self.at( stepKey ).ActingPlayer==by_player: 
				pSteps+=1

		return pSteps

	# Calculates counterfactual reach prob for subgame root Sr; product of all estimated opp action
	# probabilities along 𝓟(Sᵣ) for all possible opp hands and estimator iterations.
	# πₒ(Sᵣ) = Π( 𝓦(o,n) ) ∀ o,n ∈ 𝓟(S): (Pl(o)=clown & n=𝓟(S)o+)
	cdef matrix_flt RootCFReach( self, ll Sr ): #noexcept:

		cdef:
			vector_ll PS          = self.at( Sr ).PathKeys()
			uint      clown       = OpponentsOf( self.POVplayer )[ 0 ],                                                \
					  oSteps      = self.__count_steps( by_player=clown, along_path=PS ),                              \
					  pathLen     = PS.size,                                                                           \
					  PATHSTEPS   = 0, nH = NUM_POSSIBLE_HANDS, oppStep = 1, step
			flt3      pathWeights = cyarr( (oSteps+1, nH, T), FLTSIZE, 'f' )
			ll        nextKey
			GTNode    stepNode

		pathWeights[0,:,:] = 1
		for step from 0 <= step < pathLen-1:
			stepKey  = PS.at( step )
			stepNode = self.at( stepKey )

			if stepNode.ActingPlayer==clown:
				nextKey = PS.at( step+1 )
				pathWeights[ oppStep,:,: ] = stepNode.ConnectionGraph.get_weights( to_key=nextKey ) # (1326,T)
				oppStep += 1

		cdef flt2 rootCFReaches = NP( pathWeights ).prod( axis=PATHSTEPS ) # (1326,T)
		return matrix_flt( from_view=rootCFReaches ) # (1326,T)

	# To calc πₒ(n) ∀ n∈Sₛ, need πₒ(Sᵣ) as inductive base. ∴ start by calculating πₒ(Sᵣ) ∀ Sₛ∈𝓢ₛ
	cdef void       calculate_base_cfreaches( self ): #noexcept:

		cdef:
			uint   nSS = self.SolvableSubgames.size, s
			ll     Sr
			double cfStart = TimeNow(), cfTime

		print( f"\nCalculating πₒ(Sᵣ) ∀ S∈𝓢ₛ..." )

		# ∀ Sₛ∈𝓢ₛ, calculate the CFReach for its root Sᵣ so we can propagate it down the path later
		for s from 1 <= s <= nSS:
			Sr = self.SolvableSubgames.at( s-1 )
			self.at( Sr ).CFReaches = self.RootCFReach( Sr ) # (1326,T)
			print( PB( s,nSS ) + f" {s}/{nSS}", end='\r' )
			
		cfTime = TimeNow()-cfStart

		print( f"\nπₒ(Sᵣ) ∀ S∈𝓢ₛ successfully determined, time taken: {cfTime:.3f}sec" )

	# Simply returns the multi-hand, multi-iter, opp-action-only reach prob for n
	# πₒ skips POV nodes, ∴ Pl(n)=POV ⇒ (πₒ(n)=πₒ( 𝓹(n) )), & Pl(n)=opp ⇒ πₒ(n)=πₒ(𝓹(n)) * 𝓦(𝓹(n),n)
	cdef matrix_flt CFReach( self, ll nKey ): #noexcept:
		
		cdef:
			flt2   leadingWeights, cfview
			uint   clown      = OpponentsOf( self.POVplayer )[ 0 ]
			GTNode parentNode = self.GTParent( nKey ) 

		# πₒ skips POV nodes, ∴ Pl(n)=POV ⇒ (πₒ(n)=πₒ( 𝓹(n) ))
		if parentNode.ActingPlayer == self.POVplayer:
			return parentNode.CFReaches # (1326,T)

		# Pl(n)=opp ⇒ πₒ(n)=πₒ( 𝓹(n) ) * 𝓦( 𝓹(n),n )
		if parentNode.ActingPlayer == clown:
			leadingWeights = parentNode.ConnectionGraph.get_weights( to_key=nKey )    # (1326,T)
			cfview         = ArrMult2d( parentNode.CFReaches.view(), leadingWeights ) # (1326,T)
			return matrix_flt( from_view=cfview ) # (1326,T)

	# Inductively calculates πₒ(n) ∀n ∈ {𝓟(z)∩S} by propagating πₒ(Sᵣ) which we calc'd above
	cdef void       accumulate_zpath_cfreaches( self, ll Sr, ll zKey ): #noexcept:
		
		cdef:
			vector_ll Pz   = self.at( zKey ).TerminalPath # 𝓟(z)
			uint      rIdx = Pz.index_of( Sr ), pathLen, pathStep
			ll1       SPz  = Pz.view()[ rIdx: ] # 𝓟(z)∩S (i.e. the part of 𝓟(z) inside of S)
			ll        stepKey
			GTNode    stepNode
			
		pathLen = SPz.shape[ 0 ]
		for pathStep from 1 <= pathStep <= pathLen:
			stepKey  = SPz[ pathStep-1 ]
			stepNode = self.at( stepKey )

			# Only calc πₒ(n) if not done already for an overlapping zpath
			if stepNode.CFReaches is None: 
				stepNode.CFReaches = self.CFReach( stepKey ) # (1326,T)

	# Once πₒ(Sᵣ) is calculated, we can inductively calculate πₒ(n) ∀n∈S
	cdef void       calculate_subgame_cfreaches( self, ll Sr ): #noexcept:

		cdef:
			vector_ll ZS = self.at( Sr ).Zn # all endgames in S
			ll        zKey
			uint      nZ = ZS.size, z
			
		# ∀z ∈ Z[S], inductively calculate πₒ(n) ∀n ∈ 𝓟(z)
		for z from 1 <= z <= nZ:
			zKey = ZS.at( z-1 )
			self.accumulate_zpath_cfreaches( Sr,zKey )

	# Iterates through all solvable subgames & inductively calculates πₒ(n) ∀ n∈Sₛ ∀ Sₛ∈𝓢ₛ
	cdef void       calculate_counterfactual_reaches( self ): #noexcept:

		cdef:
			uint   nSS = self.SolvableSubgames.size, s
			double cfStart = TimeNow(), cfTime
			ll     Sr

		print( f"\nCalculating πₒ(n) ∀ n∈Sₛ ∀ Sₛ∈𝓢ₛ..." )

		for s from 1 <= s <= nSS:
			Sr = self.SolvableSubgames.at( s-1 )
			self.calculate_subgame_cfreaches( Sr )
			print( PB( s,nSS ) + f" {s}/{nSS}",end='\r' )

		cfTime = TimeNow() - cfStart
		print( f"\nπₒ ∀ Sₛ∈𝓢ₛ successfully determined, time taken: {cfTime:.3f}sec" )

	# Calculates one step of the backward propagation of fwd reaches from endgames up the tree.
	# Since we're stepping backward through 𝓟(z):
	# vKey = 𝓟(z)fKey+ ⇒ ∃ π(vKey,z) already, ∴ π(fKey,z) = 𝓦(fKey,vKey) * π(vKey,z)
	cdef flt2       TerminalFwdReach( self, ll from_key, ll via_key, ll to_zkey ): #noexcept:
		
		cdef flt2 hereToNext = self.at( from_key ).ConnectionGraph.get_weights( to_key=via_key ),                      \
				  nextToEnd  = self.at( via_key ).FwdReaches.get_weights( to_key=to_zkey )

		return ArrMult2d( hereToNext, nextToEnd ) # (1326,T)

	# Calculates subgame fwd reaches π(n,z) ∀n ∈ 𝓟(z)∩S
	# By now we already have π( 𝓹(z),z ) ∀z (inductive step), so:
	# ∀ z∈Z[S], accumulate π(n,z) ∀ n∈𝓟(z) by stepping backward through 𝓟(z)
	cdef void       accumulate_zpath_fwd_reaches( self, ll Sr, ll zKey ): #noexcept:

		cdef:
			ll     pKey, stepKey
			flt2   fwdReaches
			GTNode zNode   = self.at( zKey ), parentNode
			uint   rIdx    = zNode.TerminalPath.index_of( Sr )
			ll1    SPz     = zNode.TerminalPath.view()[ rIdx: ] # S ∩ 𝓟(z)
			uint   pathLen = SPz.shape[ 0 ], reverseStep

		for reverseStep from pathLen-1 > reverseStep >= 1:
			stepKey    = SPz[ reverseStep ]
			parentNode = self.GTParent( stepKey )
			pKey       = parentNode.Key

			# This happens if 𝓹(z).ActingPlayer is opp, since we only did this for POV nodes above
			if parentNode.FwdReaches is None: 
				parentNode.initialize_fwd_reaches()

			fwdReaches = self.TerminalFwdReach( from_key=pKey, via_key=stepKey, to_zkey=zKey )
			parentNode.FwdReaches.set_weights( to_key=zKey, weights=fwdReaches )

	#∀ z∈Z[S], get 𝓟(z) & calculate π(n,z) ∀ n∈𝓟(z)
	cdef void       calculate_subgame_fwd_reaches( self, ll Sr ): #noexcept:
		
		cdef:
			vector_ll ZS = self.at( Sr ).Zn # Z[S]
			ll        zKey
			uint      nZ = ZS.size, z

		for z from 1 <= z <= nZ:
			zKey = ZS.at( z-1 )
			self.accumulate_zpath_fwd_reaches( Sr,zKey )

	# ∀ solvable subgame S, inductively back-propagates fwd reaches along all terminal paths in S
	cdef void       calculate_fwd_reaches( self ): #noexcept:

		cdef:
			uint   nSS = self.SolvableSubgames.size, s
			ll     Sr
			double fwdStart = TimeNow(), fwdTime

		print( f"\nCalculating π(n,z) ∀ n,z∈Sₛ ∀ Sₛ∈𝓢ₛ..." )

		for s from 1 <= s <= nSS:
			Sr = self.SolvableSubgames.at( s-1 )
			self.calculate_subgame_fwd_reaches( Sr )
			print( PB( s,nSS ) + f" {s}/{nSS}", end='\r' )

		fwdTime = TimeNow() - fwdStart
		print( f"\nπ(n,z) ∀ n,z∈Sₛ successfully determined ∀ Sₛ∈𝓢ₛ, time taken: {fwdTime:.3f}sec" )

	# Calculates all counterfactual & forward reach probabilities for all solvable subgames
	cdef void       calculate_reaches( self ): #noexcept:

		print( '\n'+("="*50) )
		print( f"CALCULATING REACH PROBABILITIES".center(50) )
		print( "="*50 ) 

		cdef double rStart = TimeNow()

		# First do initial mem allocation and base steps for inductive reach calculation
		self.initialize_fwd_reach_maps() # init πfwd maps for all nodes in all solvable subgames
		self.set_base_fwd_reaches()      # inductive base step for πfwd
		self.calculate_base_cfreaches()  # inductive base step for πₒ

		# Now do the hard crunching - ultimately these are the reaches we need for our nn targets
		self.calculate_counterfactual_reaches()
		self.calculate_fwd_reaches()

		cdef double rTime = TimeNow() - rStart

		print( '\n'+("="*50) )
		print( f"ALL REACHES CALCULATED".center(50) )
		print( "="*50 ) 
		print( f"Time taken: {rTime:.3f}s ( +{(rTime/self.SegmentTravsDone):.5f}s to agg avg kTime )" )

	# Calculates an infoset's reach probability: πₒᵗ(I) = Σ{h∈I}( πₒᵗ(h) )
	cdef flt1       IReach( self, GTNode n ): #noexcept:
		cdef uint HISTORIES=0
		return NP( n.CFReaches.view() ).sum( axis=HISTORIES ) # (T,)

	# Returns array of payoffs from all endgames reachable from n, weighted by their reach probs:
	# πᵗ(h,z) * u(z) ∀ h∈I, z∈Z[I], t<T
	cdef flt3       ReachWeightedPayoffs( self, GTNode n ): #noexcept:
		
		cdef:
			ll   zKey
			uint nZ  = n.Zn.size, nH = NUM_POSSIBLE_HANDS, z
			flt3 piZ = cyarr( (nZ,nH,T), FLTSIZE, 'f' ), uZ3d
			flt1 uZ  = cyarr( (nZ,), FLTSIZE,'f' )
			
		# ∀z∈Z[n], calculate πᵗ(h,z) * CFU(z)
		for z from 1 <= z <= nZ:
			zKey       = n.Zn.at( z-1 )
			uZ[ z-1 ]  = n.UZn.payout_from( zKey )
			piZ[ z-1 ] = n.FwdReaches.get_weights( to_key=zKey ) # (nZ,1326,T) ⟵ (1326,T)

		uZ3d = newaxis( axis=2, a=newaxis( axis=1, a=NP(uZ) ) )  # (nZ,1,1)
		return ArrMult3d( piZ, uZ3d ) # πᵗ(h,z) * u(z) ∀ h∈I, z∈Z[I], t<T; (nZ,1326,T)

	# Returns multi-iter infoset expected values:
	# vᵗ(I) = Σ{h∈I}( πₒᵗ(h) * Σ{z∈Z[I]}( πᵗ(h,z)u(z) ) )
	cdef flt2       NodeValue( self, GTNode n ): #noexcept:
		
		cdef:
			uint ENDGAMES=0, PATHS=0
			flt3 UZWeighted = self.ReachWeightedPayoffs( n )           # πᵗ(h,z)u(z); (|Z[I]|,1326,T) 
			flt2 nExpVal    = NP( UZWeighted ).sum( axis=ENDGAMES )    # Σ{z∈Z[I]}( πᵗ(h,z)u(z) ); (1326,T)
			flt2 vITerms    = ArrMult2d( n.CFReaches.view(), nExpVal ) # (1326,T)
			flt1 vI         = NP( vITerms ).sum( axis=PATHS )          # (T,)

		return newaxis( vI,axis=0 ) # (1,T), extra axis for downstream convenience

	# Returns multi-iter action expected values: 
	# vᵗ(I,a) = Σ{h∈I}( πₒᵗ(h) * Σ{z∈Z[I·a]}( πᵗ(h·a,z)u(z) ) )
	cdef flt2       ActionValues( self, GTNode n ): #noexcept:

		cdef:
			uint nA = n.A.size, nH=NUM_POSSIBLE_HANDS, ENDGAMES=0, PATHS=1, a
			flt3 CFReaches = newaxis( n.CFReaches.view(),axis=0 ) # (1,1326,T)
			flt3 AExpVals  = cyarr( (nA,nH,T), FLTSIZE, 'f' ), aUZWeighted, vIATerms
			flt2 aExpVal
			ll   aKey

		for a from 1 <= a <= nA:
			aKey            = n.SuccessorKeys.at( a-1 )
			aNode           = self.at( aKey ) # node arrived at by doing action a
			aUZWeighted     = self.ReachWeightedPayoffs( aNode )     # πᵗ(h·a,z)u(z); (|Z[I·a]|, 1326, T)
			aExpVal         = NP( aUZWeighted ).sum( axis=ENDGAMES ) # Σ{z}( πᵗ(h·a,z)u(z) ); (1326,T)
			AExpVals[ a-1 ] = aExpVal

		vIATerms = ArrMult3d( CFReaches, AExpVals ) # (|A|,1326,T)
		return NP( vIATerms ).sum( axis=PATHS )     # (|A|,T)

	# The final step in deriving AdvNet targets - calc action advantages under current strat:
	# α(I,a) = Σ{t<T}( t * πₒᵗ(I) * (vᵗ(I-a)-vᵗ(I)) ) / Σ{t<T}( t * πₒᵗ(I) )
	cdef flt1       ActionAdvs( self, GTNode POVnode ): #noexcept:

		cdef:
			uint  ITERS=1
			flt2  tRange             = newaxis( NP( arange( T ) ),axis=0 )                   # (1,T)
			flt2  Ireach             = newaxis( self.IReach( POVnode ),axis=0 )              # (1,T)
			flt2  linReachTerms      = ArrMult2d( tRange, Ireach ) # Retains 0 reaches       # (1,T)
			flt2  reachDenom         = Unzero2d( Ireach ) # Avoid 0div errs; undone below    # (1,T)
			flt2  vIA                = ArrDiv2d( self.ActionValues( POVnode ), reachDenom )  # (|A|,T)
			flt2  vI                 = ArrDiv2d( self.NodeValue( POVnode ), reachDenom )     # (1,T)
			flt2  CFRegrets          = ArrSub2d( vIA, vI )                                   # (|A|,T)
			flt2  linRegretTerms     = ArrMult2d( linReachTerms, CFRegrets ) # Undoes unzero # (|A|,T)
			flt1  totalLinearRegrets = sumaxis( linRegretTerms, axis=ITERS, dtype=f32 )      # (|A|,)
			float totalLinearReach   = sumaxis( linReachTerms[0], dtype=f32 )
			uint  nA                 = vIA.shape[ 0 ]
			flt1  aAdvs              = cyarr( (nA,), FLTSIZE, 'f' )

		# Finally, these are our AdvNet targets
		if totalLinearReach!=0:
			aAdvs = NP( totalLinearRegrets,dtype=f32 ) / totalLinearReach 
		else:
			aAdvs[:] = 0	

		return aAdvs # (|A|,)

	# Uses reach probs calculated earlier to derive AdvNet targets ∀n∈S
	# Targets are saved to disk for later access by pre-training-phase data management
	cdef void       calculate_subgame_targets( self, GTNode Sr ): #noexcept:

		cdef:
			uint      s, nS
			int1      aInds
			flt1      advIA
			ll        subKey
			GTNode    subNode
			advmap    Isamples
			vector_ll S = Sr.SubKeys

		nS = S.size
		for s from 1 <= s <= nS:
			subKey  = S.at( s-1 )
			subNode = self.at( subKey )

			#TODO: Def of aInds here only works if not eliminating actions via strat accumulation
			if subNode.ActingPlayer == self.POVplayer:
				advIA = self.ActionAdvs( subNode )
				aInds = np.arange( subNode.FullAInds.size,dtype=intc )
				Iadvs = advmap( for_iter=T, I=subNode.I_tp, aInds=aInds, aTargets=advIA )
				Iadvs.save_sample_dicts( to_file=self.AdvFile )
				self.nCollectedSamples += advIA.shape[ 0 ]
				self.nSolvedPositions += 1

	# Orchestrates final calculation of AdvNet targets from collected game trajectory data.
	# Finds solvable game paths, calculates reach probabilities, derives advs, and saves them.
	cdef void       Calculate_Targets( self ): #noexcept:

		print( '\n\n'+("="*100) )		
		print( f"CALCULATING αNET TARGETS".center(100) )
		print( "="*100 )
		cdef double aStart = TimeNow()

		# First find all solvable paths & set all data necessary for target calculation
		self.build_solvable_paths()
		
		# Probability & payout arrays all path-aligned now, so accumulate reaches along those paths
		self.calculate_reaches()

		cdef:
			ll     rKey
			GTNode Sr
			uint   nSS    = self.SolvableSubgames.size, s
			double tStart = TimeNow()

		print( '\n'+("="*50) )
		print( f"DERIVING SUBGAME αTARGETS FROM REACH PROBABILITIES".center(50) )
		print( "="*50 )

		# ∀ S∈𝓢ₛ, use accumulated reach probs & collected payouts to calc α(I(n),A) ∀ n∈S
		for s from 1 <= s <= nSS:

			rKey = self.SolvableSubgames.at( s-1 )
			Sr   = self.at( rKey )
			self.calculate_subgame_targets( Sr )
			
			self.nSolvedSubgames += 1
			print( PB( s,nSS ) + f" {s}/{nSS}", end='\r' )

		cdef double tTime = TimeNow() - tStart, aTime = TimeNow() - aStart
		self.AdvCalcTime  = aTime
		print( f"\nαTargets derived successfully, time taken: {tTime:.3f}s" )

		print( '\n\n'+(f"="*100) )
		print( f"ITERATION {self.SolvingIter} αTARGETS DETERMINED AND SAVED".center(100) )
		print( f"="*100 )
		print( f"|𝓢ₛ| = {nSS}" )
		print( f"α samples saved to: {self.AdvFile}" )
		print( f"Targets collected:  {self.nCollectedSamples}" )
		print( f"Target calc time:   {aTime:.3f}s ( +{(aTime/self.SegmentTravsDone):.5f}s to agg avg kTime )" )
		print( f"Iso avg trav time:  {self.AvgTravTimeIso:.7f}s" )


	# ----- SEGMENT LOGGING OPS ------------------------------------------------
	# Allows separate CFRCollectors to report metadata about segmented collection runs


	# Records number of fully explored nodes for each betting round
	cdef void       find_exploration_depths( self ): #noexcept:
		
		cdef:
			uint     nS = self.FullyExploredKeys.size, s, sDepth
			ll       exploredKey
			gamenode exploredNode

		for s from 1 <= s <= nS:
			exploredKey  = self.FullyExploredKeys.at( s-1 )
			exploredNode = self.at( exploredKey ).GameNode
			sDepth       = exploredNode.CurrentStreet()
			self.ExplorationDepths[ sDepth ] += 1

	# Just returns a dict of relevant collection segment metrics
	cdef dict       SegmentRecords( self ): #noexcept:

		cdef dict segDict = {}
		segDict[ 'POVplayer' ]         = self.POVplayer
		segDict[ 'SegmentTravReq' ]    = self.SegmentTravReq
		segDict[ 'SegmentTravsDone' ]  = self.SegmentTravsDone
		segDict[ 'nNodesSeen' ]        = list( self.nNodesSeen )
		segDict[ 'nSolvedPositions' ]  = self.nSolvedPositions
		segDict[ 'nSolvedSubgames' ]   = self.nSolvedSubgames
		segDict[ 'nCollectedSamples' ] = self.nCollectedSamples
		segDict[ 'TraversalDuration' ] = self.TraversalDuration
		segDict[ 'AvgTravTimeIso' ]    = self.AvgTravTimeIso
		segDict[ 'AdvCalcTime' ]       = self.AdvCalcTime
		segDict[ 'SegmentDuration' ]   = self.SegmentDuration
		segDict[ 'ExplorationDepths' ] = list( self.ExplorationDepths )
		segDict[ 'TreeComplexity' ]    = self.TreeComplexity
		return segDict

	cdef void       save_records( self ): #noexcept:
		cdef dict segDict = self.SegmentRecords()
		with open( self.RecordFile,'wb' ) as segrecFile: 
			pickle.dump( segDict, segrecFile, protocol=-1 )

	# Just some info that helps us see some relevant exploration/collection behaviour
	cdef void       print_solvable_subgame_info( self ):

		cdef:
			vector_int uAvgs = vector_int()
			uint       nSS   = self.SolvableSubgames.size, s
			ll         Sr, z
			ZMap       uZS

		for s from 0 <= s < nSS:
			Sr  = self.SolvableSubgames.at( s )
			uZS = self.at( Sr ).UZn 
			uAvgs.append( uZS.UnweightedMeanU() )
			
			print( f"\n\nREACHABLE PAYOFF INFO FOR S WITH Sᵣ: (key={Sr})" )
			self.at( Sr ).GameNode.summary( Compact=TRUE )
			uZS.summary()

		print( f"\nALL SUBGAME UNWEIGHTED AVG PAYOFFS:" )
		print( list( uAvgs.view() ) )
		print( f"OVERALL u( Z[𝓢ₛ] ) UNWEIGHTED AVERAGE: {NP(uAvgs.view()).mean()}" )

	# Called at completion of a collection segment for housekeeping: logging, summary printing, etc
	cdef void       Collection_Segment_Completed( self ): #noexcept:

		cdef:
			uint   nSS   = self.SolvableSubgames.size,                                                                 \
			       nEP   = self.FullyExploredKeys.size,                                                                \
			       nCS   = self.nCollectedSamples, d, n
			double cTime = TimeNow()-self.SegmentStart, A
			list   nSeen = [ str( n ) for n in self.nNodesSeen ],                                                      \
				   avgnA = [ str( round(A,3) ) for A in self.ASizes ]
				   
		nSeen[ self.POVplayer ] = '*' + nSeen[ self.POVplayer ] + '*'
		avgnA[ self.POVplayer ] = '*' + avgnA[ self.POVplayer ] + '*'

		self.SegmentDuration = <uint>cTime
		self.find_exploration_depths()
		self.save_records()

		print( '\n\n'+(f'='*100) )
		print( f"ITERATION {self.SolvingIter} COLLECTION SEGMENT COMPLETE".center(100) )
		print( f'='*100 )
		print( f"Segment total time taken:  {HMS( cTime )}" )
		print( f"Segment iso avg trav time: {self.AvgTravTimeIso:.7f}s" )
		print( f"Segment target calc time:  {self.AdvCalcTime:.3f}s" )
		print( f"Segment total nodes seen:  [ {' | '.join( nSeen )} ]" )
		print( f"Segment avg |A| per node:  [ {' | '.join( avgnA[ 1:] )} ]" )
		print( f"Segment tree complexity:   {self.TreeComplexity:.3f}" )
		print( f"SEGMENT MDATA SAVED TO:    {self.RecordFile}\n" )

		#self.print_solvable_subgame_info()

# The very last step after an iter's training phase completes is destruction of multi-worker temp
# data. If any still exists, we know we're not ready to start this iter's collection run yet.
cdef void __await_prev_iter_completion( str advDir, str recDir ): #noexcept:
	
	cdef:
		list   tempAdvFiles   = [ p.name for p in Path( advDir ).iterdir() ] # num temp files currently present
		bint   Prev_Iter_Done = len( tempAdvFiles )==0
		uint   SCAN_INTERVAL  = 5 # num seconds to wait before re-scanning temp data dir
		double startTime      = TimeNow(), waitTime

	if Prev_Iter_Done: 
		return

	print( '\n'+(f'='*100) )
	print( f"SEGMENTED ADV FILES PRESENT, AWAITING ITER START SIGNAL".center(100) )
	print( f'='*100+'\n\n' )

	while not Prev_Iter_Done:
		sleep( SCAN_INTERVAL )

		tempAdvFiles   = [ p.name for p in Path( advDir ).iterdir() ]
		Prev_Iter_Done = len( tempAdvFiles )==0
		waitTime       = TimeNow() - startTime

		print( LINE_UP*2, end='\r' )

		print( LINE_CLEAR, end='\r' )
		print( f"MONITORING DIRECTORY: {recDir}" )

		print( LINE_CLEAR, end='\r' )
		print( f"ELAPSED WAIT TIME:    {HMS( waitTime )}" )

	print( '\n\n'+(f'='*100) )
	print( f"PREVIOUS ITERATION COMPLETED, COMMENCING NEXT CFR ITER".center(100) )
	print( f'='*100 )

# Executes this worker's data collection segment: traverse gametree, derive targets, and save.
cdef void _Do_Collection_Segment( str dataDir, int pRank, int sRank, int mSize, int gameSize, int nPlayers, int travs ): #noexcept:

	cdef str advDir    = dataDir + "/segadvs",                                                                         \
			 recDir    = dataDir + "/segrecs",                                                                         \
			 metaFile  = dataDir + "/metadata.pickle",                                                                 \
			 modelFile = dataDir + "/models.pickle"

	# If doing multiple serial segments, segmented data being present doesn't imply prev iter ongoing
	if sRank==0: 
		__await_prev_iter_completion( advDir, recDir )

	cdef:
		uint         tDone     = get_presolved_iters( metaFile ),                                                      \
			         t         = tDone + 1,                                                                            \
			         K         = travs,                                                                                \
			         kDone     = get_rank_pretravs( recDir, pRank ),                                                   \
			         POVp      = (( t+INITIAL_POV ) % nPlayers) + 1,                                                   \
			         pIdx      = POVp - 1
		CFRCollector collector = CFRCollector( dataDir, pRank, sRank, nPlayers, gameSize, t, POVp, K )

	# Useful global calc phase var; saves us from having to make this an additional arg everywhere
	global T; T = t 

	print( "\n"+(f"="*50) )
	print( f"COLLECTOR RANK:              P{collector.RANK_P}S{collector.RANK_S}" )
	print( f"PRESOLVED MODELS FOUND:      {tDone}" )
	print( f"P-RANK PRECOLLECTED K FOUND: {kDone}" )
	print( f"="*50 )

	print( "\n"+(f"="*100) )
	print( f"COMMENCING SDCFR ITER {t} COLLECTION SEGMENT".center(100) )
	print( f"="*100 )

	# Configure this worker's estimator singletons for this iter
	ESTIMATOR.setup_advnet( modelSize=mSize, modelIter=t-1, modelFile=modelFile, GPUrank=pRank )
	ESTIMATOR.ADVNET.train( False )
	ESTIMATOR.setup_multimodel( modelSize=mSize, iterSpan=t-1, modelFile=modelFile, GPUrank=pRank )
	ESTIMATOR.MULTIMODEL.train( False )

	cdef str GPU = f"cuda:{pRank}"
	print( f"Prev iter AdvNet and MultiModel ( M{mSize}T{t-1} ) state setup completed on GPU {GPU}" )
	print( f"\tESTIMATOR.ADVNET.ModelIter    = {ESTIMATOR.ADVNET.ModelIter}, .training = {ESTIMATOR.ADVNET.training}" )
	print( f"\tESTIMATOR.MULTIMODEL.IterSpan = {ESTIMATOR.MULTIMODEL.IterSpan}" )

	collector.Traverse() # Traversal phase: gametree DFS to collect trajectories
	collector.Calculate_Targets() # Calc phase: Derive AdvNet targets from collected trajectory data
	collector.Collection_Segment_Completed() # Collection phase done; do phase-end housekeeping


def Do_Collection_Segment( str dataDir, int pRank, int sRank, int mSize, int gameSize, int nPlayers, int travs ):
	_Do_Collection_Segment( dataDir, pRank, sRank, mSize, gameSize, nPlayers, travs )


# *-* #