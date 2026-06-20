# distutils: language = c
# cython: language_level 3

cimport cython

cimport numpy as cnp
cnp.import_array()

from strategos_tools.core.CONSTS        cimport *
from strategos_tools.core.containers    cimport *
from strategos_tools.env.event_ops      cimport gameevent
from strategos_tools.env.gamenode_ops   cimport gamenode
from strategos_tools.env.infoset_ops    cimport infoset
from strategos_tools.env.actionset_ops  cimport actionset
from strategos_tools.utils.data_structs cimport advmap, CFR_metadata


cdef uint T


cdef class ZMap:

	cdef:
		ll         n
		vector_ll  Zn
		vector_int UZn
		uint       size

	cdef bint Contains( self, ll zKey ) #noexcept

	cdef int  payout_from( self, ll zKey ) #noexcept

	cdef void append( self,ll zKey,int U_z ) #noexcept

	cdef int  UnweightedMeanU( self ) #noexcept

	cdef void summary( self )


cdef class ConnectionMap:

	cdef:
		ll        from_key
		vector_ll to_keys
		MultiMat  weights

	cdef flt2 get_weights( self, ll to_key ) #noexcept

	cdef void set_weights( self, ll to_key, flt2 weights ) #noexcept

	cdef void replace_connection( self, ll oldKey, ll newKey ) #noexcept

	cdef bint Connects_To( self, ll key ) #noexcept


cdef class GTNode:

	cdef:
		uint          ActingPlayer, TraversingPlayer
		ll            Key
		gamenode      GameNode
		infoset       I_tp, I_ap # Current infosets for traversing & acting players
		actionset     A

		vector_int    FullAInds, RemainingAInds
		bint          Fully_Explored, Solvable

		bint          Is_Terminal
		vector_ll     TerminalPath # 𝓟(z) if node is terminal
		int           uz           # uₜₚ(z) if node is terminal
		
		ll            GTParentKey
		vector_ll     SuccessorKeys, SubKeys, Zn
		ConnectionMap ConnectionGraph # AXES=(a,h,t), (|A|,1326,T); (n·a).GTKey() ⟶ σᵗ(h,a) ∀ a∈A(I_ap), h∈H, & t<T
		ConnectionMap FwdReaches      # AXES=z⟶(h,t), |Z[n]|⟶(1326,T); z.GTKey() ⟶ πᵗ(h,z) ∀ z∈Z[n], h∈H, & t<T
		ZMap          UZn             # AXES=z⟶(h,),  |Z[n]|⟶(1326,); z.GTKey() ⟶ z.CFPayoffs ∀ z∈Z[n]
		matrix_flt    CFReaches       # πoᵗ(h) ∀ h∈H & t≤T; axes=(h,t)

	cdef vector_int    __AInds( self, uint numActions ) #noexcept

	cdef ConnectionMap __CGraph( self, uint estimatorRank=* ) #noexcept

	cdef vector_ll     __get_path_node_keys( self ) #noexcept

	cdef vector_ll     __get_successor_node_keys( self ) #noexcept

	cdef void          __INIT_NON_TERMINUS__( self, uint estimatorRank ) #noexcept

	cdef void          __INIT_TERMINUS__( self ) #noexcept

	cdef void          __INIT__( self, uint traversalPOV, gamenode fromNode, bint Is_Terminal, uint estimatorRank=* ) #noexcept

	cdef bint            Has_Direct_Connection_To( self, ll key ) #noexcept

	cdef void            skip_dealer_connection( self, ll skip_deal_key, ll skip_to_key ) #noexcept

	cdef void            exploration_completed( self ) #noexcept

	cdef void            append_reachable_endgame( self, ll zKey, int uz ) #noexcept

	cdef void            initialize_fwd_reaches( self ) #noexcept

	cdef uint1           count_path_action_points( self ) #noexcept

	cdef vector_ll       PathKeys( self ) #noexcept


cdef class NodeVector:

	cdef:
		uint   size, capacity
		void **_data
		list   __nodeList
		# ^This list must hold the actual nodes simply so they stay alive & in place for the pointers stored in _data.
		# NB this list is not intended to ever be used for access as py list access is very inefficient ( vomit ).
		# For access, get node ptr from _data and do <GTNode> cast instead; this SHOULD bypass python overhead.

	cdef uint   __get_min_spanning_cap( self ) #noexcept

	cdef void   resize( self, uint newCap=* ) #noexcept

	cdef void   shrink_wrap( self ) #noexcept

	cdef void   append( self, GTNode newNode ) #noexcept

	cdef GTNode at( self, uint idx ) #noexcept


cdef class GameTree:

	cdef vector_ll GTKeys
	cdef NodeVector Nodes

	cdef bint   Contains( self, ll nodeKey ) #noexcept

	cdef GTNode node_at( self, ll nodeKey ) #noexcept

	cdef void   append( self, GTNode newNode ) #noexcept


cdef class CFRCollector:

	cdef:
		bint      K_Phase_Complete
		str       AdvFile, RecordFile
		uint      RANK_P, RANK_S, nPlayers, GameSize, SBlind, SolvingIter, POVplayer, SegmentTravReq, SegmentTravsDone,\
			      nSolvedPositions, nSolvedSubgames, nCollectedSamples, RootNodeRolls, MinStack, MaxStack, StackRange
		double    StackDeviation, SegmentStart, TraversalDuration, AvgTravTimeIso, AdvCalcTime, SegmentDuration,       \
				  TreeComplexity
		
		uint1     CurrentStacks, ExplorationDepths, nNodesSeen
		flt1      ASizes
		vector_ll zKeys, FullyExploredKeys, SolvableKeys, SolvableSubgames
		GameTree  GTree

		object    pyRNG

	cdef void     __INIT__( self, str dataDir, uint parallelRank, uint serialRank, uint nPlayers, uint gameSize, uint for_iter, uint POVplayer, uint segTravs ) #noexcept

	cdef GTNode     at( self, ll nodeKey ) #noexcept

	# ----- TRAVERSAL PHASE OPS ------------------------------------------------

	cdef bint       Has_Encountered( self, ll nKey ) #noexcept

	cdef void       extend_subgames( self, GTNode new_subnode ) #noexcept

	cdef bint       Node_Has_Unexplored_Actions( self, ll nKey ) #noexcept

	cdef void       node_fully_explored( self, ll nKey ) #noexcept

	cdef bint       Dealer_Node_Is_Parent_Of( self, GTNode N ) #noexcept

	cdef ll         find_deal_sequence_start( self, ll above_key ) #noexcept

	cdef void       remove_parent_dealer_node( self, GTNode pathNode ) #noexcept

	cdef flt1       NodesPerTraversal( self ) #noexcept

	cdef void     __print_collection_progress( self ) #noexcept

	cdef void       trav_completed( self ) #noexcept

	cdef void     __accumulate_path_node_count( self, GTNode zNode ) #noexcept

	cdef void     __complete_terminal_initialization( self, GTNode zNode ) #noexcept

	cdef void       initialize_entry( self, gamenode n, bint Is_Terminal ) #noexcept

	cdef int        ActionSample( self, flt1 from_strategy ) #noexcept

	cdef uint     __next_unexplored_index( self, ll for_key ) #noexcept

	cdef gamenode   NextSuccessor( self, ll for_key, int aIdx=* ) #noexcept

	cdef void       Collect( self, gamenode currentNode ) #noexcept

	cdef void       K_phase_completed( self ) #noexcept

	cdef void     __reroll_stacks( self ) #noexcept

	cdef gamenode __reroll_root_node( self ) #noexcept

	cdef void       Traverse( self ) #noexcept

	# ----- CALC PHASE OPS -----------------------------------------------------

	cdef ll         PrecedingPNodeKey( self, ll nKey ) #noexcept

	cdef bint       All_SubNodes_Explored( self, ll nKey ) #noexcept

	cdef void       find_solvable_nodes( self ) #noexcept

	cdef void       find_solvable_subgames( self ) #noexcept

	cdef void       build_solvable_paths( self ) #noexcept

	cdef void       initialize_fwd_reach_maps( self ) #noexcept

	cdef GTNode     GTParent( self, ll of_key ) #noexcept

	cdef void       set_base_fwd_reaches( self ) #noexcept

	cdef uint     __count_steps( self, uint by_player, vector_ll along_path ) #noexcept

	cdef matrix_flt RootCFReach( self, ll Sr ) #noexcept

	cdef void       calculate_base_cfreaches( self ) #noexcept

	cdef matrix_flt CFReach( self, ll nKey ) #noexcept

	cdef void       accumulate_zpath_cfreaches( self, ll Sr, ll zKey ) #noexcept

	cdef void       calculate_subgame_cfreaches( self, ll Sr ) #noexcept

	cdef void       calculate_counterfactual_reaches( self ) #noexcept

	cdef flt2       TerminalFwdReach( self, ll from_key, ll via_key, ll to_zkey ) #noexcept

	cdef void       accumulate_zpath_fwd_reaches( self, ll Sr, ll zKey ) #noexcept

	cdef void       calculate_subgame_fwd_reaches( self, ll Sr ) #noexcept

	cdef void       calculate_fwd_reaches( self ) #noexcept

	cdef void       calculate_reaches( self ) #noexcept

	cdef flt1       IReach( self, GTNode n ) #noexcept

	cdef flt3       ReachWeightedPayoffs( self, GTNode n ) #noexcept

	cdef flt2       NodeValue( self, GTNode n ) #noexcept

	cdef flt2       ActionValues( self, GTNode n ) #noexcept

	cdef flt1       ActionAdvs( self, GTNode POVnode ) #noexcept

	cdef void       calculate_subgame_targets( self, GTNode Sr ) #noexcept

	cdef void       Calculate_Targets( self ) #noexcept

	# ----- SEGMENT LOGGING OPS ------------------------------------------------

	cdef void       find_exploration_depths( self ) #noexcept

	cdef dict       SegmentRecords( self ) #noexcept

	cdef void       save_records( self ) #noexcept

	cdef void       print_solvable_subgame_info( self )

	cdef void       Collection_Segment_Completed( self ) #noexcept


cdef void __await_prev_iter_completion( str advDir, str recDir ) #noexcept

cdef void  _Do_Collection_Segment( str dataDir, int pRank, int sRank, int mSize, int gameSize, int nPlayers, int travs ) #noexcept


# *-* # 