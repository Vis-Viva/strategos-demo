from strategos_tools.core.PYCONSTS  import *
from strategos_tools.utils.data_ops import get_presolved_iters, get_rank_pretravs
from strategos_tools.AIOps.nn_utils import validate_device
from strategos_tools.CFR            import CollectionOps

import argparse, numpy as np
from pathlib import Path

np.seterr( divide='raise', invalid='raise' )


def ArgParser():

	BOLD   = "\033[1m"
	UNBOLD = "\033[0m"
	args   = "--parallel_rank --serial_rank --model_size --game_size --n_traversals --data_dir"
	desc0  = f"{BOLD}command{UNBOLD}: python CFRCollect.py " + args + "\n\n"
	desc1  = "Spins up one CFR collection worker process.\n"
	desc2  = "Conducts DFS HUNLTH gametree traversal and calculates advantage targets.\n"
	desc3  = "Generates (state, action, advantage) tuples for training AdvNet."
	desc   = desc0 + desc1 + desc2 + desc3
	dForm  = argparse.RawDescriptionHelpFormatter
	ap     = argparse.ArgumentParser( prog='CFRCollector', formatter_class=dForm, description=desc )

	deviceHelp = "Device this worker's estimator is assigned to. -1 for CPU, 0+ for GPU IDs."
	ap.add_argument( "device", type=int, help=deviceHelp )

	pRankHelp = "This worker's parallel process index, should generally match device ID for multi-GPU runs."
	ap.add_argument( "parallel_rank", type=int, help=pRankHelp )

	sRankHelp = "If doing a serial sequence of multiple collection segments per parallel rank, " +                     \
				"this worker's position in that sequence."
	ap.add_argument( "serial_rank", type=int, help=sRankHelp )

	mSizeHelp = "Scaling parameter for AdvNet layer sizes."
	ap.add_argument( "model_size", type=int, help=mSizeHelp )

	gSizeHelp = "Base starting stack (randomized ±25%% per-hand per-player) - affects size of game tree."
	ap.add_argument( "game_size", type=int, help=gSizeHelp )

	nTravHelp = "Number of partial gametree traversals (i.e. hands played) this worker should conduct."
	ap.add_argument( "n_traversals", type=int, help=nTravHelp )

	defDataDir = str( Path.cwd()/'data' )
	dirHelp    = f"Root data directory to store CFR records and collected data (default: {defDataDir})."
	ap.add_argument( "-d", "--data_dir", type=str, default=defDataDir, help=dirHelp )

	return ap

def get_segmented_adv_files( advDir ):
	return [ p.name for p in Path( advDir ).iterdir() ]

def main( pRank, sRank, mSize, gameSize, nPlayers, travs, device, dataDir ):

	advDir       = dataDir + "/segadvs"
	recDir       = dataDir + "/segrecs"
	metaFile     = dataDir + "/metadata.pickle"
	deviceStr    = "CPU" if device==-1 else f"GPU cuda:{device}"
	waiting      = len( get_segmented_adv_files( advDir ) ) > 0 
	T            = get_presolved_iters( metaFile )
	rankPretravs = get_rank_pretravs( recDir, pRank )
	tStr         = "0" if T==0 else f"*** {T} ***"
	kStr         = "0" if rankPretravs==0 else f"*** {rankPretravs} ***"

	if T==0 and rankPretravs==0 and not waiting: 
		print( LOGO )
		print( "\n" + ("="*100) )
		print( f"BEGINNING NEW SDCFR RUN".center(100) )
		print( f"="*100 )
		print( f"RUN PARAMETERS:" )
		print( f"\tRANK              = P{pRank}S{sRank}" )
		print( f"\tMODEL SIZE        = {mSize}" )
		print( f"\tEstimator device  = {deviceStr}" )
		print( f"\tPresolved Iters   = {tStr}" )
		print( f"\tP-Rank Travs Done = {kStr}" )
		print( f"\tSegment Trav Req  = {travs}" )
		print( f"\tStack Size        = {gameSize}" )
		
	else:
		print( "\n"+(f"="*100) )
		print( f"CONTINUING EXISTING SDCFR RUN".center(100) )
		print( f"="*100 )
		print( f"RUN PARAMETERS:" )
		print( f"\tRANK              = P{pRank}S{sRank}" )
		print( f"\tMODEL SIZE        = {mSize}" )
		print( f"\tEstimator device  = {deviceStr}" )
		print( f"\tPresolved Iters   = {tStr}" )
		print( f"\tP-Rank Travs Done = {kStr}" )
		print( f"\tSegment Trav Req  = {travs}" )
		print( f"\tStack Size        = {gameSize}" )

	CollectionOps.Do_Collection_Segment( device, dataDir, pRank, sRank, mSize, gameSize, nPlayers, travs )

if __name__=='__main__':

	args    = ArgParser().parse_args()
	pRank   = args.parallel_rank
	sRank   = args.serial_rank
	mSize   = args.model_size
	gSize   = args.game_size
	nTrav   = args.n_traversals
	device  = validate_device( args.device )
	dataDir = args.data_dir

	raise SystemExit( main( pRank, sRank, mSize, gSize, NUM_PLAYERS, nTrav, device, dataDir ) )
