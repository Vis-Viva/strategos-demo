from strategos_tools.core.PYCONSTS      import *
from strategos_tools.utils.funcs        import hms, clear_current_line, clear_prev_lines
from strategos_tools.utils.data_structs import CFR_metadata
from strategos_tools.utils.data_ops     import get_presolved_iters, unsegment_iter_data, post_collection_cleanup

import argparse
from time    import sleep, time as TimeNow
from pathlib import Path


def ArgParser():

	BOLD   = "\033[1m"
	UNBOLD = "\033[0m"
	args   = "--n_segments --data_dir"
	desc0  = f"{BOLD}command{UNBOLD}: python CFRMonitor.py " + args + "\n\n"
	desc1  = "Manages multi-process CFR collection output.\n"
	desc2  = "Monitors specified directory for finished collection records from n workers.\n"
	desc3  = "When n finished records present, unifies segmented worker data into single pool for training."
	desc   = desc0 + desc1 + desc2 + desc3
	dForm  = argparse.RawDescriptionHelpFormatter
	ap     = argparse.ArgumentParser( prog='CFRMonitor', formatter_class=dForm, description=desc )

	segHelp = "Number of collection segments to monitor for output from."
	ap.add_argument( "n_segments", type=int, help=segHelp )

	defDataDir = str( Path.cwd()/'data' )
	dirHelp    = f"Directory to store CFR records and collected data (default: {defDataDir})."
	ap.add_argument( "-d", "--data_dir", type=str, default=defDataDir, help=dirHelp )

	return ap

def get_segmented_record_files( recordDir ):
	return [ p.name for p in Path( recordDir ).iterdir() ]

def await_collection_phase_completion( nSegments, solvingIter, recordDir ):

	segFiles             = get_segmented_record_files( recordDir )
	segmentsDone         = len( segFiles )
	Iter_Collection_Done = segmentsDone==nSegments

	print( f"\nCompleted segment records [ detected | required ] = [ {segmentsDone} | {nSegments} ]" )

	print( '\n'+("="*100) )
	print( f"AWAITING COMPLETION OF ITER {solvingIter} {nSegments}-SEGMENT COLLECTION PHASE".center(100) )
	print( ("="*100)+'\n\n' )

	SCAN_INTERVAL = 1
	monitorTime   = 0
	startTime     = TimeNow()
	while not Iter_Collection_Done:
		sleep( SCAN_INTERVAL )

		segFiles             = get_segmented_record_files( recordDir )
		segmentsDone         = len( segFiles )
		Iter_Collection_Done = segmentsDone==nSegments
		monitorTime          = TimeNow()-startTime 
		
		clear_current_line()
		clear_prev_lines( 3 )

		print( f"MONITORING DIRECTORY:      {recordDir}" )
		print( f"COMPLETED SEGRECS PRESENT: {segmentsDone}" )
		print( f"ELAPSED MONITORING TIME:   {hms( monitorTime )}",end='\r' )

	print( f"\n\nALL REQUIRED COMPLETE SEGRECS ({nSegments}) PRESENT, EXITING COLLECTION AWAIT PROCESS" )

def get_metadata( metaFile ):

	itersDone = get_presolved_iters( metaFile )

	if not itersDone:
		print( f"\t\tNo presolved iters found, creating new metadata..." )
		mData = CFR_metadata( metaFile )
		mData.pysave()
		print( f"\t\tEmpty metadata for new CFR run created and saved successfully." )
		
	else: 
		print( f"\t\t{itersDone} presolved iters found, loading existing metadata..." )
		mData = CFR_metadata.pyload( metaFile )
		print( f"\t\tExisting CFR metadata loaded successfully." )

	return mData

def main( nSegments, dataDir ):

	s        = 's' if nSegments>1 else ''
	recDir   = dataDir + "/segrecs"
	advDir   = dataDir + "/segadvs"
	metaFile = dataDir + "/metadata.pickle"

	print( LOGO )

	print( "="*100 )
	print( f"LAUNCHING NEW CFR MONITOR".center(100) )
	print( "="*100 )
	print( f"CFR run segments:       {nSegments}" )
	print( f"Root data directory:    {dataDir}" )
	print( f"Segmented record dir:   {recDir}" )
	print( f"Segmented adv data dir: {advDir}" )
	print( f"CFR metadata file:      {metaFile}" )
	print()

	print( "="*50 )
	print( f"Checking for existing data directories..." )

	if Path( dataDir ).is_dir():
		print( f"\tExisting data directory found: {dataDir}" )
		
		if Path( recDir ).is_dir():
			print( f"\tExisting segmented record directory found: {recDir}" )

		else:
			print( f"\tNo existing segmented record directory found, creating..." )
			Path( recDir ).mkdir( parents=True, exist_ok=True )
			print( f"\t\tSegmented record dir created: {recDir}" )

		if Path( advDir ).is_dir():
			print( f"\tExisting segmented adv data directory found: {advDir}" )

		else:
			print( f"\tNo existing segmented adv data directory found, creating..." )
			Path( advDir ).mkdir( parents=True, exist_ok=True )
			print( f"\t\tSegmented adv data dir created: {advDir}" )

	else:
		print( f"\tNo existing data directory found, creating..." )
		Path( dataDir ).mkdir( parents=True, exist_ok=True )
		print( f"\t\tCreated data directory: {dataDir}" )
		Path( recDir ).mkdir( parents=True, exist_ok=True )
		print( f"\t\tCreated segmented record directory: {recDir}" )
		Path( advDir ).mkdir( parents=True, exist_ok=True )
		print( f"\t\tCreated segmented adv data directory: {advDir}" )

	print( f"\tChecking for existing CFR metadata at {metaFile}..." )
	mData = get_metadata( metaFile )
	print( "="*50 )

	solvingIter = mData.get_current_iter()
	POVplayer   = (( solvingIter + INITIAL_POV ) % NUM_PLAYERS ) + 1
	trainFile   = dataDir + f"/p{POVplayer}advs_TRAIN.pickle"
	valFile     = dataDir + f"/p{POVplayer}advs_VAL.pickle"

	await_collection_phase_completion( nSegments, solvingIter, recDir )
	mData.collection_phase_completed( recDir )
	post_collection_cleanup( recDir, solvingIter )
	unsegment_iter_data( advDir, trainFile, valFile )

	print( '\n'+("="*100) )
	print( f"SEGMENTED ADVS/METADATA UNIFIED & DESTROYED; TRAIN & VAL SAMPLES SHUFFLED & SAVED".center(100) )
	print( f"READY FOR SDCFR ITER {solvingIter} TRAINING PHASE".center(100) )
	print( ("="*100)+'\n' )
	print( f"CFRMonitor process shutting down, have a nice day :)\n" )

if __name__=='__main__':

	args      = ArgParser().parse_args()
	nSegments = args.n_segments
	dataDir   = args.data_dir

	raise SystemExit( main( nSegments, dataDir ) )

