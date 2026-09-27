from os import listdir, remove as destroy

import argparse
from pathlib import Path


def ArgParser():

	BOLD   = "\033[1m"
	UNBOLD = "\033[0m"
	args   = "--data_dir"
	desc0  = f"{BOLD}command{UNBOLD}: python clearbuffers.py " + args + "\n\n"
	desc1  = "Clears collected samples, trained models, CFR metadata, and segmented worker files.\n"
	desc   = desc0 + desc1
	dForm  = argparse.RawDescriptionHelpFormatter
	ap     = argparse.ArgumentParser( prog='clearbuffers', formatter_class=dForm, description=desc )

	defDataDir = str( Path.cwd()/'data' )
	dirHelp    = f"Root data directory containing CFR records and collected data (default: {defDataDir})."
	ap.add_argument( "-d", "--data_dir", type=str, default=defDataDir, help=dirHelp )

	return ap

def main( dataDir ):

	recDir = dataDir + "/segmented_records"
	advDir = dataDir + "/segmented_samples"

	if Path( recDir ).is_dir():
		print( f"Existing segmented record directory found: {recDir}" )
	else:
		print( f"No existing segmented record directory found, creating..." )
		Path( recDir ).mkdir( parents=True, exist_ok=True )
		print( f"Segmented record dir created: {recDir}" )

	if Path( advDir ).is_dir():
		print( f"Existing segmented adv data directory found: {advDir}" )
	else:
		print( f"No existing segmented adv data directory found, creating..." )
		Path( advDir ).mkdir( parents=True, exist_ok=True )
		print( f"Segmented adv data dir created: {advDir}" )

	open( dataDir + "/p1advs.pickle",'wb+' ).close()
	print("Collected P1 samples cleared." )

	open( dataDir + "/p2advs.pickle",'wb+' ).close()
	print("Collected P2 samples cleared." )

	open( dataDir + "/p1advs_TRAIN.pickle",'wb+' ).close()
	print("Collected P1 train samples cleared." )

	open( dataDir + "/p1advs_VAL.pickle",'wb+' ).close()
	print("Collected P1 val samples cleared." )

	open( dataDir + "/p2advs_TRAIN.pickle",'wb+' ).close()
	print("Collected P2 train samples cleared." )

	open( dataDir + "/p2advs_VAL.pickle",'wb+' ).close()
	print("Collected P2 val samples cleared." )

	open( dataDir + "/models.pickle",'wb+' ).close()
	print("Trained models cleared." )

	open( dataDir + "/metadata.pickle",'wb+' ).close()
	print("CFR metadata cleared." )

	for segfile in listdir( recDir ):
		destroy( recDir + "/" + segfile )
		print( f"Segment record file {recDir + '/' + segfile} destroyed." )

	for segfile in listdir( advDir ):
		destroy( advDir + "/" + segfile )
		print( f"Segment record file {advDir + '/' + segfile} destroyed." )

if __name__=='__main__':

	args    = ArgParser().parse_args()
	dataDir = args.data_dir

	raise SystemExit( main( dataDir ) )
