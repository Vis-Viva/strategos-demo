from strategos_tools.core.PYCONSTS  import *
from strategos_tools.AIOps.training import ModelTrainer

import argparse
from pathlib import Path
from torch.cuda import is_available as cuda_available


def ArgParser():

	BOLD   = "\033[1m"
	UNBOLD = "\033[0m"
	args   = "--model_size --num_epochs --batch_override --learning_rate --data_dir"
	desc0  = f"{BOLD}command{UNBOLD}: python CFRTrain.py " + args + "\n\n"
	desc1  = "Conducts Strategos's AdvNet training process on collected data.\n"
	desc   = desc0 + desc1
	dForm  = argparse.RawDescriptionHelpFormatter
	ap     = argparse.ArgumentParser( prog='CFRTrain', formatter_class=dForm, description=desc )

	cpuHelp = "Whether to run training on CPU instead of GPU (default: 0)."
	ap.add_argument( "--cpu", type=int, default=0, help=cpuHelp )
	
	mSizeHelp = "Scaling parameter for AdvNet layer sizes (default: 128)."
	ap.add_argument( "-m", "--model_size", type=int, default=128, help=mSizeHelp )

	epochHelp = "Number of training epochs (default: 4096)."
	ap.add_argument( "-e", "--num_epochs", type=int, default=4096, help=epochHelp )

	bSizeHelp = "Override for auto-determined multiGPU batchsize. " +                                                  \
				"Leave this alone unless you know what you're doing."
	ap.add_argument( "-b", "--batch_override", type=int, default=0, help=bSizeHelp )

	lRateHelp = "Learning rate to use for training (default: 0.001)."
	ap.add_argument( "-lr", "--learning_rate", type=float, default=0.001, help=lRateHelp )

	defDataDir = str( Path.cwd()/'data' )
	dirHelp    = "Master data directory containing training/val sets, trained models, " +                              \
				 f"and metadata (default: {defDataDir})."
	ap.add_argument( "-d", "--data_dir", type=str, default=defDataDir, help=dirHelp )

	return ap

def main( dataDir, cpu, modelSize, epoch_override, bsize_override, lrate_override ):

	modelDevice = "cpu" if cpu else "cuda"

	print()
	print( f"CALLING MODELTRAINER WITH ARGS:" )
	print( f"dataDir:        {dataDir}" )
	print( f"modelSize:      {modelSize}" )
	print( f"modelDevice:    {modelDevice}" )
	print( f"epoch_override: {epoch_override}" )
	print( f"bsize_override: {bsize_override}" )
	print( f"lrate_override: {lrate_override}" )
	print()
	input( f"Press enter to proceed" )
	
	ModelTrainer( dataDir, modelDevice, modelSize, epoch_override, bsize_override, lrate_override )

if __name__=='__main__':

	args = ArgParser().parse_args()
	if args.cpu not in {0,1}:
		raise ValueError( f"Got an unexpected value for --cpu. Expected 0/1, got {args.cpu}." )
	if not args.cpu and not cuda_available():
		errMsg1 = f"GPU training selected (--cpu 0), but no CUDA devices detected. "
		errMsg2 = f"Check CUDA configuration and retry, or run in CPU mode."
		raise ValueError( errMsg1 + errMsg2 )

	cpu     = args.cpu
	mSize   = args.model_size
	epochs  = args.num_epochs
	bSize   = args.batch_override
	lRate   = args.learning_rate
	dataDir = args.data_dir

	main( dataDir, cpu, mSize, epochs, bSize, lRate )
