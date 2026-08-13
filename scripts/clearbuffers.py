from strategos_tools.core.PYCONSTS import SEG_REC_DIR, SEG_ADV_DIR
from os import listdir, getcwd, remove as destroy

DATADIR = getcwd() + "/data/"

open( DATADIR + "p1advs.pickle",'wb+' ).close()
print("Collected P1 samples cleared." )

open( DATADIR + "p2advs.pickle",'wb+' ).close()
print("Collected P2 samples cleared." )

open( DATADIR + "p1advs_TRAIN.pickle",'wb+' ).close()
print("Collected P1 train samples cleared." )

open( DATADIR + "p1advs_VAL.pickle",'wb+' ).close()
print("Collected P1 val samples cleared." )

open( DATADIR + "p2advs_TRAIN.pickle",'wb+' ).close()
print("Collected P2 train samples cleared." )

open( DATADIR + "p2advs_VAL.pickle",'wb+' ).close()
print("Collected P2 val samples cleared." )

open( DATADIR + "models.pickle",'wb+' ).close()
print("Trained models cleared." )

open( DATADIR + "metadata.pickle",'wb+' ).close()
print("CFR metadata cleared." )

for segfile in listdir( SEG_REC_DIR ):
	destroy( SEG_REC_DIR + segfile )
	print( f"Segment record file {SEG_REC_DIR + segfile} destroyed." )

for segfile in listdir( SEG_ADV_DIR ):
	destroy( SEG_ADV_DIR + segfile )
	print( f"Segment record file {SEG_ADV_DIR + segfile} destroyed." )
