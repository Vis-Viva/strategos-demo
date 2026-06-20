#distutils: language = c
#cython: language_level 3
#cython: profile = False


cimport cython
cimport numpy as cnp
cnp.import_array()

from cpython.array cimport clone as pyarr
from cython.view   cimport array as cyarr

from strategos_tools.env.player_ops   cimport Are_Opponents
from strategos_tools.env.gamenode_ops cimport DummyNode
from strategos_tools.utils.funcs      cimport progressbar as PB

import sys, pickle, datetime, select
from os     import listdir, remove as destroy, getcwd as cwd
from time   import sleep, time as TimeNow
from random import shuffle as permute

import numpy as np, torch as pt
from numpy import asarray as NP, ascontiguousarray as CONTIG, expand_dims as newaxis, float32 as f32, uintc, intc
from torch import as_tensor as TENSOR, from_numpy as NP2TENSOR, float32 as tf32, int32 as tint32


# ==================================================================================================
# This module provides the data management tools that bridge the neural CFR collection and training 
# phases. Operations include unifying segmented temp data from multiple collection workers into a
# single training pool, training data retrieval, and various other disk read/write utils.
# ==================================================================================================


# Returns list of trained AdvNet model dicts, option to specify CFR iter of models to return
cdef list   load_models( str from_file, int for_iter=-1 ): #noexcept:

	cdef dict mDict
	cdef list allModels=[]

	with open( from_file,'rb' ) as modelFile:
		while True:
			try: 
				mDict = pickle.load( modelFile )
			except EOFError: 
				break

			if (for_iter == -1) or (mDict[ 'IterNum' ]==for_iter):
				allModels.append( mDict[ 'state_dict' ] )

	return allModels

# DEPRECATED: We save adv sample dicts individually now, not whole advmaps
cdef list  _load_advmaps( str from_file, int for_iter=-1 ): #noexcept:
	
	cdef:
		advmap amap
		object adict
		list   advmaps=[]

	with open( from_file,'rb' ) as samplefile:

		while True:
			try: 
				adict = pickle.load( samplefile )
			except EOFError: 
				break

			amap = advmap.from_dict( adict )

			if (for_iter == -1) or (amap.IterNum == for_iter): 
				advmaps.append( amap )

	return advmaps

# DEPRECATED: Since we save sample dicts individually, we don't need to extract them from loaded maps
cdef list __extract_all_samples( list from_advmaps ): #noexcept:
	
	cdef:
		uint   nMaps   = <uint>len( from_advmaps ), m
		list   tSamples=[]
		advmap amap

	print( f"\nExtracting individual samples from obtained advmaps..." )

	for m from 1 <= m <= nMaps:
		amap = from_advmaps[ m-1 ]
		tSamples.extend( amap.extract_samples() )
		print( PB(m,nMaps) + f" ( {m}/{nMaps} )          ", end='\r' )
		
	print( f"\nSuccessfully extracted {len( tSamples )} individual samples." )

	return tSamples

# Returns list of sample dicts from segmented iter data file, used during post-collection data unification
cdef list __load_segmented_sample_dicts( str from_file ): #noexcept:

	cdef list loadedDicts=[]

	with open( from_file,'rb' ) as sampleFile:
		while True:
			try: loadedDicts.append( pickle.load( sampleFile ) )
			except EOFError: break

	return loadedDicts

# Returns list of sample dicts from persistent unified file so it can be extended with new samples
cdef list __load_existing_sample_dicts( str from_file ): #noexcept:

	cdef list existingDicts=[]

	with open( from_file,'rb' ) as sampleFile: 
		try: 
			existingDicts = pickle.load( sampleFile )
		except EOFError: 
			pass

	return existingDicts

# Clears samples from unified file after they've been loaded so new expanded collection can be written
# TODO: If process crashes after this but before writing new data, old data is all lost. Backup before clear?
cdef void __clear_existing_samples( str in_file ): #noexcept:
	open( in_file,'wb+' ).close()

# After unifying newly collected data, serialize it to persistent storage
cdef void __save_unsegmented_samples( list shuffledSamples, str to_file, bint Clear_Existing=FALSE ): #noexcept:
	
	if Clear_Existing:
		__clear_existing_samples( in_file=to_file )
		print( f"\tExisting data in file {to_file} cleared; file ready for updated data." )

	with open( to_file,'wb+' ) as Finalized_Sample_File:
		print( "\tSaving updated sample list, plz be patient..." )
		pickle.dump( shuffledSamples, Finalized_Sample_File, protocol=-1 )

# Handles the "extension" part of the per-iter, post-collection data management process: 
# Load existing train & val data ⟶ shuffle & split new data into train & val sets ⟶ append to existing ⟶
# shuffle results ⟶ clear old data from disk (it's all in mem now anyway) ⟶ save extended train & val data
cdef void __append_new_data( str trainFile, str valFile, list iterSamples, float val_split=0.25 ): #noexcept:

	print( f"\nFound {len( iterSamples )} new iter samples. Loading pre-existing data..." )

	cdef:
		list tAdvs     = __load_existing_sample_dicts( trainFile ),                                                    \
			 vAdvs     = __load_existing_sample_dicts( valFile ),                                                      \
			 newAdvs   = iterSamples,                                                                                  \
			 newTAdvs,                                                                                                 \
			 newVAdvs
		uint nNewAdvs  = <uint>len( newAdvs ),                                                                         \
			 nTAdvs    = <uint>len( tAdvs ),                                                                           \
			 nVAdvs    = <uint>len( vAdvs ),                                                                           \
			 nNewVAdvs = <uint>(nNewAdvs * val_split),                                                                 \
			 nNewTAdvs = nNewAdvs - nNewVAdvs

	print( f"Loaded {nTAdvs} existing train samples & {nVAdvs} existing val samples" )

	print( f"Shuffling, splitting, and appending new data..." )
	permute( newAdvs )
	newTAdvs = newAdvs[ :nNewTAdvs ]
	newVAdvs = newAdvs[ nNewTAdvs: ]
	print( f"\tShuffled new data and split into {nNewTAdvs} train & {nNewVAdvs} val samples." )

	tAdvs.extend( newTAdvs )
	vAdvs.extend( newVAdvs )
	permute( tAdvs )
	permute( vAdvs )
	nTAdvs = <uint>len( tAdvs )
	nVAdvs = <uint>len( vAdvs )
	print( f"\tAppended new data to existing data; resulting collections shuffled." )

	# TODO: The "Clear_Existing" part of this is dangerous - how do we avoid this?
	print( f"\nSaving {nTAdvs} combined training samples to file {trainFile}..." )
	__save_unsegmented_samples( tAdvs, to_file=trainFile, Clear_Existing=TRUE )
	print( f"Training data saved." )

	# TODO: The "Clear_Existing" part of this is dangerous - how do we avoid this?
	print( f"\nSaving {nVAdvs} combined val samples to file {valFile}..." )
	__save_unsegmented_samples( vAdvs, to_file=valFile, Clear_Existing=TRUE )
	print( f"Validation data saved." )

# Orchestrates per-iter, post-collection data management process (unification + extension): 
# Find new segmented per-worker temp adv files ⟶ load sample dicts from each ⟶ combine them ⟶
# append new unified data to existing data
cdef void  _unsegment_iter_data( str advDir, str trainFile, str valFile, float val_split=0.25 ): #noexcept:

	cdef:
		list allFiles  = listdir( advDir ), segSamples=[], newIterSamples=[]
		uint nSegments = <uint>len( allFiles ), s
		str  segFile

	print( f"\nUnifying segmented adv data from {advDir}..." )
	print( f"Target files:" )
	print( f"\tTraining data:   {trainFile}" )
	print( f"\tValidation data: {valFile}" )
	for s from 1 <= s <= nSegments:
		segFile = advDir + '/' + allFiles[ s-1 ]
		print( f"\n\tFound segFile {segFile} for collection segment #{s}" )

		segSamples = __load_segmented_sample_dicts( from_file=segFile )
		print( f"\tLoaded {len( segSamples )} samples from segFile #{s}" )

		newIterSamples.extend( segSamples )
		print( f"\tExtended iterSamples with samples from segment #{s}. Running total iter samples: {len(newIterSamples)}" )
		
	print( f"\nGathered {len( newIterSamples )} total combined iter samples from segmented data." )

	__append_new_data( trainFile, valFile, newIterSamples, val_split )

# Returns list of TRUE SAMPLES (not sample dicts) able to be fed into the DATAMACHINE constructor
cdef list  _load_true_samples( str from_file ): #noexcept:
	
	cdef list sDicts = __load_existing_sample_dicts( from_file ), trueSamples=[]
	cdef dict sDict
	
	for sDict in sDicts: 
		trueSamples.append( advsample.from_dict( sDict ) )

	return trueSamples

# Final operation after completion of a CFR iter; just cleans up temp multi-worker collection files
cdef void  _post_iter_cleanup( str advDir, uint for_iter ): #noexcept:

	cdef:
		list allFiles  = listdir( advDir ) # No longer need this segmented training data
		uint nSegments = <uint>len( allFiles ), s
		str  segmentFile

	print( f"\nCleaning up iteration {for_iter} temp data..." )
	print( f"Destroying {nSegments} temp segmented adv files from dir: {advDir}..." )

	for s from 1 <= s <= nSegments:
		segmentFile = advDir + '/' + allFiles[ s-1 ]
		destroy( segmentFile )
		print( f"\n\tSegAdv file {cwd() + '/' + segmentFile} destroyed." )

	print( f"\nFINAL SEGMENTED DATA CLEANUP COMPLETE. Ready to commence iter {for_iter+1}." )


# ---------- PYTHON INTERFACE FUNCTIONS ------------------------------------------------------------
# So we can do disk read/write ops from the python scripts that implement the per-iter data
# management process bridging SDCFR phases.


# Use this in python-level nn code to get samples for DATAMACHINE constructor
def load_nn_samples( from_file ): 
	return _load_true_samples( from_file )

# Use this in python-level data mgmt code to combine new segmented data after collection phase completes
def unsegment_iter_data( advDir, trainFile, valFile, val_split=0.25 ):
	_unsegment_iter_data( advDir, trainFile, valFile, val_split )

# Tells us how many traversals have already been completed by the specified worker
def get_rank_pretravs( recDir, pRank ):
	
	segFiles = listdir( recDir )
	if len( segFiles )==0: 
		return 0
		
	rankPrefix = recDir + '/' + f"segrecords_P{pRank}"
	if not any([ segfile.startswith( rankPrefix ) for segfile in segFiles ]): 
		return 0

	rankPreTravs=0
	for segfile in segFiles:
		if segfile.startswith( rankPrefix ):
			with open( segfile,'rb' ) as sfile:
				segdict = pickle.load( segfile )
				rankPreTravs += segdict[ 'SegmentTravsDone' ]

	return rankPreTravs

# Just queries metadata for how many iters we've fully completed
def get_presolved_iters( metaFile ):
	try: 
		mData = CFR_metadata.load( from_file=metaFile )
	except EOFError: 
		return 0
	return mData.CFRItersCompleted

# Same but gives the current iter in progress
def get_current_iter( metaFile ):
	try: 
		mData = CFR_metadata.load( from_file=metaFile )
	except EOFError: 
		return 1
	return mData.CurrentIter

# Final iter-end temp file cleanup, called at end of iter's training phase
def post_iter_cleanup( advDir, for_iter ):
	_post_iter_cleanup( advDir, for_iter )


# *-* #