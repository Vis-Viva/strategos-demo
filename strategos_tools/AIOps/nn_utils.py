from strategos_tools.utils.data_structs import AdvNetInputs, MMInputs

import torch as pt


# ==================================================================================================
# Just a toolbox of utilities that do some nice things related our nn models.
# Generally useful for debugging or testing out new features etc. 
# ==================================================================================================


def get_parameter_sets( model1, model2 ):

	stateDicts  = [ model1.state_dict(), model2.state_dict() ]
	mParameters = [ list( state.items() ) for state in stateDicts ]
	pTensors    = [ [],[] ]

	for modelNum in range( len( mParameters ) ):
		modelParams = mParameters[ modelNum ]
		for parameter in modelParams:
			pTensor = parameter[ 1 ].to( 'cuda:0' )
			pTensors[ modelNum ].append( pTensor )

	return pTensors

def test_parameter_equivalence( parameterList1, parameterList2 ):
	parameterPairs  = zip( parameterList1, parameterList2 )
	parametersEqual = [ pt.equal( parameterPair[0], parameterPair[1] ) for parameterPair in parameterPairs ]
	return all( parametersEqual )

def test_parameter_closeness( parameterList1, parameterList2 ):
	parameterPairs  = zip( parameterList1, parameterList2 )
	parametersClose = [ pt.allclose( parameterPair[0], parameterPair[1] ) for parameterPair in parameterPairs ]
	return all( parametersClose )

def test_model_equivalence( model1, model2 ):
	
	pSets = get_parameter_sets( model1, model2 )
	pset1 = pSets[ 0 ]
	pset2 = pSets[ 1 ]
	parameters_equal = test_parameter_equivalence( pset1, pset2 )
	parameters_close = test_parameter_closeness( pset1, pset2 )

	print()
	print( f"model1 = model2: {parameters_equal}" )
	print( f"model1 ≈ model2: {parameters_close}" )
	print()

def AdvNetCompiler( aNet, GPUrank=0, mode='max-autotune' ):

	print( f"\n===== COMPILING ADVNET =====" )
	print( f"Compile mode: {mode}" )

	print( f"Constructing compiled model..." )
	aNetCompiled = pt.compile( aNet,mode=mode )
	print( f"Compiled model constructed." )

	print( f"\n=== INITIALIZING COMPILED INFERENCE GRAPH ===" )

	if not aNet.training:
		dummyInputs = AdvNetInputs.DummyInputs( GPUrank )
		H  = dummyInputs.H
		A  = dummyInputs.A
		nA = dummyInputs.nA
		hCc, hCr, hCs = dummyInputs.hC_c, dummyInputs.hC_r, dummyInputs.hC_s
		fCc, fCr, fCs = dummyInputs.fC_c, dummyInputs.fC_r, dummyInputs.fC_s
		tCc, tCr, tCs = dummyInputs.tC_c, dummyInputs.tC_r, dummyInputs.tC_s
		rCc, rCr, rCs = dummyInputs.rC_c, dummyInputs.rC_r, dummyInputs.rC_s

	else:
		dummyInputs = AdvNetInputs.DummyInputs( 0 )
		H  = dummyInputs.H.to('cpu')
		A  = dummyInputs.A.to('cpu')
		nA = dummyInputs.nA
		hCc, hCr, hCs = dummyInputs.hC_c.to('cpu'), dummyInputs.hC_r.to('cpu'), dummyInputs.hC_s.to('cpu')
		fCc, fCr, fCs = dummyInputs.fC_c.to('cpu'), dummyInputs.fC_r.to('cpu'), dummyInputs.fC_s.to('cpu')
		tCc, tCr, tCs = dummyInputs.tC_c.to('cpu'), dummyInputs.tC_r.to('cpu'), dummyInputs.tC_s.to('cpu')
		rCc, rCr, rCs = dummyInputs.rC_c.to('cpu'), dummyInputs.rC_r.to('cpu'), dummyInputs.rC_s.to('cpu')

	aNetCompiled( H, hCc,hCr,hCs, fCc,fCr,fCs, tCc,tCr,tCs, rCc,rCr,rCs, A, nA )

	print( f"AdvNet compiled and initialized successfully." )

	return aNetCompiled

def MMCompiler( mm, iterSpan, GPUrank=0, mode='max-autotune' ):

	print( f"\n===== COMPILING MULTIMODEL =====" )
	print( f"Compile mode: {mode}" )

	print( f"Constructing compiled MM..." )
	mmCompiled = pt.compile( mm,mode=mode )
	print( f"Compiled MM constructed." )

	print( f"\n=== INITIALIZING COMPILED INFERENCE GRAPH ===" )
	dummyInputs = MMInputs.DummyInputs( iterSpan, GPUrank )
	mmCompiled( dummyInputs )

	print( f"MultiModel compiled and initialized successfully." )

	return mmCompiled