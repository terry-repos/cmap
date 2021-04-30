from global_imports import session_vars as g
import copy 
import numpy as np
from pyutils.dict_utils import *

def make_globals_readable() :

	ID = g.cur['DAT_ID']
	print("ID: ", ID)
	nSamples = g.dat[ ID ]['ChanConfig']['NUM_SAMPLES']
	nChans = g.dat[ ID ]['ChanConfig']['NUM_CHANS']
	sampleRate =  g.dat[ ID ][ 'ChanConfig' ]['SAMPLE_RATE']
	pars = copy.copy( g.cur['STEP']['PARAMS'])
	currSlice = copy.copy( g.cur['SLICE'] )

	return ID, nSamples, nChans, sampleRate, pars, currSlice 


def make_params_readable() :

	return copy.deepcopy(g.cur['STEP']['PARAMS'])


def get_curr_ID() :

	return copy.deepcopy(g.cur['DAT_ID'])


def get_nChans() :
	# print( "get_curr_ID: ", get_curr_ID(), " g.dat[get_curr_ID()]['ChanConfig']['NUM_CHANS']: ", g.dat[get_curr_ID()]['ChanConfig']['NUM_CHANS'], " g.dat[get_curr_ID()]['Samples'].shape ",  g.dat[get_curr_ID()]['Samples'].shape )
	# if 'Events' in g.dat[get_curr_ID()].keys() :
	# 	print_keys_hierarchy(g.dat[get_curr_ID()]['Events'], "Events")
	return copy.copy( g.dat[get_curr_ID()]['ChanConfig']['NUM_CHANS'] )


def get_step_dir() :

	return copy.copy(g.cur['STEP']['OUTPUT']['DIR'])


def get_curr_sl() :

	return copy.copy(g.cur['SLICE'])



def get_chan_slice_with_neighbours(inChanN, nChansEitherSide, rowSize):
	# print("nChansEitherSide: ", nChansEitherSide, " get_nChans: ", get_nChans())

	if nChansEitherSide == get_nChans() :
		# print("neighboursChans == nChans")
		chanSlice = np.s_[ : ]

	elif nChansEitherSide == 0 :
		# print("neighboursChans == 0")
		chanSlice = np.s_[ inChanN ]

	else :
		startChanSlice = inChanN - nChansEitherSide
		endChanSlice = inChanN + nChansEitherSide + 1

		if not startChanSlice in g.cur['STEP']['PARAMS']['CHAN_RANGE'] :

			return None

			# startChanSlice = g.cur['STEP']['PARAMS'][ 'CHAN_RANGE' ][0]
			# endChanSlice = g.cur['STEP']['PARAMS'][ 'CHAN_RANGE' ][rowSize]


		elif not endChanSlice in g.cur['STEP']['PARAMS'][ 'CHAN_RANGE' ] :

			return None

			# endChanSlice = g.cur['STEP']['PARAMS'][ 'CHAN_RANGE' ][-1]

		chanSlice = np.s_[startChanSlice : endChanSlice]	

	return chanSlice



def get_row_size( nChansEitherSide, nChans=None ) :
	
	nChans = get_nChans()
	if nChansEitherSide == nChans :
		rowSize = nChans

	elif nChansEitherSide == 0 :
		rowSize = 1

	else:
		rowSize = nChansEitherSide * 2 + 1	

	return rowSize


	

