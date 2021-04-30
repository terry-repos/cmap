import sys
import os

sys.path.append('..')
from global_imports import session_vars as g
from pyutils.io_utils import *
from algorithms.stats import *
from pyutils.dict_utils import *

def update_samples_info() :

	g.command_history.add('Updating process info ... ', os.path.basename(__file__),g.inspect.stack()[0][3])

	ID = g.cur['DAT_ID']
	if 'SAMPLES_PROCESSING' in g.cur['STEP']['PARAMS'].keys() :
		if g.cur['STEP']['PARAMS']['SAMPLES_PROCESSING'] :
			if 'Samples' in g.dat[ ID ].keys() :

				NUM_SAMPLES = g.dat[ ID ]['Samples'].shape[1]
				NUM_CHANS = g.dat[ ID ]['Samples'].shape[0]	

				g.dat[ID]['ChanConfig']['NUM_CHANS'] = g.dat[ ID ]['Samples'].shape[0]
				g.dat[ID]['ChanConfig']['NUM_SAMPLES'] = g.dat[ ID ]['Samples'].shape[1]		

				g.dat[ID]['ChanConfig']['SAMPLE_RANGE'] = [1, g.dat[ ID ]['ChanConfig'] ['NUM_SAMPLES']]

				if not 'CHANNEL_ORDER' in g.cur['ChanConfig'].keys() :
					g.cur['ChanConfig']['CHANNEL_ORDER'] = []
				if len( g.cur['ChanConfig']['CHANNEL_ORDER'] ) == 0:
					g.dat[ID]['ChanConfig']['CHANNEL_ORDER'] = list( range(0, (g.dat[ID]['ChanConfig']['NUM_CHANS'])) )

				else :
					g.dat[ID]['ChanConfig']['CHANNEL_ORDER'] = g.cur['ChanConfig']['CHANNEL_ORDER']


				if g.params['DEFAULT_OUTPUT_PARAMS_FOR_EACH_STEP']['OUTPUT']['CALC_SAMPLES_STATS']:

					if not 'SamplesStatsPerChan' in g.dat[ ID ].keys():
						g.dat[ID]['SamplesStatsPerChan'] = {}
						g.dat[ID]['SamplesStatsAllChans'] = {}
					print( "g.cur['STEP']['OUTPUT']['DIR']: ", g.cur['STEP']['OUTPUT']['DIR'])
					g.dat[ ID ][ 'SamplesStatsPerChan' ][ g.cur['STEP']['OUTPUT']['DIR'] ], g.dat[ ID ][ 'SamplesStatsAllChans' ][ g.cur['STEP']['OUTPUT']['DIR'] ] = calc_samples_stats( g.dat[ ID ]['Samples'] )


	if not 'ProcessInfo' in g.dat[ID].keys() :

		g.dat[ID]['ProcessInfo'] = {}
		g.dat[ID]['ProcessInfo']['STEPS'] = []
		g.dat[ID]['ProcessInfo']['SAMPLES_PROCESSED_LEVEL'] = g.params['GEN']['LOAD_PROCESSED_LEVEL']


	g.dat[ID]['ProcessInfo']['STEPS'].append(copy.copy(g.cur['STEP']))





