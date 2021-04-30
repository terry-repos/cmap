import sys
import os
import copy 

import numpy as np

from scipy import signal
import sklearn
from sklearn import preprocessing

sys.path.append('..')
from global_imports import external_modules
from global_imports import session_vars as g
from global_imports.simplify_vars import *

from algorithms.stats import *
from algorithms.preprocessing import *



def clean_samples( ) :

	g.command_history.add('', os.path.basename(__file__),g.inspect.stack()[0][3], locals())

	cleanParams = make_params_readable()
	ID = get_curr_ID()
	sl = get_curr_sl()
	nChans = get_nChans()
	# print("sl: ", sl, " nChans: ", nChans, "samples.shape: ", g.dat[ID]['Samples'].shape)
	
	if cleanParams['METHODS'] == 'REMOVE_BASELINE' :
		remove_baseline(ID, cleanParams, sl)

	if cleanParams['METHODS'] == 'REMOVE_BASELINE_SLIDING' :
		remove_baseline_sliding(ID, cleanParams, sl)		

	if cleanParams['METHODS'] == 'NORMALISE' :
		normalise(ID, cleanParams, sl)		

	elif cleanParams['METHODS'] == 'CLIP_EXTREME_VALS' :
		clip_extreme_vals(ID, cleanParams, sl)

	elif cleanParams['METHODS'] == 'REMOVE_ARTIFACT_COMPONENTS' :
		remove_artifact_components(ID, cleanParams, sl)

	elif cleanParams['METHODS'] == 'REMOVE_BAD_CHANS' :
		remove_bad_chans( ID, cleanParams, sl )	

	elif cleanParams['METHODS'] == 'REMOVE_SYNC_NOISE' :
		remove_sync_noise( ID, cleanParams, sl )	

	elif cleanParams['METHODS'] == 'TEMPORAL_FILTER' :
		filter_temporal(ID, cleanParams, sl)

	elif cleanParams['METHODS'] == 'SPATIAL_FILTER' :
		filter_spatial(ID, cleanParams, sl)		

	elif cleanParams['METHODS'] == 'FILTER' :
		filter_signal(ID, cleanParams, sl)	

	elif cleanParams['METHODS'] == 'ABSOLUTE_VALUE' :
		absolute_value(ID, cleanParams, sl)		




def remove_bad_chans( ID, params, sl ) :

	if params['BAD_CHAN_DETECTION_METHOD'] == 'KURTOSIS' :
		kurtosisArr = scipy.stats.kurtosis( g.dat[ID]['Samples'][sl], axis=1 )
		print( "kurtosisArr: ", kurtosisArr )
		print( "kurtosisArr.shape: ", kurtosisArr.shape )
	

	elif params['BAD_CHAN_DETECTION_METHOD'] == 'VAR' :

		print("np variance: ", np.var(np.var(g.dat[ID]['Samples'][sl])))
		print("np std: ", np.std( g.dat[ID]['Samples'][sl])) 
		print("std median dev: ", get_std_dev_with_median( g.dat[ID]['Samples'][sl], 4)) 	



def normalise( ID, params, sl ) :
	# print("BEFORE g.dat[ID]['Samples'][sl]: ", g.dat[ID]['Samples'][sl] )
	# print(sl)
	if params['SUB_METHOD'] == 'MIN_MAX_SCALER' :
		locsOfNonZeros = np.where( (g.dat[ID]['Samples'][sl] < 0) | (g.dat[ID]['Samples'][sl] > 0) )
		# print("locsOfNonZeros: ", locsOfNonZeros)
		if len(locsOfNonZeros[0]) > 0:
			# print("g.dat[ID]['Samples'][",sl,"]: ", g.dat[ID]['Samples'][sl].shape)
			g.dat[ID]['Samples'][sl] = sklearn.preprocessing.MinMaxScaler().fit_transform( g.dat[ID]['Samples'][sl][0] )

	elif params['SUB_METHOD'] == 'DIVISIVE' :
		g.dat[ID]['Samples'][sl] = ( g.dat[ID]['Samples'][sl] - np.min(g.dat[ID]['Samples'][sl]) ) / ( np.max(g.dat[ID]['Samples'][sl]) - np.min(g.dat[ID]['Samples'][sl]) )



def remove_baseline( ID, params, sl ) : 

	if params['SUB_METHOD'].upper() == 'DETREND':
		g.dat[ID]['Samples'][sl] = signal.detrend(g.dat[ID]['Samples'][sl])



def remove_baseline_sliding( ID, params, sl ) : 

	print("removing baseline for chan: ", sl[0])
	if params['SUB_METHOD'].upper() == 'ENVELOP_SECANT':
		removalData = get_envelope_secant(g.dat[ID]['Samples'][sl], params['WINDOW_SIZE'])

	g.dat[ ID ][ 'Samples' ][ sl ] = np.subtract( g.dat[ ID ][ 'Samples' ][ sl ], removalData)
	# print("after removal: ", g.dat[ ID ][ 'Samples' ][ sl ])



def remove_sync_noise( ID, params, sl ) : 

	if params['SUB_METHOD'].upper() == 'MEDIAN':
		print("Attempting to remove synchronous noise!")
		g.dat[ID][ 'Samples' ][ sl ] = g.dat[ID][ 'Samples' ][ sl ] - np.median( g.dat[ID][ 'Samples' ][ sl ], axis=0)



def filter_temporal( ID, params, sl ) : 

	if params['SUB_METHOD'].upper() == 'SAVITZKY_GOLAY':
		if params["WINDOW_SIZE"] % 2 == 0:
			params["WINDOW_SIZE"] += 1		
		g.dat[ID][ 'Samples' ][ sl ] = scipy.signal.savgol_filter( g.dat[ID][ 'Samples' ][ sl ], params["WINDOW_SIZE"], params["P_ORDER"] )



def filter_spatial( ID, params, sl ) : 

	if params['SUB_METHOD'].upper() == 'WIENER':
		g.dat[ID][ 'Samples' ][ sl ] = scipy.signal.wiener( g.dat[ID][ 'Samples' ][ sl ], params["WINDOW_SIZE"] )



def clip_extreme_vals( ID, params, sl ) : 
	if params[ 'SUB_METHOD' ].upper() == 'STD_DEV_FROM_MEDIAN':
		extremeVals = get_std_dev_from_median(sl, params['SCALAR'])

	if params[ 'SUB_METHOD' ].upper() == 'OUTLIER_PERCENTILE':
		extremeVals = get_mean_outliers_within_upper_percentiles( g.dat[ ID ][ 'Samples' ][ sl ], params['OUTLIER_PERCENTILE'] )	
		# print(extremeVals)	

	g.dat[ID][ 'Samples' ][ sl ] = np.clip(g.dat[ID][ 'Samples' ][ sl ], extremeVals[0], extremeVals[1])


def absolute_value( ID, params, sl ) : 
	g.dat[ID][ 'Samples' ][ sl ] = np.absolute(g.dat[ID][ 'Samples' ][ sl ])	




def remove_baseline_envelop_secant( ID, params, sl ):
	# (allSamples - min(sample) / (max(sample) - min(sample)))
	envelopeBaseline = get_envelope_secant(g.dat[ID]['Samples'][sl],params['WINDOW_SIZE']  )
	g.dat[ID]['Samples'][sl] = (g.dat[ID]['Samples'][sl] - np.min(g.dat[ID]['Samples'][sl])) \
								/ (np.max(g.dat[ID]['Samples'][sl]) - np.min(g.dat[ID]['Samples'][sl]))	



	



