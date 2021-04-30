import sys
import inspect
import time
import numpy as np

import scipy
from scipy.stats import kurtosis
from pyutils.time_utils import *

np.set_printoptions(precision=3, edgeitems=3)

def get_mean_outliers_within_upper_percentiles( samples, percentileVal ) :

	startTime = time.time()
	# Get percentile edges
	percentileEdges = np.percentile(samples, [percentileVal, 100 - percentileVal])

	upperPercentiles = np.max(percentileEdges)
	outliers = samples[ np.where(samples > upperPercentiles) ]

	thresholdPoint = np.min( outliers )
	time_taken( startTime, inspect.stack()[0][3] )

	return (-1 * thresholdPoint, thresholdPoint)


def get_std_dev_with_median( chDat, scalar ) :

	# print(chanSlice)
	lowerLimit = np.median( chDat ) - \
				scalar * np.std( chDat )

	upperLimit = np.median( chDat ) + \
				scalar * np.std( chDat )

	return ( lowerLimit, upperLimit )


def get_n_matches(refLbls, predsLbls) :

 	return len( np.where(refLbls==predsLbls)[0] )


def compute_classification_results(trueLabels, predLabels, foldI=None) :

	resultsSummary = {}
	rawResults = {}

	nItems = len(trueLabels)

	## Calc positives.

	refPositiveLocs = np.where( trueLabels==1 )[0]
	predsPositiveLocs = np.where( predLabels==1 )[0]

	TPs = np.zeros(shape=(nItems))
	FPs = np.copy(TPs)

	if len(predsPositiveLocs) > 0 :

		TPlocs = refPositiveLocs[ np.isin( refPositiveLocs, predsPositiveLocs ) ]
		FPlocs = predsPositiveLocs[ np.isin( predsPositiveLocs, refPositiveLocs, invert=True ) ]

		TPs[ TPlocs ] = 1
		FPs[ FPlocs ] = 1

		nTPs = len(TPlocs)
		nFPs = len(FPlocs)


	else :

		TPs = np.array([])
		FPs = np.array([])

		TPlocs = []
		FPlocs = []

		predPositiveLocs = []

		FPlocs = []		

		nTPs = 0
		nFPs = 0


	rawResults['TPs'] = TPs
	rawResults['FPs'] = FPs

	resultsSummary = {}			

	resultsSummary[ 'nTPs' ] = len( TPlocs )
	resultsSummary[ 'nFPs' ] = len( FPlocs )

	resultsSummary[ 'nML_Ps' ] = len( predsPositiveLocs )
	resultsSummary[ 'nAct_Ps' ] = len( refPositiveLocs )


	## Calc negatives.
	TNs = np.zeros( shape=(nItems) )
	FNs = np.copy( TNs )

	refNegLocs = np.where( trueLabels==0 )[0]
	predsNegLocs = np.where( predLabels==0 )[0]	

	if len(predsNegLocs) > 0 :

		TNlocs = refNegLocs[ np.isin( refNegLocs, predsNegLocs ) ]
		FNlocs = predsNegLocs[ np.isin( predsNegLocs, refNegLocs, invert=True ) ]

		TNs[ TNlocs ] = 1
		FNs[ FNlocs ] = 1

		nTNs = len( TNlocs )
		nFNs = len( FNlocs )

	else :

		TNs = np.array([])
		FNs = np.array([])

		predsNegLocs = []		

		nTNs = 0
		nFNs = 0


	rawResults['TNs'] = TNs
	rawResults['FNs'] = FNs

	resultsSummary['nTNs'] = nTNs
	resultsSummary['nFNs'] = nFNs	

	resultsSummary['nML_Ns'] = len( predsNegLocs )

	resultsSummary['nAct_Ns'] = len( refNegLocs )

	resultsSummary['n'] = nItems

	overallAcc = np.round( ( np.sum( trueLabels==predLabels ) / nItems * 100 ), 2 )
	accChecking = (nTNs + nTPs) / (nTPs + nFPs + nTNs + nFNs) * 100

	if nTPs > 0:
		resultsSummary['Sensitivity'] = nTPs / (nTPs + nFNs)
		resultsSummary['Precision'] = nTPs / (nTPs + nFPs)
		resultsSummary['F1Score'] = (2 * nTPs) / (2 * nTPs + nFPs + nFNs)

	else :
		resultsSummary['Sensitivity'] = 0
		resultsSummary['Precision'] = 0	


	if nTNs > 0 :
		resultsSummary['Specificity'] = nTNs / (nTNs + nFPs)

	else :
		resultsSummary['Specificity'] = 0	


	resultsSummary['Acc'] = overallAcc
	resultsSummary['AccCheck'] = accChecking

	if not foldI==None :
		resultsSummary['Fold'] = foldI

	resultsSummary['AsStr'] = 'Prec_' + str( resultsSummary['Precision'] ) + "_Sens_" + str( resultsSummary['Sensitivity'] ) +  "_Spec_" + str( resultsSummary['Specificity'] ) + "\n"

	return rawResults, resultsSummary



def compute_classification_avg( inList ) :
	newDict = {}
	listI = 0

	for item in inList :
		listI += 1

		for khi, val in item.items() :
			if khi[0]=='n' :
				newKhiStr = khi + "_tot"
			else:
				newKhiStr = khi + "_avg"
			if isinstance(val, int) or isinstance(val, float) or 'ndarray' in str(type(val)) :
				if newKhiStr not in newDict.keys() :
					newDict[ newKhiStr ] = np.array( val )

				else:
					newDict[ newKhiStr ] = np.append( newDict[ newKhiStr ], val )			

	for key, val in newDict.items() :
		if 'ndarray' in str( type( val ) ) :
			if key[0]=='n' :
				newDict[ key ] = np.sum( val)

			else :
				newDict[ key ] = np.mean( val )

	return newDict

def calc_samples_stats( inSamples ) :

	perChanStats = {}

	perChanStats['Min'] = np.array([])
	perChanStats['Max'] = np.array([])
	perChanStats['Mean'] = np.array([])
	perChanStats['UpperPercentile5'] = np.array([])
	perChanStats['StdDev'] = np.array([])
	perChanStats['Kurtosis'] = np.array([])

	for row in range( 0, inSamples.shape[0] ) :

		chanRow = inSamples[row, :]
		perChanStats['Min'] = np.append( perChanStats['Min'], np.min( chanRow ))
		perChanStats['Max'] = np.append( perChanStats['Max'], np.max( chanRow ))
		perChanStats['Mean'] = np.append( perChanStats['Mean'], np.mean( chanRow ))
		perChanStats['UpperPercentile5'] = np.append( perChanStats['UpperPercentile5'], get_mean_outliers_within_upper_percentiles( chanRow, 5 )[1])
		perChanStats['StdDev'] = np.append( perChanStats['StdDev'], np.std( chanRow ))
		perChanStats['Kurtosis'] = np.append( perChanStats['Kurtosis'], scipy.stats.kurtosis( chanRow ) )


	flattenedData = inSamples.flatten()
	allSamplesStats = {}

	allSamplesStats[ 'Kurtosis' ] = scipy.stats.kurtosis( flattenedData )
	allSamplesStats[ 'Min' ] = np.min( flattenedData )
	allSamplesStats[ 'Max' ] = np.max( flattenedData )
	allSamplesStats[ 'Mean' ] = np.mean( flattenedData )
	allSamplesStats[ 'UpperPercentile5' ] = get_mean_outliers_within_upper_percentiles( flattenedData, 5 )[1]
	allSamplesStats[ 'StdDev' ] = np.std( flattenedData )
	# print("allSamplesStats dict: ", allSamplesStats)
	# quit()
	return perChanStats, allSamplesStats














