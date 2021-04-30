import sys
import os
import copy 
import time
import inspect
import gc

import numpy as np

from scipy import signal

sys.path.append('..')
from global_imports import external_modules
from global_imports import session_vars as g
from global_imports.simplify_vars import *

from pyutils.dict_utils import *
from pyutils.time_utils import *

from algorithms.stats import *
from algorithms.grouping_events import *



from hierarchies.hierarchies_events import *
from hierarchies.hierarchies import *

from hierarchies.indexing_events import *
from hierarchies.indexing_wins import *

from steps.helpers.classification_helpers import *


def detect(shouldMatchEvents=False) :

	g.command_history.add('', os.path.basename(__file__),g.inspect.stack()[0][3], locals())

	if shouldMatchEvents:
		g.cur['SLICE'] = copy.copy( np.s_[::] )

	detectParams = make_params_readable()
	ID = get_curr_ID()
	sl = get_curr_sl()
	nChans = get_nChans()

	eventIndices = np.array([])
	copyHier = False

	if 'IN_FIRST_CHANNEL' in g.cur.keys() :
		if g.cur['IN_FIRST_CHANNEL']:
			print("shouldMatchEvents: ", shouldMatchEvents, ID, " ", sl, " detectParams: ", detectParams)
			copyHier=True
	else:
		print("shouldMatchEvents: ", shouldMatchEvents, ID, " ", sl, " detectParams: ", detectParams)
		copyHier=True


	if 'Events' not in g.dat[ID].keys() :
		g.dat[ID]['Events'] = {}

	if detectParams['METHODS'] == 'PEAKS' :
		eventIndices = find_peaks( ID, detectParams, sl )
		# g.dat[ID]['Events'] =  set_indices_within_dict( g.dat[ID]['Events'], detectParams['OUTPUT_EVENTS_HIERARCHY'], sl[0], eventIndices, get_nChans() )
	elif detectParams['METHODS'] == 'PEAKS_ABOVE_THRESHOLD' :
		eventIndices = find_peaks_above_threshold( ID, detectParams, sl, nChans )

	elif detectParams['METHODS'] == 'PEAKS_WITHIN' :
		eventIndices = find_peaks_within( ID, detectParams, sl )

	elif detectParams['METHODS'] == 'INTERSECTING_EVENTS' :
		eventIndices = find_intersecting_events( ID, detectParams, sl )

	elif detectParams['METHODS'] == 'MIDDLE_OF_WAVES' :
		eventIndicesDict, paramsSummaryStr = find_middle_of_waves( ID, detectParams, sl )
		insertPos = len( detectParams['OUTPUT_EVENTS_HIERARCHY'] ) - 1
		detectParams['OUTPUT_EVENTS_HIERARCHY'].insert( insertPos, paramsSummaryStr )

	elif detectParams['METHODS'] == 'MID_WAVES_PARAM_SEARCH' :
		eventIndicesDict, referenceIndices = find_mid_waves_param_search( ID, detectParams, sl, nChans )
		referenceHier = copy.deepcopy( detectParams['REFERENCE_EVENTS_HIERARCHIES'] )
		detectParams[ 'OUTPUT_EVENTS_HIERARCHY' ] = copy.deepcopy( get_hierarchies_from_dict( eventIndicesDict, hierToAppend=referenceHier, wrapHierInList=True, allHiers=[], currHier=[]) )
		# print( "detectParams[ 'OUTPUT_EVENTS_HIERARCHY' ]: ", detectParams[ 'OUTPUT_EVENTS_HIERARCHY' ] )

	elif detectParams['METHODS'] == 'MATCH_SINGULAR_EVENTS' or shouldMatchEvents == True :
		eventIndicesDict, referenceIndices = match_singular_events( ID, detectParams, sl, nChans, shouldMatchEvents )
		referenceHier = copy.deepcopy( detectParams['REFERENCE_EVENTS_HIERARCHIES'] )
		detectParams[ 'OUTPUT_EVENTS_HIERARCHY' ] = copy.deepcopy( get_hierarchies_from_dict( eventIndicesDict, hierToAppend=referenceHier, wrapHierInList=True, allHiers=[], currHier=[]) )
		print( "detectParams[ 'OUTPUT_EVENTS_HIERARCHY' ]: ", detectParams[ 'OUTPUT_EVENTS_HIERARCHY' ] )		

	elif detectParams['METHODS'] == 'NEAREST' :
		eventIndices = find_nearest( ID, detectParams, sl )		

	elif detectParams['METHODS'] == 'OUTSIDE_THRESHOLD' :
		eventIndices, hier = find_outside_threshold( ID, detectParams, sl, nChans )

	elif detectParams['METHODS'] == 'WITHOUT_NEIGHBOURS' :
		eventIndices = find_events_without_neighbours( ID, detectParams, sl )

	elif detectParams['METHODS'] == 'WITHIN_AMPLITUDE' :
		eventIndices = find_events_within_amplitude( ID, detectParams, sl, nChans )		

	elif detectParams['METHODS'] == 'THRESHOLD_INTERCEPTS' :
		eventIndices, hier = find_threshold_intercepts( ID, detectParams, sl, nChans )

	elif detectParams['METHODS'] == 'START_END_INTERCEPTS' :
		eventIndices1, eventIndices2, hier = find_start_end_intercepts( ID, detectParams, sl )

	elif detectParams['METHODS'] == 'START_END_PAIRS' :
		eventIndices1, eventIndices2, hier = find_start_end_pairs( ID, detectParams, sl )

	elif detectParams['METHODS'] == 'MID_POINT' :
		eventIndices, hier, plotHier = find_mid_point_of_startEndPairs( ID, detectParams, sl )

	elif detectParams['METHODS'] == 'ON_INCLINE' :
		eventIndices = find_events_on_slope( ID, detectParams, sl, onIncline=True )	

	elif detectParams['METHODS'] == 'ON_DECLINE' :
		eventIndices = find_events_on_slope( ID, detectParams, sl, onIncline=False )			

	elif detectParams['METHODS'] == 'IN_LOCAL_MINIMA' :
		eventIndices = find_events_in_local_minima( ID, detectParams, sl )					

	elif detectParams['METHODS'] == 'ADJACENT_TO_PEAK' :
		eventIndices = find_events_adjacent_to_peaks( ID, detectParams, sl )	

	elif detectParams['METHODS'] == 'ISOLATE_WITHIN_GROUP' :
		eventIndices = find_single_event_in_group_of_neighbours( ID, detectParams, sl )	

	elif detectParams['METHODS'] == 'EVERY_X_SAMPLES' :
		eventIndices = find_events_every_x_samples( ID, detectParams, sl )	

	elif detectParams['METHODS'] == 'AWAY_FROM' :
		eventIndices = find_events_away_from( ID, detectParams, sl )
		
	elif detectParams['METHODS'] == 'GROUP_PROPAGATING' :
		eventIndicesDict = group_propagating( ID, detectParams, sl, nChans )
		# referenceHier = copy.deepcopy( detectParams['REFERENCE_EVENTS_HIERARCHIES'] )
		print_keys_hierarchy( eventIndicesDict, "eventIndicesDict" )
		detectParams[ 'OUTPUT_EVENTS_HIERARCHY' ] = copy.deepcopy( get_hierarchies_from_dict( eventIndicesDict, hierToAppend=[], wrapHierInList=True, allHiers=[], currHier=[]) )
		print( "detectParams[ 'OUTPUT_EVENTS_HIERARCHY' ]: ", detectParams[ 'OUTPUT_EVENTS_HIERARCHY' ] )

	if copyHier :
		# print("in copyhier: ", )
		if 'hier' in locals():
			detectParams[ 'OUTPUT_EVENTS_HIERARCHY' ] = copy.deepcopy( hier )	
			print( "in hier: ", hier )

		# print("about to copy detected hierarchy to plotting hierarchy")
		if not 'EVENT_HIERARCHIES_TO_PLOT' in g.cur['STEP']['OUTPUT']['PLOT_CHANS'].keys() :
			g.cur['STEP']['OUTPUT']['PLOT_CHANS']['EVENT_MARKER_PARAMS']['EVENT_HIERARCHIES_TO_PLOT'] = [[copy.deepcopy(detectParams['OUTPUT_EVENTS_HIERARCHY'])]]
			
		else:
			g.cur['STEP']['OUTPUT']['PLOT_CHANS']['EVENT_MARKER_PARAMS']['EVENT_HIERARCHIES_TO_PLOT'] += [[copy.deepcopy(detectParams['OUTPUT_EVENTS_HIERARCHY'])]]
		
		if 'plotHier' in locals():
			g.cur['STEP']['OUTPUT']['PLOT_CHANS']['EVENT_MARKER_PARAMS']['EVENT_HIERARCHIES_TO_PLOT'] = [[ plotHier ]]

		eventsHierAsStr = list_as_str(detectParams['OUTPUT_EVENTS_HIERARCHY'])

		g.cur['STEP']['PARAMS']['OUTPUT_EVENTS_HIERARCHY'] = copy.deepcopy( detectParams[ 'OUTPUT_EVENTS_HIERARCHY' ] )
		print("after output event hier copy g.cur['STEP']: ", g.cur['STEP'])
	# if not check_hierarchy_in_dict( g.dat[ ID ][ 'Events' ], detectParams['OUTPUT_EVENTS_HIERARCHY'] ) :
	# 	g.dat[ ID ][ 'Events' ] = update_dict_with_a_new_initialised_hierarchy( dict( g.dat[ID]['Events']) , detectParams['OUTPUT_EVENTS_HIERARCHY'] )	

	# print(eventsHierAsStr, " nChans: ", nChans, " sl: ", sl, "eventIndices.shape: ", eventIndices.shape, " samples.shape: ", g.dat[ID]['Samples'].shape)
	if eventIndices.shape[0] > 0 :
		g.dat[ ID ]['Events'] = update_dict_with_a_new_ndarray(  g.dat[ID]['Events'], detectParams['OUTPUT_EVENTS_HIERARCHY'], eventIndices, sl, get_nChans()  )

	if 'eventIndices1' in locals():
		# print("detectParams['OUTPUT_EVENTS_HIERARCHY']	", detectParams['OUTPUT_EVENTS_HIERARCHY'])

		if eventIndices1.shape[0] > 0 :
			g.dat[ ID ]['Events'] = update_dict_with_a_new_ndarray(  g.dat[ID]['Events'], detectParams['OUTPUT_EVENTS_HIERARCHY'][0][0], eventIndices1, sl, get_nChans()  )

	
	if 'eventIndices2' in locals():

		if eventIndices2.shape[0] > 0 :
			g.dat[ ID ]['Events'] = update_dict_with_a_new_ndarray(  g.dat[ID]['Events'], detectParams['OUTPUT_EVENTS_HIERARCHY'][0][1], eventIndices2, sl, get_nChans()  )

	if 'eventIndicesDict' in locals():
		g.dat[ ID ][ 'Events' ] = merge_dicts( g.dat[ ID ]['Events'], eventIndicesDict )

	if 'referenceIndices' in locals():
		# print(referenceHier)
		if not check_hierarchy_in_dict_explicit( g.dat[ ID ][ 'Events' ], referenceHier ) :
			print("Hier ", referenceHier  ,"not in dict creating!")
			if list_contains_list(referenceHier):
				referenceHier = referenceHier[0]
			# print("referenceIndices.shape: ", referenceIndices.shape)
			g.dat[ ID ][ 'Events' ] = update_dict_with_a_new_ndarray(  g.dat[ID]['Events'], referenceHier, referenceIndices,  sl, get_nChans()  )
		g.cur['STEP']['OUTPUT']['PLOT_CHANS']['EVENT_MARKER_PARAMS']['EVENT_HIERARCHIES_TO_PLOT'] += [[ copy.deepcopy( referenceHier ) ]]




def insert_group_event_hierarchies( dataEventsDict, baseHierarchy, groupIndicesDict ) :
	pass 



def group_propagating( ID, params, sl, nChans, fromTheseIndices=np.array([]) ) : 

	startTime = time.time()
	print("ID: ", ID, "fromTheseIndices.shape: ", fromTheseIndices.shape)

	if 'candidateIndcs' in locals() :
		print("candidateIndcs.shape: ", candidateIndcs.shape)

	# print("GROUP_PROPAGATING locals(): ", locals())
	if len( fromTheseIndices ) == 0 :
		if 'EVENTS_FROM_THIS_HIERARCHY' in params.keys() :
			if 'GET_DIRECTION' in params['SUB_METHOD']:
				candidateIndcs = get_vals_from_dict_with_this_hierarchy( g.dat[ID]['Events'], params['EVENTS_FROM_THIS_HIERARCHY'] )

			else:
				candidateIndcs = get_indices_that_contains_all_specified_hierarchies( dict(g.dat[ID]['Events']), list(params['EVENTS_FROM_THIS_HIERARCHY']), sl, [] )

	if len( candidateIndcs ) == 0 :
		return np.array( [] )

	# print("candidateIndcs: ", candidateIndcs)
	groupedIndicesSet = group_propagating_events( g.dat[ID]['Samples'][sl], params, candidateIndcs, nChans )
	outputGrouped = {}
	outputGrouped[ 'Detect' ] = {}
	outputGrouped[ 'Detect' ][ 'GroupPropagating' ] = groupedIndicesSet

	print("groupedIndicesSet: ", groupedIndicesSet)
	# print("groupsPropsDict: ", groupsPropsDict)

	time_taken( startTime, inspect.stack()[0][3] )

	return outputGrouped



def group_propagating_param_search(ID, params, sl, nChans, fromTheseIndices=np.array([])) : 

	startTime = time.time()
	outEventsDict = stack_list_as_hierarchical_dict( params["OUTPUT_EVENTS_HIERARCHY"], {} )

	print("In group_propagating_param_search!")
	paramsList = build_group_propagating_param_variations(params)
	print("ID: ", ID, "fromTheseIndices.shape: ", fromTheseIndices.shape)

	if 'candidateIndcs' in locals():
		print("candidateIndcs.shape: ", candidateIndcs.shape)

	# print("GROUP_PROPAGATING locals(): ", locals())
	if len( fromTheseIndices ) == 0 :
		if 'EVENTS_FROM_THIS_HIERARCHY' in params.keys() :
			candidateIndcs = get_indices_that_contains_all_specified_hierarchies( dict(g.dat[ID]['Events']), list(params['EVENTS_FROM_THIS_HIERARCHY']), sl, [] )

	if len( candidateIndcs ) == 0 :
		return np.array( [] )

	referenceIndices = get_indices_that_contains_all_specified_hierarchies( g.dat[ID]['Events'], params['REFERENCE_EVENTS_HIERARCHIES'], sl, explicit=False )

	print("referenceIndices: ", referenceIndices)
	indicesDict = {}

	for pars in paramsList :
		# print("pars: ", pars, " sl: ", sl)

		correctIndices = get_init_channel_array(nChans)
		allGroupDetectedIndices = get_init_channel_array(nChans)

		# indicesDict[ paramsSummaryStrs ] = {}
		# indicesDict[ paramsSummaryStrs ][ 'CorrectIndex' ] = get_init_channel_array(nChans)
		# indicesDict[ paramsSummaryStrs ][ 'AllIndex' ] = get_init_channel_array(nChans)
		groupedIndicesSet, groupsPropsDict = group_propagating_events( g.dat[ID]['Samples'][sl], pars, candidateIndcs, nChans )
		print_keys_hierarchy(groupedIndicesSet)

		for chanNum in range(0,nChans) :

			slInSl = np.s_[chanNum,::]

			groupDetectedIndices = get_indices_that_contains_all_specified_hierarchies( groupedIndicesSet, ['GroupIndex'], slInSl, explicit=False )
			allGroupDetectedIndices = insert_1d_input_arr_into_2d_arr(chanNum, allGroupDetectedIndices, groupDetectedIndices)
			# print("groupDetectedIndices: ", groupDetectedIndices)
			if len( groupDetectedIndices ) > 0 :
				refIndices = map_npwhere_to_event_structured_indices_by_row(referenceIndices, chanNum)

				nearestIndices, refMatches = find_nearest( ID, pars, slInSl, fromTheseIndices=groupDetectedIndices, nearestTheseIndices=refIndices ) 
				correctIndices = insert_1d_input_arr_into_2d_arr(chanNum, correctIndices, nearestIndices)

			gc.collect()

		nCorrect = get_num_non_x_items(correctIndices, -1)
		nIncorrect = get_num_non_x_items(allGroupDetectedIndices, -1) - nCorrect
		perCorrect = round( (nCorrect / len(referenceIndices[0]) * 100), 2)

		if nCorrect > 0:
			ratioInToCor = round( (nIncorrect / nCorrect), 2)
		else:
			ratioInToCor = 999999

		paramsSummaryStrs = "nghSpn" + str( pars['NEIGHBOUR_SPAN'] ) + "nChnSub" + str(pars["MIN_CHANS_TO_FORM_SUB_GROUP"]) + "nChnGrp" + str(pars["MIN_CHANS_TO_FORM_GROUP"]) + "delay" + str(pars["MIN_DELAY"]) + "-" + str(pars["MAX_DELAY"]) + "corPer" + str(perCorrect) + "rat" + str(ratioInToCor)
		indicesDict[ paramsSummaryStrs ] = {}
		indicesDict[ paramsSummaryStrs ][ 'CorrectIndex' ] = correctIndices
		indicesDict[ paramsSummaryStrs ][ 'AllIndex' ] = allGroupDetectedIndices
		print("paramsSummaryStrs: ", paramsSummaryStrs)


	outEventsDict = set_childmost_value_from_hierarchical_dict( outEventsDict, indicesDict )
	print_keys_hierarchy(outEventsDict, "outEventsDict")
	print("referenceIndices: ", referenceIndices)
	outReferenceIndices = map_npwhere_to_event_structured_indices(referenceIndices, nChans)
	print("outReferenceIndices: ", outReferenceIndices)
	time_taken(startTime, inspect.stack()[0][3])
	return outEventsDict




def find_peaks(ID, params, sl) : 

	# print(g.dat[ID]['Samples'][sl].shape)
	# print(sl)
	return signal.find_peaks_cwt( g.dat[ID]['Samples'][sl], params['WIDTHS_OF_PEAKS'] )



def find_peaks_above_threshold(ID, params, sl, nChans) : 
	startTime = time.time()
	# print(g.dat[ID]['Samples'][sl].shape)
	# print(sl)
	indicesAboveThreshold, hier = find_threshold_intercepts( ID, params, sl ) 
	# print("indicesAboveThreshold.shape: ", indicesAboveThreshold.shape)
	startIndices, endIndices, thirdHier = find_start_end_intercepts( ID, params, sl, indicesAboveThreshold ) 

	if (len(list(startIndices)) > 0) and (len(list(endIndices)) > 0) :
		peakIndices = find_peaks_within(ID, params, sl, list(startIndices), list(endIndices))

	else :
		peakIndices = np.array([])

	time_taken(startTime, inspect.stack()[0][3])

	return peakIndices



def find_intersecting_events(indicesOne, indicesTwo ) :
	return indicesOne[np.where(indicesOne==indicesTwo)]



def find_peaks_within(ID, params, sl, startIndices=[], endIndices=[]) : 
	# print(sl)
	# print("Find peaks within: ", sl)
	# print("	print_keys_hierarchy(evts):")
	# print_keys_hierarchy(copy.copy(g.dat[ID]['Events']))
	if len(startIndices)==0 or len(endIndices)==0:
		startIndices, endIndices = get_start_end_event_windows( copy.deepcopy(g.dat[ID]['Events']), copy.copy(sl), params)

	eventWindows = []

	# print("startIndices: ", startIndices)
	# print("endIndices: ", endIndices)

	for sEvt in startIndices:
		endEvtI = 0
		for eEvt in endIndices :
			endEvtI += 1
			dist = eEvt - sEvt
			print(dist)
			if ( dist > params['MIN_WAVE_WIDTH']) and (dist < params['MAX_WAVE_WIDTH']) :
				endIndices = copy.deepcopy(endIndices[endEvtI:])
				eventWindows.append( [sEvt, eEvt] )						
				break
	
	sampShape =  g.dat[ID]['Samples'][sl[0]].shape[0]

	peakIndices = np.array([])
	if len(eventWindows) > 0:
		for win in eventWindows:
			print("evntWin: ", win)
			# print("samples shape: ", g.dat[ID]['Samples'][sl[0]].shape)

			# print("g.dat[ID]['Samples'][sl[0], win[0]:win[1]]: ", g.dat[ID]['Samples'][sl[0]].shape, win[0]:win[1]])
			peakIndex = np.argmax(g.dat[ID]['Samples'][sl[0], win[0]:win[1]]) + win[0]
			# print("argmax peakIndex: ", peakIndex)
			peakIndices = np.append(peakIndices, peakIndex.astype(int))

	return peakIndices



def find_mid_point_of_startEndPairs( ID, params, sl, fromTheseStartIndices=np.array([]), fromTheseEndIndices=np.array([]) ) :
	startTime = time.time()

	if len(fromTheseStartIndices) == 0 :
		if 'START_EVENTS_HIERARCHY' in params.keys():
			fromTheseStartIndices = get_indices_that_contains_all_specified_hierarchies( dict(g.dat[ID]['Events']), list( params['START_EVENTS_HIERARCHY']), sl )

	if len(fromTheseEndIndices) == 0 :
		if 'END_EVENTS_HIERARCHY' in params.keys():
			fromTheseEndIndices = get_indices_that_contains_all_specified_hierarchies( dict(g.dat[ID]['Events']), list( params['END_EVENTS_HIERARCHY']), sl )
	
	midEvents = np.array([])
	pairs = np.vstack((fromTheseStartIndices,fromTheseEndIndices))
	samps = g.dat[ID]['Samples'][sl]

	for pairCol in range(pairs.shape[1]) :

		startEvent = int(pairs[0,pairCol])
		endEvent = int(pairs[1,pairCol])		
		midEvent = int(round((endEvent - startEvent)/2) + pairs[0,pairCol])

		if params["MID_POINT_NEED_BE_GREATER_THAN"]:
			# print(midEvent)
			# print("samps.shape: ", samps.shape)
			midEventVal = samps[ midEvent ]
			# print("midEventVal: ", midEventVal)
			startEventVal = samps[ startEvent ]
			endEventVal = samps[ endEvent ]

			# print("startEventVal: ", startEventVal," endEventVal: ", endEventVal, " midEventVal: ", midEventVal)	

			if midEventVal > endEventVal and midEventVal > startEventVal:
				midEvents = np.append(midEvents, midEvent)
		else:
			midEvents = np.append(midEvents, midEvent)


	newOutputEventHier = params["OUTPUT_EVENTS_HIERARCHY"]
	startEventHier = copy.deepcopy(params['START_EVENTS_HIERARCHY'])
	midEventHier = copy.deepcopy(newOutputEventHier) + ["PeakIndex"]
	endEventHier = copy.deepcopy(params['END_EVENTS_HIERARCHY'])

	if params["INCLUDE_START_END_PAIRS_IN_PLOTS"]:
		plottingHier = [[startEventHier, midEventHier, endEventHier]]
	else:
		plottingHier = [[ midEventHier]]
	
	# print("plottingHier: ", plottingHier)
	# quit()
	# print("outputHier: ", outputHier)
	# quit()
	# print("startIndices: ", startIndices)
	time_taken(startTime, inspect.stack()[0][3])

	return midEvents, midEventHier, plottingHier



def find_nearest( ID, params, sl, fromTheseIndices=np.array([]), nearestTheseIndices=np.array([]), returnNearest=True ) :
	# print_keys_hierarchy(g.dat[ID]['Events'])
	# print(params)
	startTime = time.time()
	nearestEvents = np.array([])	
	referenceMatchedEvents = np.array([])	


	if len(nearestTheseIndices) == 0 :
		if 'NEAREST_THESE_EVENTS_HIERACHY' in params.keys():
			nearestTheseIndices = get_indices_that_contains_all_specified_hierarchies( dict(g.dat[ID]['Events']), list(params['NEAREST_THESE_EVENTS_HIERACHY']), sl )
		else:
			return nearestEvents, referenceMatchedEvents
	# print("nearestTheseIndices: ", nearestTheseIndices)

	if len(fromTheseIndices) == 0 :
		fromTheseIndices = get_indices_that_contains_all_specified_hierarchies( dict(g.dat[ID]['Events']), list(params['KEEP_THESE_EVENTS_HIERARCHY']), sl )
	# print("fromTheseIndices: ", fromTheseIndices)


	if len(nearestTheseIndices) > 0 :
		if 'MAX_DISTANCE_FROM_REFERENCE_EVENT' in params.keys() :
			maxDist = params[ 'MAX_DISTANCE_FROM_REFERENCE_EVENT' ]
		else:
			maxDist = 75


		for index in fromTheseIndices :
			if len(nearestTheseIndices) == 0:
				break

			dist = nearestTheseIndices-index

			minDist = np.min(np.abs(dist))
			if len(np.where(dist==minDist)[0]) > 0:
				outDist = minDist * -1
			else:
				outDist = minDist 


			if ( minDist < maxDist ) :
				if returnNearest:
					nearestTheseIndices = np.delete( nearestTheseIndices, np.argmin(np.abs(dist)) )

				nearestEvents = np.append(nearestEvents, index)
			
				refIndex = int(index - outDist)

				# print("index", index, " refindex: ", refIndex, "outDist: ", outDist, " minDist: ", minDist)

				referenceMatchedEvents = np.append(referenceMatchedEvents, int(refIndex) )
	
	time_taken(startTime, inspect.stack()[0][3])

	return np.unique(nearestEvents), np.unique(referenceMatchedEvents.astype(int))



def find_events_without_neighbours(ID, params, sl, nChans, fromTheseIndices=np.array([])) : 
	startTime = time.time()
	minGap = params['REFRACTORY_PERIOD']

	if len(fromTheseIndices) == 0 :
		if 'EVENTS_FROM_THIS_HIERARCHY' in params.keys() :
			fromTheseIndices = get_indices_that_contains_all_specified_hierarchies( dict(g.dat[ID]['Events']), list(params['EVENTS_FROM_THIS_HIERARCHY']), sl, [] )
		else:
			fromTheseIndices = np.array([]) 

	if len(fromTheseIndices) > 0 :
		# print("fromTheseIndices: ", fromTheseIndices)
		anteriorDistNeighbs = np.diff( fromTheseIndices ) 
		anteriorSolitaryIndices = np.where( anteriorDistNeighbs > minGap )[0]
		posteriorSolitaryIndices = anteriorSolitaryIndices + 1
	
		solitaryIndices = np.hstack( ( anteriorSolitaryIndices, posteriorSolitaryIndices ) )	

		solitaryIndicesMappedToOriginal = np.unique( np.hstack( ( fromTheseIndices[0], fromTheseIndices[ solitaryIndices ], fromTheseIndices[-1] ) ) )

		# solitaryIndicesMappedToOriginal = fromTheseIndices[ anteriorSolitaryIndices[1] ]

		# if len( anteriorSolitaryIndices ) > 1 :
		# 	# anteriorSolitaryIndices = map_npwhere_to_event_structured_indices( anteriorSolitaryIndices, nChans )
		# 	# posteriorSolitaryIndices = anteriorSolitaryIndices + 1
		# 	# solitaryIndices = np.hstack( ( anteriorSolitaryIndices, posteriorSolitaryIndices ) )			
		# 	solitaryIndicesMappedToOriginal = fromTheseIndices[ anteriorSolitaryIndices[0] ]

		# else :
		# 	anteriorSolitaryIndices = copy.deepcopy( anteriorSolitaryIndices[0] )
		# 	posteriorSolitaryIndices = anteriorSolitaryIndices + 1
		# 	solitaryIndices = np.hstack( ( anteriorSolitaryIndices, posteriorSolitaryIndices ) )
		# 	print("posteriorSolitaryIndices.shape: ", posteriorSolitaryIndices.shape)
		# 	print("fromTheseIndices.shape: ", fromTheseIndices.shape)


		# solitaryIndicesMappedToOriginal = np.unique( np.hstack( ( fromTheseIndices[0], fromTheseIndices[ solitaryIndices ], fromTheseIndices[-1] ) ) )

	else:
		solitaryIndicesMappedToOriginal = np.array([])

	time_taken(startTime, inspect.stack()[0][3])


	return solitaryIndicesMappedToOriginal




def find_events_with_neighbours(ID, params, sl, fromTheseIndices=np.array([])) : 

	# print("Finding events without neighbours")

	maxGap = params['WITHIN_WINDOW']

	if not fromTheseIndices.any() :

		fromTheseIndices = get_indices_that_contains_all_specified_hierarchies( dict(g.dat[ID]['Events']), list(params['EVENTS_FROM_THIS_HIERARCHY']), sl, [] )

	if len( fromTheseIndices ) > 0 :

		anteriorDistNeighbs = np.diff( fromTheseIndices ) 

		anteriorNeighbourIndices = np.where( anteriorDistNeighbs < maxGap )[0]

		posteriorNeighbourIndices = anteriorSolitaryIndices + 1

		socialIndices = np.unique( np.hstack( ( anteriorNeighbourIndices, posteriorNeighbourIndices ) ) )

		socialIndicesMappedToOriginal = fromTheseIndices[ socialIndices ]


	else:
		socialIndicesMappedToOriginal = np.array([])

	return socialIndicesMappedToOriginal



def find_threshold_intercepts(ID, params, sl, nChans) : 

	startTime = time.time()
	parsStr = caps_under_to_cap_lower(str(params["SUB_METHOD"]) + str(params["VALUE"]) + "minGap" + str(params["REFRACTORY_PERIOD"]))
	thresholdInterceptHierarchy = params['OUTPUT_EVENTS_HIERARCHY'] + [parsStr, 'InterceptsIndex']
	indicesAboveThreshold, threshHier = find_outside_threshold( ID, params, sl, nChans )
	# print("indicesAboveThreshold: ", indicesAboveThreshold)

	indicesAboveThresholdAndSolitary = find_events_without_neighbours( ID, params, sl, nChans, indicesAboveThreshold )


	# print("indicesAboveThresholdAndSolitary: ", indicesAboveThresholdAndSolitary)
	# quit()

	time_taken(startTime, inspect.stack()[0][3])

	return indicesAboveThresholdAndSolitary, thresholdInterceptHierarchy



def find_events_within_amplitude( ID, params, sl, nChans ) :

	eventIndices = get_indices_that_contains_all_specified_hierarchies( g.dat[ID]['Events'], params['KEEP_THESE_EVENTS_HIERARCHY'], sl )

	# print("sl: ", sl)
	# print("eventIndices: ", eventIndices)

	eventAmpVals = g.dat[ ID ][ 'Samples' ][sl][ eventIndices ]
	# print("eventAmpVals: ", eventAmpVals)

	if params['SUB_METHOD'].upper() == 'OUTLIER_PERCENTILE' :
		threshold = get_mean_outliers_within_upper_percentiles( eventAmpVals, params['OUTLIER_PERCENTILE'] )

	# print("threshold: ", threshold)

	indicesAboveThreshold = np.where( g.dat[ ID ][ 'Samples' ][ sl ][ eventIndices ] > threshold[1] )

	indicesMappedToEventsStructure = map_npwhere_to_event_structured_indices( [ eventIndices[0][ indicesAboveThreshold ], eventIndices[1][indicesAboveThreshold ] ], nChans )

	return indicesMappedToEventsStructure		



def find_start_end_intercepts(ID, params, sl, fromTheseIndices=np.array([])) :

	# print("params['EVENTS_FROM_THIS_HIERARCHY']): ", params['EVENTS_FROM_THIS_HIERARCHY'])
	# print_keys_hierarchy(g.dat[ID]['Events'], "g.dat[ID]['Events']")
	if not len(fromTheseIndices) > 0:
		fromTheseIndices = get_indices_that_contains_all_specified_hierarchies( dict(g.dat[ID]['Events']), list(params['EVENTS_FROM_THIS_HIERARCHY']), sl, [] )
	# print("fromTheseIndices: ", fromTheseIndices)
	# quit()
	startIndices = find_events_on_slope( ID, params, sl, onIncline=True, fromTheseIndices=fromTheseIndices ).astype(int)
	# print("startIndices: ", startIndices.shape)
	isolMeth = copy.deepcopy(params["ISOLATION_METHOD"])
	if isolMeth == "MIN_WIDTH":
		params["ISOLATION_METHOD"] = "LAST_IN_GROUP"

	elif isolMeth == "MAX_WIDTH":
		params["ISOLATION_METHOD"] = "FIRST_IN_GROUP"


	startIndices = find_single_event_in_group_of_neighbours( ID, params, sl, fromTheseIndices=startIndices )
	nStartIndices = startIndices.shape[0]

	if isolMeth == "MIN_WIDTH":
		params["ISOLATION_METHOD"] = "FIRST_IN_GROUP"
	elif isolMeth == "MAX_WIDTH":
		params["ISOLATION_METHOD"] = "LAST_IN_GROUP"

	endIndices = find_events_on_slope( ID, params, sl, onIncline=False, fromTheseIndices=fromTheseIndices ).astype(int)

	endIndices = find_single_event_in_group_of_neighbours( ID, params, sl, fromTheseIndices=endIndices ) 	
	nEndIndices = endIndices.shape[0]

	parsStr = 'slopeLen' + str(params["LENGTH_OF_SLOPE"]) + "vinOverlap" + str(params["WINDOW_OVERLAP"])

	newOutputEventHier = params["OUTPUT_EVENTS_HIERARCHY"] + [parsStr]
	startEventHier = copy.deepcopy(newOutputEventHier) + ["StartIndex"]
	endEventHier = copy.deepcopy(newOutputEventHier) + ["EndIndex"]
	outputHier = [[startEventHier, endEventHier]]
	# print("outputHier: ", outputHier)
	# quit()
	# print("startIndices: ", startIndices)


	return startIndices, endIndices, outputHier



# def find_mid_point_of_startEndPairs(ID, params, sl, fromTheseStartIndices=np.array([]), fromTheseEndIndices=np.array([])) :
# 	startTime = time.time()
# 	if len(fromTheseStartIndices) == 0 :
# 		if 'START_EVENTS_HIERARCHY' in params.keys():
# 			fromTheseStartIndices = get_indices_that_contains_all_specified_hierarchies( dict(g.dat[ID]['Events']), list( params['START_EVENTS_HIERARCHY']), sl )

# 	if len(fromTheseEndIndices) == 0 :
# 		if 'END_EVENTS_HIERARCHY' in params.keys():
# 			fromTheseEndIndices = get_indices_that_contains_all_specified_hierarchies( dict(g.dat[ID]['Events']), list( params['END_EVENTS_HIERARCHY']), sl )
	
# 	midEvents = np.array([])
# 	maxDist = params['MAX_WAVE_WIDTH']
# 	minDist = params['MIN_WAVE_WIDTH']

# 	for startIndex in fromTheseStartIndices :
# 		possibleEndRange = (startIndex+minDist, startIndex+maxDist)
# 		endIndicesGreaterThan = fromTheseEndIndices[ np.where( fromTheseEndIndices > possibleEndRange[0] ) ]
# 		endIndicesLessThan = endIndicesGreaterThan[ np.where( endIndicesGreaterThan < possibleEndRange[1] ) ]

# 		if endIndicesLessThan.shape[0] > 0:
# 			midEvents = np.append(midEvents, (round((np.max(endIndicesLessThan)-startIndex)/2) + startIndex))
# 	# print("midEvents: ", midEvents)
# 	time_taken(startTime, inspect.stack()[0][3])

# 	return midEvents



def find_start_end_pairs( ID, params, sl, fromTheseStartIndices=np.array([]), fromTheseEndIndices=np.array([]) ) : 

	startTime = time.time()
	if len(fromTheseStartIndices) == 0 :
		if 'START_EVENTS_HIERARCHY' in params.keys():
			fromTheseStartIndices = get_indices_that_contains_all_specified_hierarchies( dict(g.dat[ID]['Events']), list( params['START_EVENTS_HIERARCHY']), sl )

	if len(fromTheseEndIndices) == 0 :
		if 'END_EVENTS_HIERARCHY' in params.keys():
			fromTheseEndIndices = get_indices_that_contains_all_specified_hierarchies( dict(g.dat[ID]['Events']), list( params['END_EVENTS_HIERARCHY']), sl )
	
	pairedStartEvents = np.array([])	
	pairedEndEvents = np.array([])
	midIndices = np.array([])

	maxWidth = params['MAX_WAVE_WIDTH']
	minWidth = params['MIN_WAVE_WIDTH']
	samps = g.dat[ID]['Samples'][sl]

	for startIndex in fromTheseStartIndices :
		attempt = True
		if pairedEndEvents.shape[0] > 0:
			if startIndex <= pairedEndEvents[-1]:
				attempt=False

		if attempt :

			possibleEndRange = (startIndex+minWidth, startIndex+maxWidth)
			lessThanIndices = np.where( fromTheseEndIndices < possibleEndRange[1] )
			endIndicesLessThan = fromTheseEndIndices[  lessThanIndices ]
			endIndicesGreaterThan = endIndicesLessThan[ np.where( endIndicesLessThan >= possibleEndRange[0] ) ].astype(int)


			startEventVal = samps[int(startIndex)]

			while True :
				if endIndicesGreaterThan.shape[0] > 0 :

					if params["ISOLATION_METHOD"] == "PEAK" :

						possibleEndIndex = int(np.max( endIndicesGreaterThan ))
						rangeVals = np.arange( startIndex, possibleEndIndex ).astype(int)
						midIndex = rangeVals[int(np.argmax( samps[ rangeVals ] ))]
						idealEndIndex = int((midIndex - startIndex) * 2 + startIndex)
						possibleEndIndex = endIndicesGreaterThan[np.argmin( np.abs( endIndicesGreaterThan-idealEndIndex ) )]
					
					else:
						if params["ISOLATION_METHOD"] == 'MAX_WIDTH' :

							possibleEndIndex = np.max(endIndicesGreaterThan)

						elif params["ISOLATION_METHOD"] == 'MIN_WIDTH' :

							possibleEndIndex = np.min(endIndicesGreaterThan)

						midIndex = int( ( possibleEndIndex-startIndex ) / 2 + startIndex)

					if params["MID_POINT_NEED_BE_GREATER_THAN"]:
						midEventVal = samps[ midIndex ]
						endEventVal = samps[ int( possibleEndIndex ) ]		

						if midEventVal > endEventVal and midEventVal > startEventVal :

							pairedStartEvents = np.append( pairedStartEvents, int(startIndex) )
							pairedEndEvents = np.append( pairedEndEvents, int(possibleEndIndex) )
							fromTheseEndIndices = np.delete( fromTheseEndIndices, lessThanIndices )
							break

						else :
							endIndicesGreaterThan = np.setdiff1d( endIndicesGreaterThan, possibleEndIndex )
					else:
						pairedStartEvents = np.append( pairedStartEvents, int(startIndex) )
						pairedEndEvents = np.append( pairedEndEvents, int(possibleEndIndex) )
						fromTheseEndIndices = np.delete( fromTheseEndIndices, lessThanIndices )		
						break				

					# print("possibleEndIndex: ", possibleEndIndex, " endIndicesGreaterThan: ", endIndicesGreaterThan)

				else:
					break
				

	parsStr = 'wavWidth' + str(params["MIN_WAVE_WIDTH"]) + "-" + str(params["MAX_WAVE_WIDTH"]) + str(caps_under_to_cap_lower(params["ISOLATION_METHOD"]))

	newOutputEventHier = params["OUTPUT_EVENTS_HIERARCHY"] + [parsStr]
	startEventHier = copy.deepcopy(newOutputEventHier) + ["StartIndex"]
	endEventHier = copy.deepcopy(newOutputEventHier) + ["EndIndex"]
	outputHier = [[startEventHier, endEventHier]]
	# print("outputHier: ", outputHier)
	# quit()
	# print("startIndices: ", startIndices)
	time_taken(startTime, inspect.stack()[0][3])

	return pairedStartEvents, pairedEndEvents, outputHier
	# print("midEvents: ", midEvents)




def find_mid_waves_param_search( ID, params, sl, nChans ) : 
	startTime = time.time()
	outEventsDict = stack_list_as_hierarchical_dict( params["OUTPUT_EVENTS_HIERARCHY"] )

	print("In find_mid_waves_param_search!")
	paramsList = build_mid_wave_param_variations(params)

	referenceIndices = get_indices_that_contains_all_specified_hierarchies( g.dat[ID]['Events'], params['REFERENCE_EVENTS_HIERARCHIES'], sl, explicit=False )

	print("referenceIndices: ", referenceIndices)
	indicesDict = {}

	for pars in paramsList :
		# print("pars: ", pars, " sl: ", sl)

		correctIndices = get_init_channel_array(nChans)
		detectedIndices = get_init_channel_array(nChans)

		# indicesDict[ paramsSummaryStrs ] = {}

		# indicesDict[ paramsSummaryStrs ][ 'CorrectIndex' ] = get_init_channel_array(nChans)
		# indicesDict[ paramsSummaryStrs ][ 'AllIndex' ] = get_init_channel_array(nChans)

		for chanNum in range(0,nChans) :
			slInSl = np.s_[chanNum, ::]
			refIndices = map_npwhere_to_event_structured_indices_by_row(referenceIndices, chanNum)

			indicesOfThresholdIntercepts, thresholdHier = find_threshold_intercepts(ID, pars, slInSl, nChans)

			if len( indicesOfThresholdIntercepts ) > 0 :
				startIndices, endIndices = find_start_end_intercepts( ID, pars, slInSl, indicesOfThresholdIntercepts)
				# print("startIndices.shape: ", startIndices.shape, " endIndices.shape: ", endIndices.shape)
				midIndices = find_mid_point_of_startEndPairs( ID, pars, slInSl, startIndices, endIndices )
				detectedIndices = insert_1d_input_arr_into_2d_arr(chanNum, detectedIndices, midIndices)

				if len(midIndices) > 1:
					midIndices = find_single_event_in_group_of_neighbours( ID, pars, slInSl, midIndices )


			if len( midIndices ) > 0:
				nearestIndices, refMatches = find_nearest( ID, pars, slInSl, fromTheseIndices=midIndices, nearestTheseIndices=refIndices ) 
				correctIndices = insert_1d_input_arr_into_2d_arr(chanNum, correctIndices, nearestIndices)

			gc.collect()
		nCorrect = get_num_non_x_items(correctIndices, -1)
		nIncorrect = get_num_non_x_items(detectedIndices, -1) - nCorrect
		perCorrect = round( (nCorrect / len(referenceIndices[0]) * 100), 2)
		ratioInToCor = round( (nIncorrect / nCorrect), 2)
		paramsSummaryStrs = "out" + str( pars['OUTLIER_PERCENTILE'] ) + "ref" + str(pars["REFRACTORY_PERIOD"]) + "rngWdth" + str(pars["MIN_WAVE_WIDTH"]) + "-" + str(pars["MAX_WAVE_WIDTH"]) + "corPer" + str(perCorrect) + "rat" + str(ratioInToCor)
		indicesDict[ paramsSummaryStrs ] = {}
		indicesDict[ paramsSummaryStrs ][ 'CorrectIndex' ] = correctIndices
		indicesDict[ paramsSummaryStrs ][ 'AllIndex' ] = detectedIndices
		print("paramsSummaryStrs: ", paramsSummaryStrs)


	outEventsDict = set_childmost_value_from_hierarchical_dict( outEventsDict, indicesDict )
	print_keys_hierarchy(outEventsDict, "outEventsDict")
	print("referenceIndices: ", referenceIndices)
	outReferenceIndices = map_npwhere_to_event_structured_indices(referenceIndices, nChans)
	print("outReferenceIndices: ", outReferenceIndices)

	time_taken(startTime, inspect.stack()[0][3])

	return outEventsDict, outReferenceIndices



def match_singular_events( ID, params, sl, nChans, shouldMatchEvents ) :

	startTime = time.time()
	outEventsDict = stack_list_as_hierarchical_dict( params["OUTPUT_EVENTS_HIERARCHY"] )

	if shouldMatchEvents:
		params['EVENTS_UNDER_SCRUTINY'] = copy.deepcopy( params["OUTPUT_EVENTS_HIERARCHY"] )

	# referenceIndices = get_indices_that_contains_all_specified_hierarchies( g.dat[ID]['Events'], params['REFERENCE_EVENTS_HIERARCHIES'], sl, explicit=False )

	# scrutinyIndices = get_indices_that_contains_all_specified_hierarchies( g.dat[ID]['Events'], params['EVENTS_UNDER_SCRUTINY'], sl, explicit=False )
	scrutinyIndices = get_vals_from_dict_with_this_hierarchy( g.dat[ ID ][ 'Events' ], params[ 'EVENTS_UNDER_SCRUTINY' ] )	

	refHiers = copy.deepcopy(get_full_hierarchies(g.dat[ ID ][ 'Events' ], params[ 'REFERENCE_EVENTS_HIERARCHIES' ]))
	# print( "refHiers: ", refHiers )

	for refHier in refHiers :
		if not 'referenceIndices' in locals() :
			referenceIndices = get_vals_from_dict_with_this_hierarchy( g.dat[ ID ][ 'Events' ], refHier, explicit=False )

		else:
			referenceIndices = np.hstack(( referenceIndices, get_vals_from_dict_with_this_hierarchy( g.dat[ ID ][ 'Events' ], refHier, explicit=False )) )

	# print("params['EVENTS_UNDER_SCRUTINY']: ", params['EVENTS_UNDER_SCRUTINY'], " scrutinyIndices: ", scrutinyIndices.shape)
	# print("params['REFERENCE_EVENTS_HIERARCHIES']: ", params['REFERENCE_EVENTS_HIERARCHIES'], " referenceIndices: ", referenceIndices.shape)

	# quit()

	# print("params: ", params)

	indicesDict = {}

	correctIndices = get_init_channel_array(nChans)
	detectedIndices = get_init_channel_array(nChans)
	missedIndices = get_init_channel_array(nChans)

	for chanNum in range(0,nChans) :

		slInSl = np.s_[chanNum, ::]
		detectedIndices = insert_1d_input_arr_into_2d_arr( chanNum, detectedIndices, scrutinyIndices[ chanNum, : ] )

		if len( scrutinyIndices[ chanNum, : ] ) > 0:
			nearestIndices, refMatches = find_nearest( ID, params, slInSl, fromTheseIndices=scrutinyIndices[ chanNum, : ], nearestTheseIndices=get_non_x_items(referenceIndices[ chanNum, :] )) 
			correctIndices = insert_1d_input_arr_into_2d_arr( chanNum, correctIndices, np.unique(nearestIndices) )

		thisChanMissedIndices = np.unique(np.setdiff1d( get_non_x_items(referenceIndices[ chanNum, :]), refMatches  ))
		# print("missed.shape: ", thisChanMissedIndices.shape,  " correctIndices.shape: ", refMatches)

		missedIndices = insert_1d_input_arr_into_2d_arr(chanNum, missedIndices, thisChanMissedIndices) 

	indicesDict[ 'CorrectIndex' ] = correctIndices
	indicesDict[ 'MissedIndex' ] = missedIndices
	indicesDict[ 'AllIndex' ] = detectedIndices

	nTPs = get_num_non_x_items( indicesDict[ 'CorrectIndex' ] )
	nGuesses = get_num_non_x_items( indicesDict[ 'AllIndex' ] )
	nFNs = get_num_non_x_items( indicesDict[ 'MissedIndex' ] )

	nFPs = nGuesses - nTPs



	sensitivity = str(round(nTPs / (nTPs + nFNs), 4))
	precision = str(round(nTPs / (nTPs + nFPs), 4))


	outEventsDict = {}
	hierStr = "Sens" + str(sensitivity) + "Prec" + str(precision) + caps_under_to_cap_lower(list_as_str( params[ 'EVENTS_UNDER_SCRUTINY' ], "_" ))

	outEventsDict[ hierStr ] = set_childmost_value_from_hierarchical_dict( outEventsDict, indicesDict )
	print_keys_hierarchy(outEventsDict, "outEventsDict")
	# print("referenceIndices: ", referenceIndices)
	# outReferenceIndices = map_npwhere_to_event_structured_indices(referenceIndices, nChans)
	# print("outReferenceIndices: ", outReferenceIndices)
	time_taken(startTime, inspect.stack()[0][3])

	return outEventsDict, referenceIndices



def find_outside_threshold(ID, params, sl, nChans) : 
	startTime = time.time()
	parsStr = caps_under_to_cap_lower(str(params["SUB_METHOD"]) + list_as_str(params["VALUE"],""))

	# print(" g.dat[ID]['Samples'][sl]: ",  g.dat[ID]['Samples'][sl])
	if params['SUB_METHOD'].upper() == 'OUTLIER_PERCENTILE' :
		threshold = get_mean_outliers_within_upper_percentiles( g.dat[ID]['Samples'][sl], params['VALUE'])

	elif params['SUB_METHOD'].upper() == 'RAW_VALUE' :
		threshold = params["VALUE"]

	if 'DIRECTION' in params.keys() :

		indices = np.array([])
		if 'UPPER' in params["DIRECTION"]:
			indices = np.where( g.dat[ID]['Samples'][sl] > threshold[1] )[0]

		if 'LOWER' in params["DIRECTION"]:
			lowerIndices = np.where( g.dat[ID]['Samples'][sl] < threshold[0] )[0]
			indices = np.append(indices, lowerIndices)

		parsStr	+= list_as_str( params["DIRECTION"], "" )

	else :
		indices = np.where( g.dat[ID]['Samples'][sl] > threshold[1] )[0]

	# if len(indicesAboveThreshold) > 1:
	# 	indicesAboveThreshold = map_npwhere_to_event_structured_indices(indicesAboveThreshold, nChans)
	# print("indicesAboveThreshold: ", indicesAboveThreshold)
	aboveThresholdHier = copy.deepcopy(params['OUTPUT_EVENTS_HIERARCHY']) + [parsStr, 'ThresholdIndex']

	time_taken(startTime, inspect.stack()[0][3])

	return indices, aboveThresholdHier



# def find_mid_of(ID, params, sl) : 

# 	# print_keys_hierarchy(copy.copy(g.dat[ID]['Events']))
# 	startEndWindows = get_start_end_event_windows(copy.copy(g.dat[ID]['Events']), copy.copy(sl), params)
	
# 	sampShape =  g.dat[ID]['Samples'][sl[0]].shape[0]

# 	peakIndices = np.array([])
# 	if len(startEndWindows) > 0:
# 		for win in startEndWindows:
# 			# print("Win: ", win)
# 			# print("samples shape: ", g.dat[ID]['Samples'][sl[0]].shape)
# 			if ( win[1] < sampShape ) and ( (win[1] - win[0]) > params['MIN_WINDOW_SIZE']) :

# 				# print("g.dat[ID]['Samples'][sl[0], win[0]:win[1]]: ", g.dat[ID]['Samples'][sl[0], win[0]:win[1]])
# 				peakIndex = np.argmax(g.dat[ID]['Samples'][sl[0], win[0]:win[1]]) + win[0]
# 				# print("argmax peakIndex: ", peakIndex)
# 				peakIndices = np.append(peakIndices, peakIndex.astype(int))

# 	return peakIndices



def find_single_event_in_group_of_neighbours( ID, params, sl, fromTheseIndices=np.array([]) ) : 

	if not len(fromTheseIndices) > 0:
		# print("Trying to get some indices: ")
		fromTheseIndices = get_indices_that_contains_all_specified_hierarchies( dict(g.dat[ID]['Events']), list(params['EVENTS_FROM_THIS_HIERARCHY']), sl, [] )

	# print("fromTheseIndices: ", fromTheseIndices)
	distNeighbours = np.diff( fromTheseIndices ) 
	# print("distNeighbours: ", distNeighbours)
	windowSize = params['WINDOW_OVERLAP']
	# print("windowsSize: ", windowSize)

	selectedEvents = np.array( [] )

	if len( fromTheseIndices ) > 0 :

		skipUntil = 0
		currDistIndex = 0
		fromTheseIndicesInt = fromTheseIndices.astype(int)

		for originalIndex in fromTheseIndicesInt :

			if skipUntil==0 :
				groupOfNeighbours = np.array([])
				# print("currDistIndex: ", currDistIndex, "originalIndex: ", originalIndex)
				# for last index
				# if currDistIndex==len(fromTheseIndices) :

				groupOfNeighbours = np.append(groupOfNeighbours, originalIndex).astype( int )

				for dist in distNeighbours[ currDistIndex : ] :
					currDistIndex += 1

					# print(" dist: ", dist, " winSize: ", windowSize)
					if ( dist > windowSize ) :
						break

					else :				
						groupOfNeighbours = np.append( groupOfNeighbours, fromTheseIndicesInt[ currDistIndex ] ).astype(int)

				if len( groupOfNeighbours.shape ) > 0 :
					skipUntil = groupOfNeighbours.shape[0] - 1

				sampleValsGroup = g.dat[ID][ 'Samples' ][ sl[0], groupOfNeighbours ]
			
				if "ISOLATION_METHOD" in params.keys() :

					if params["ISOLATION_METHOD"] == "MAX_OF_GROUP" :
						chosenSampleIndex = np.argmax( sampleValsGroup ).astype( int )

					elif params["ISOLATION_METHOD"] == "MIN_OF_GROUP" :
						chosenSampleIndex = np.argmin( sampleValsGroup ).astype( int )						

					elif params["ISOLATION_METHOD"] == "FIRST_IN_GROUP" :
						chosenSampleIndex = 0

					elif params["ISOLATION_METHOD"] == "LAST_IN_GROUP" :
						chosenSampleIndex = -1			

				else:
					chosenSampleIndex = np.argmax( sampleValsGroup ).astype( int )

				if 'ndarray' in str(type(chosenSampleIndex)) :
					chosenSampleIndex = chosenSampleIndex[0]

				# print("chosenSampleIndex: ", chosenSampleIndex, " groupOfNeighbours.shape: ", groupOfNeighbours.shape)

				# print("sampleValsGroup: ", sampleValsGroup, "maxSampleValIndex: ", maxSampleValIndex, " groupOfNeighbours[maxSampleValIndex]: ", groupOfNeighbours[maxSampleValIndex])
				selectedEvents = np.append( selectedEvents, groupOfNeighbours[ chosenSampleIndex ] )
			else:
				skipUntil -= 1
	# print("fromTheseIndicesInt.shape: ", fromTheseIndicesInt.shape)
	# print("maxEvents.shape: ", maxEvents.shape)

	return selectedEvents



def find_events_on_slope(ID, params, sl, onIncline=False, fromTheseIndices=np.array([])) : 

	samplesToAnalyse = g.dat[ID]['Samples'][sl]

	if "LENGTH_OF_SLOPE" in params.keys() :
		lengthOfSlope = params['LENGTH_OF_SLOPE']
	else:
		lengthOfSlope = 20

	if len(fromTheseIndices) == 0 :
		if 'EVENTS_FROM_THIS_HIERARCHY' in params.keys():
			fromTheseIndices = get_indices_that_contains_all_specified_hierarchies( dict(g.dat[ID]['Events']), list(params['EVENTS_FROM_THIS_HIERARCHY']), sl, [] )
		else:
			return np.array([])

	extremeVals = (0, samplesToAnalyse.shape[0] )
	slopeEvents = np.array([])

	fromTheseIndices = fromTheseIndices.astype(int)

	for index in fromTheseIndices :

		startSlopeIndex = int(index - round(lengthOfSlope / 2))
		endSlopeIndex = int(index + round(lengthOfSlope / 2))

		# print("Extreme vals: ", extremeVals)
		# print("startSlopeIndex: ", startSlopeIndex)

		if startSlopeIndex < extremeVals[0] :
			startSlopeIndex = int(extremeVals[0])

		if (endSlopeIndex > extremeVals[1]) :
			endSlopeIndex = int(extremeVals[1])


		rightSlopeMean = np.mean( samplesToAnalyse[ index : endSlopeIndex ] )
		leftSlopeMean = np.mean( samplesToAnalyse[ startSlopeIndex : index ] )

		if onIncline :
			if rightSlopeMean > leftSlopeMean :
				# print(rightSlopeMean, " > ", leftSlopeMean)
				slopeEvents = np.append(slopeEvents, index)

		else:
			if rightSlopeMean < leftSlopeMean :
				# print(leftSlopeMean, " < ", rightSlopeMean)
				slopeEvents = np.append(slopeEvents, index)			

	return slopeEvents.astype(int)	





def find_events_every_x_samples(ID, params, sl, fromTheseIndices=np.array([])) : 

	everyXsamples = params['EVERY_X_SAMPLES']
	nSamples = g.dat[ID]['Samples'][sl].shape
	nRepeats = np.floor_divide(nSamples, everyXsamples)
	indices = np.linspace(0, nSamples, nRepeats)

	return indices	



def find_events_away_from(ID, params, sl, fromTheseIndices=np.array([]), awayFromTheseIndices=np.array([])) : 

	minSampleDist = params["MIN_DIST_FROM_EXCLUSIONARY_EVENT"]
	nChansEitherSide = params["NUM_OF_NEIGHBOURING_CHANS_EITHER_SIDE"]	
	rowSize = get_row_size( nChansEitherSide )

	if len(fromTheseIndices) == 0 :
		fromTheseIndices = get_indices_that_contains_all_specified_hierarchies( dict(g.dat[ID]['Events']), list(params['EVENTS_FROM_THIS_HIERARCHY']), sl, [] )
	# print("fromTheseIndices: ", fromTheseIndices)

	if len(awayFromTheseIndices) == 0 :
		chanSlice = get_chan_slice_with_neighbours( sl[0], nChansEitherSide, rowSize )		
		# print("chanSlice: ", chanSlice)
		awayFromTheseIndices = get_indices_that_contains_all_specified_hierarchies( dict(g.dat[ID]['Events']), list(params['EVENTS_EXCL_THIS_HIERARCHY']), chanSlice, [] ).flatten()

	clearIndices = np.array([])

	if len(fromTheseIndices) > 0 :

		for index in fromTheseIndices :
			# print( "Chanslice: ", chanSlice, " awayFromTheseIndices: ", awayFromTheseIndices )
			nearestIndex = np.min( np.abs( awayFromTheseIndices-index ) )
			# print("nearestIndex: ", nearestIndex)
			if ( nearestIndex > minSampleDist ) :
				clearIndices = np.append( clearIndices, index )

	# print("fromTheseIndices: ", fromTheseIndices)
	# print("awayFromTheseIndices: ", awayFromTheseIndices)
	return clearIndices






