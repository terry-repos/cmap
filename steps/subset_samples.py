import sys
import copy 
import math
import numpy as np

sys.path.append('..')
from global_imports import external_modules
from global_imports import session_vars as g
from global_imports.simplify_vars import *

from pyutils.io_utils import *
from pyutils.time_utils import *
from pyutils.dict_utils import *
from pyutils.np_arr_padding import pad_2d_lobsided
from hierarchies.indexing_events import *


def subset_samples() :

	g.command_history.add('', os.path.basename(__file__),g.inspect.stack()[0][3], locals())

	ID, nSamples, nChans, sampleRate, subsetParams, sl = make_globals_readable()

	if 'RANGES' in subsetParams.keys() :

		if subsetParams['METHODS'] == 'DELETE_TIMESTAMP_RANGE' :
			subset_by_range(subsetParams, ID, nSamples, excl=False, sampleRate=sampleRate)

		elif subsetParams['METHODS'] == 'DELETE_CHANNEL_RANGE' :
			subset_by_range(subsetParams, ID, nChans, excl=False)		


	if 'WINDOWS' in subsetParams.keys() :
		
		winStr = "WnS" + str(subsetParams['WINDOWS'][0]) + "WnE" + str(subsetParams['WINDOWS'][1])

		nChansEitherSide = subsetParams[ 'NUM_OF_NEIGHBOURING_CHANS_EITHER_SIDE' ]
		rowSize = get_row_size( nChansEitherSide )
		chansStr = "nNbChns" + str( math.floor( rowSize / 2 ) )
		
		if chansStr not in subsetParams['OUTPUT_EVENTS_HIERARCHY'] :
			subsetParams['OUTPUT_EVENTS_HIERARCHY'].insert(1, chansStr)

		if winStr not in subsetParams['OUTPUT_EVENTS_HIERARCHY'] :
			subsetParams['OUTPUT_EVENTS_HIERARCHY'].insert(1, winStr)

		print_keys_hierarchy( g.dat[ID]['Windows'], "Windows")
		print_keys_hierarchy( g.dat[ID]['Events'], "Events")

		print( "Windowing: ", winStr, " nChansEitherSide: ", nChansEitherSide )

		# wins dict
		if not 'Windows' in g.dat[ID].keys() :
			g.dat[ID]['Windows'] = {}

		if not 'EvntWnMaps' in g.dat[ID].keys() :
			g.dat[ID]['EvntWnMaps'] = {}			

		# Events Dict
		# initArray = np.full(shape=(get_nChans(), 1), fill_value=-1)

		# eventwinDict = copy.copy( stack_list_as_hierarchical_dict( subsetParams['OUTPUT_EVENTS_HIERARCHY'], {} ) )

		# eDic = copy.copy( stack_list_as_hierarchical_dict( subsetParams['OUTPUT_EVENTS_HIERARCHY'], {} ) )

		# g.dat[ID]['Events'].update( copy.copy(eDic) )
		# g.dat[ID]['Events'] = copy.deepcopy(update_dict_with_a_new_initialised_hierarchy(g.dat[ID]['Events'], subsetParams['OUTPUT_EVENTS_HIERARCHY'], nChans=get_nChans()))

		g.dat[ID]['Windows'] = copy.deepcopy( update_dict_with_a_new_initialised_hierarchy(  g.dat[ID]['Windows'],  subsetParams[ 'OUTPUT_EVENTS_HIERARCHY' ], nChans=get_nChans()) )	

		g.dat[ID]['EvntWnMaps'] = copy.deepcopy( update_dict_with_a_new_initialised_hierarchy(  g.dat[ID]['EvntWnMaps'],  subsetParams[ 'OUTPUT_EVENTS_HIERARCHY' ], nChans=get_nChans()) )	


		# eventwinDict = copy.copy( set_childmost_value_from_hierarchical_dict( eventwinDict, initArray) )

		if subsetParams['METHODS'].upper() in ['WIN_BY_LABELS', 'TRAIN', 'TRAIN_AND_TEST'] :

			print( "======================================" )
			print( "windowing ", subsetParams['EVENTS_FROM_THIS_HIERARCHY'] , " into ", subsetParams['OUTPUT_EVENTS_HIERARCHY'] )
			updatedEventwinDict = subset_into_event_wins( subsetParams, ID, rowSize )

			if updatedEventwinDict :
				g.dat[ ID ][ 'Windows' ] = dict_collections_update( dict( g.dat[ID]['Windows']) , dict(updatedEventwinDict) )	

		print_keys_hierarchy( g.dat[ ID ][ 'Windows' ], "Windows")


		# print("g.dat[ID]['Windows']: ", g.dat[ID]['Windows'])
		# print("g.dat[ID]['Events']: ", g.dat[ID]['Events'])

		# g.dat[ID]['Events'] =  set_indices_within_dict( g.dat[ID]['Events'], subsetParams['OUTPUT_EVENTS_HIERARCHY'], sl[0], eventwinIndices, get_nChans() )		



def subset_into_event_wins(subsetParams, ID, rowSize, fromTheseIndices=np.array([])) :

	g.command_history.add('', os.path.basename(__file__),g.inspect.stack()[0][3], locals())

	windows = subsetParams['WINDOWS']

	nChansEitherSide = subsetParams['NUM_OF_NEIGHBOURING_CHANS_EITHER_SIDE']

	windowsSize = np.abs( windows[1] - windows[0] )

	windowsSizeWithNeighbs = windowsSize * rowSize
		
	print("g.cur['STEP']['PARAMS']['CHAN_RANGE']: ", g.cur['STEP']['PARAMS']['CHAN_RANGE'])

	chanRange = list( g.cur['STEP']['PARAMS']['CHAN_RANGE'] )
	winChanSlice=''

	if 'GET_EVENTS_FROM_SPECIFIC_CHAN_RANGE' in subsetParams.keys() :

		chanRange = subsetParams['GET_EVENTS_FROM_SPECIFIC_CHAN_RANGE']

		if len(chanRange) > 1 :
			chanRange = range( chanRange[0], chanRange[1] )

	allIndices = []

	fillVal = 0

	chanI = 0
	winI = 0

	# print("subsetParams: ", subsetParams)

	for chan in chanRange :

		winEventIndices = np.array([])

		winChanSlice = get_chan_slice_with_neighbours( chan, nChansEitherSide, rowSize )	

		eventsChanSlice = np.s_[chan,::]

		fromTheseIndices = copy.copy(get_indices_that_contains_all_specified_hierarchies( dict(g.dat[ID]['Events']), list( subsetParams['EVENTS_FROM_THIS_HIERARCHY'] ), eventsChanSlice ))

		nWins = len( fromTheseIndices )

		# print_keys_hierarchy(g.dat[ID]['Events'], "Events")

		# print("==windowing for chan ",chan ," and slice " , winChanSlice, " and this event chan slice ", eventsChanSlice, " and this hierarchy ", list_as_str([list(subsetParams['EVENTS_FROM_THIS_HIERARCHY'])]), " with these number of indices: ", nWins)

		for index in fromTheseIndices :

			if not index == -1 :
				startwindows = index + windows[0]
				endwindows = index + windows[1]

				# print("index: ", index, " winChanSlice: ", winChanSlice, " startwindows: ", startwindows, " endwindows: ", endwindows)
				if (startwindows > 0) and ( endwindows < g.dat[ID]['Samples'].shape[1] ) :
					thiswinSize = (endwindows - startwindows)		

					# print("In valid indices, thiswinSize : ", thiswinSize)
					if thiswinSize == windowsSize :
						# newEventIndex = startwindows + thiswinSize*chanI + round(thiswinSize/2)
						# print("winChanSlice: ", winChanSlice, " startwindows: ", startwindows, " endwindows : ", endwindows )

						winData = np.copy( g.dat[ ID ][ 'Samples' ][ winChanSlice, startwindows : endwindows ].flatten() )

						if winData.shape[0] > 0 :
							# # startPos = winI * winData.shape[0] 

							# # newEventIndex = startPos + chanI * thiswinSize + round(thiswinSize / 2)

							# winEventIndices = np.append( winEventIndices, newEventIndex )	

							# winsForChan[startPos:(startPos + winData.shape[0])] = np.copy(winData)
							chanIndexCoords = np.array([chan, index]).astype(int)

							if 'windowAllChans' in locals():
								# print("winData.shape: ", winData.shape, " windowAllChans.shape: ", windowAllChans.shape)

								windowAllChans = np.vstack(( windowAllChans, winData ))
								evntWnMapping = np.vstack(( evntWnMapping, chanIndexCoords ))

							else:
								windowAllChans = np.copy( winData )
								evntWnMapping = np.copy( chanIndexCoords )



					# winI += 1

			# print("windowAllChans : ", windowAllChans)
			# chanI += 1


	if 'windowAllChans' in locals():
		nWins = windowAllChans.shape[0]
		# print_keys_hierarchy(g.dat[ID]['Events'], "EVENTS")

		# g.dat[ID]['Events'] =  set_indices_within_dict( g.dat[ID]['Events'], subsetParams['OUTPUT_EVENTS_HIERARCHY'], chan, winEventIndices, get_nChans() )
		g.dat[ID]['EvntWnMaps'] =  set_vals_in_dict( g.dat[ID]['EvntWnMaps'], subsetParams['OUTPUT_EVENTS_HIERARCHY'], evntWnMapping, nWins )
		print("** ", nWins, " wins created for ", subsetParams['OUTPUT_EVENTS_HIERARCHY'], " Additionally, an event-win map has been created of size ", evntWnMapping.shape," event-wins coords")

		# print_keys_hierarchy(g.dat[ID]['EvntWnMaps'], "EvntWnMaps")


	# print("windowAllChans: ", windowAllChans)
	# print("g.dat[ID]['Windows'] : ", g.dat[ID]['Windows'] )
	# print("list(subsetParams['EVENTS_FROM_THIS_HIERARCHY']): ", list(subsetParams['EVENTS_FROM_THIS_HIERARCHY']))
	# print("subsetParams['OUTPUT_EVENTS_HIERARCHY']: ", subsetParams['OUTPUT_EVENTS_HIERARCHY'])

		return set_vals_in_dict( g.dat[ID]['Windows'], subsetParams['OUTPUT_EVENTS_HIERARCHY'], windowAllChans, nWins )

	else :
		return None


	# allIndices.append(fromTheseIndices)
	# print("len(allIndices): ", len(allIndices))

	# return windowAllChans	



def subset_by_range(subsetParams, ID, maxVal, excl=False, sampleRate=None) :

	g.command_history.add('', os.path.basename(__file__),g.inspect.stack()[0][3], locals())
	
	extremes = (0, maxVal)
	rois = subsetParams['RANGES']

	rangesFormatted = []
	for roi in rois :

		for roiI in [0, 1] :
			if roi[ roiI ] == None :
				roi[ roiI ] = extremes[ roiI ]

		if sampleRate : # indicates subsetting of samples
			# print("SampleRate: ", sampleRate)
			rangesFormatted.append( get_indices_range_of_data( roi, extremes, sampleRate ) )
			axisToDelete = 1

		else: # indicates reduction of channels
			rangesFormatted.append(roi)
			axisToDelete = 0

	# if excl==False: #inclusion involves deleting start and end indices (extremes) 
		# First delete end indices
	print("PRE DEL g.dat[ID]['Samples'].shape ", g.dat[ID]['Samples'].shape)

	delete_indices(ID, rangesFormatted, axisToDelete, subsetParams)

	# print("New channel order: ", g.dat[ID]['ChanConfig']['CHANNEL_ORDER'])
	print("POST g.dat[ID]['Samples'].shape ", g.dat[ID]['Samples'].shape)


		# print_keys_hierarchy(g.dat[ID]['Events'] )
	# else: # exclusion is straight forward deletion.

	# 	delete_indices(ID, rangeFormatted, axisToDelete, subsetParams)
	# 	if 'ALSO_SUBSET_EVENTS' in subsetParams.keys() :
	# 		if subsetParams['ALSO_SUBSET_EVENTS'] :
	# 			g.dat[ID]['Events'] = dict(keep_event_indices( dict(g.dat[ID]['Events']), [extremes[0], rangeFormatted[0]]))
	# 			g.dat[ID]['Events'] = dict(keep_event_indices( dict(g.dat[ID]['Events']), [rangeFormatted[1], extremes[1]]))

		# print_keys_hierarchy(g.dat[ID]['Events'] )
	# print( "AFTER g.dat[ID]['Events'].size: ", sys.getsizeof(g.dat[ID]['Events']) )



def delete_indices(ID, rangesToDel, axisToDel=1, subsetPars=None) :

	g.command_history.add('', os.path.basename(__file__),g.inspect.stack()[0][3], locals())
	sliceIndices = np.array([])

	sortedRanges = sorted(rangesToDel, reverse=True)
	print("sortedRanges: ", sortedRanges)
	for rangeToDel in sortedRanges :
		sliceIndices = np.s_[ rangeToDel[0]:rangeToDel[1] ]
	# print_keys_hierarchy(g.dat[ID]['Events'])

		print( "sliceIndices: ", sliceIndices )

	# print("g.dat[ID]['Samples'].shape: ", g.dat[ID]['Samples'].shape)
		if not 'ONLY_SUBSET_EVENTS' in subsetPars.keys() :
			if axisToDel==1 :

				g.dat[ ID ][ 'Samples' ] = np.delete( g.dat[ID]['Samples'], sliceIndices, axis=1)
				g.dat[ ID ]['Timestamps' ] = np.delete( g.dat[ID]['Timestamps'], sliceIndices, axis=0)
			
			else :
				g.dat[ID]['Samples'] = np.delete(g.dat[ID]['Samples'], sliceIndices, axis=0)
				if 'CHANNEL' in subsetPars['METHODS'].upper() :
					rangeOfChannelsDeleted = range(rangeToDel[0], rangeToDel[1])
					oldChannelOrder = copy.deepcopy( g.dat[ID]['ChanConfig']['CHANNEL_ORDER'] )
					print("Old channel order: ", g.dat[ID]['ChanConfig']['CHANNEL_ORDER'])
					newChannelOrder = remove_this_list_from_that_list( rangeOfChannelsDeleted, oldChannelOrder )
					g.dat[ ID ][ 'ChanConfig' ][ 'CHANNEL_ORDER' ] = copy.deepcopy( newChannelOrder )
					print("New channel order: ", g.dat[ID]['ChanConfig']['CHANNEL_ORDER'])			

		if 'ALSO_SUBSET_EVENTS' in subsetPars.keys() :
			if subsetPars['ALSO_SUBSET_EVENTS'] :
				if 'Events' in g.dat[ID].keys() :
					print("also subsetting events!")
					print_keys_hierarchy(g.dat[ID]['Events'], " events.")

					# print("PRIOR g.dat[ID]['Events']: ", g.dat[ID]['Events'])
					g.dat[ID]['Events'] = dict( delete_event_indices( dict(g.dat[ID]['Events'] ), rangeToDel, axisToDel) )
					print_keys_hierarchy(g.dat[ID]['Events'], " events.")




	print("g.dat[ID]['Samples'].shape: ", g.dat[ID]['Samples'].shape)




