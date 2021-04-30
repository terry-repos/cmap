import sys
import numpy as np
import itertools

from pyutils.dict_utils import *
from pyutils.string_utils import *
from pyutils.time_utils import *

from pyutils.list_utils import *
from hierarchies.indexing_events import *


# def group_propagating_events( samplesDict, pars, candidateIndices, nChans ) :

# 	groupIndicesDic = {}
# 	groupPropDict = {}

# 	if 'DIRECTION' in pars.keys() :
# 		directionalitySought = pars[ 'DIRECTION' ].lower()
# 		directionalitySought = directionalitySought[0].upper() + directionalitySought[1:]
# 	else :
# 		directionalitySought = "BiDirectional"

# 	groupIndicesParsStr = 'maxGap' + str(pars["MAX_DELAY"]) + "nghbSpan" + str(pars["NEIGHBOUR_SPAN"]) + "minChansSub" + str(pars["MIN_CHANS_TO_FORM_SUB_GROUP"])

# 	groupIndicesDic[ groupIndicesParsStr ] = get_group_indices( directionalitySought, pars, candidateIndices, nChans )
# 	# print_keys_hierarchy( groupIndicesDic, "groupIndicesDic" )

# 	groupPropDict[ groupIndicesParsStr ] = get_grouped_event_properties( samplesDict, groupIndicesDic[groupIndicesParsStr] )

# 	eventStructCompatibleDic = make_event_struct_compatible( groupIndicesDic, groupIndicesParsStr, nChans)
# 	# print_keys_hierarchy( groupPropDict, "groupPropDict" )

# 	return eventStructCompatibleDic, groupPropDict

def group_propagating_events( samplesDict, pars, candidateIndices, nChans ) :

	groupIndicesDic = {}
	groupPropDict = {}

	if 'DIRECTION' in pars.keys() :
		directionalitySought = pars[ 'DIRECTION' ].lower()
		directionalitySought = directionalitySought[0].upper() + directionalitySought[1:]
	else :
		directionalitySought = "Agnostic"


	groupIndicesDic = get_group_indices( directionalitySought, pars, candidateIndices, nChans )
	# print_keys_hierarchy( groupIndicesDic, "groupIndicesDic" )

	# groupPropDict[ groupIndicesParsStr ] = get_grouped_event_properties( samplesDict, groupIndicesDic[groupIndicesParsStr] )

	# eventStructCompatibleDic = make_event_struct_compatible( groupIndicesDic, groupIndicesParsStr, nChans)
	# print_keys_hierarchy( groupPropDict, "groupPropDict" )

	return groupIndicesDic


def make_event_struct_compatible( eventsDic, groupIndicesParsStr, nChans ) :

	eventStructCompatDic = {}
	eventStructCompatDic[ groupIndicesParsStr ] = {}

	for direction, indicesTypeDict in eventsDic[groupIndicesParsStr].items() :
		eventStructCompatDic[ groupIndicesParsStr ][ direction ] = {}

		for indicesType, indices in indicesTypeDict.items() :
			# print_keys_hierarchy(eventsDic, "eventsDic")
			# print("groupIndicesParsStr: ", groupIndicesParsStr, " direction: ", direction, " indicesType: ", indicesType)
			indicesToConvert = eventsDic[ groupIndicesParsStr ][ direction ][ indicesType ]
			flattenedIndices = flatten_list_of_list(indicesToConvert)
			eventStructCompatDic[ groupIndicesParsStr ][ direction ][ indicesType ] = transform_event_coords_to_channel_array( flattenedIndices, nChans )

	return eventStructCompatDic



# def get_group_indices( directionalitySought, pars, candidateIndices, nChans ) :

# 	groupIndices = {}
# 	possibleGroups = search_indices_for_groups( directionalitySought, pars, candidateIndices, nChans )
# 	# print("possibleGroups: ", possibleGroups)
# 	# print("group_indices locals(): ", locals())
# 	groupIndices = copy.deepcopy( link_sub_groups( possibleGroups, pars ) )

# 	# print("directionalitySought: ", directionalitySought)

# 	if 'INTERPOLATE' in pars.keys():
# 		if pars[ 'INTERPOLATE' ]:
# 			groupIndices = copy.deepcopy( interpolate_events_within_group( groupIndices, pars ) )

# 	if 'GET_START_END_INDICES' in pars.keys():
# 		if pars[ 'GET_START_END_INDICES' ]:
# 			groupIndices = copy.deepcopy( get_start_end_indices( groupIndices ) )

# 	return groupIndices

def get_group_indices( directionalitySought, pars, candidateIndices, nChans ) :

	groupIndices = {}


	if pars["SUB_METHOD"].upper() == "NEAREST" :
		groupIndicesParsStr = 'maxGap' + str(pars[ "MAX_DELAY_TO_NEIGHBOUR" ]) + "nghbSpan" + str(pars["MAX_NEIGHBOUR_SPAN"]) + "minChans" + str(pars["MIN_INDICES_TO_FORM_GROUP"])
		groupIndices[groupIndicesParsStr] = {}
		groupIndices[groupIndicesParsStr]['Agnostic'] = {}
		groupIndices[groupIndicesParsStr]['Agnostic']['GroupIndex'] = search_indices_for_groups( directionalitySought, pars, candidateIndices, nChans )

	elif pars["SUB_METHOD"].upper() == "GET_DIRECTION" :
		groupIndices = divy_groups_into_directions( candidateIndices, pars, nChans )

	elif pars["SUB_METHOD"].upper() == "INTERPOLATE" :
		groupIndices = interpolate( candidateIndices, pars, nChans )		


	# # print("possibleGroups: ", possibleGroups)
	# # print("group_indices locals(): ", locals())
	# groupIndices = copy.deepcopy( link_sub_groups( possibleGroups, pars ) )

	# print("directionalitySought: ", directionalitySought)

	# if 'INTERPOLATE' in pars.keys():
	# 	if pars[ 'INTERPOLATE' ]:
	# 		groupIndices = copy.deepcopy( interpolate_events_within_group( groupIndices, pars ) )

	# if 'GET_START_END_INDICES' in pars.keys():
	# 	if pars[ 'GET_START_END_INDICES' ]:
	# 		groupIndices = copy.deepcopy( get_start_end_indices( groupIndices ) )

	return groupIndices



# def search_indices_for_groups( directionalitySoughtSpecified, pars, candidateIndices, nChans ):
# 	startTime = time.time()

# 	possibleGroups = {}

# 	if directionalitySoughtSpecified=='BiDirectional':
# 		possibleGroups['Antegrade'] = []
# 		possibleGroups['Retrograde'] = []

# 	else:
# 		possibleGroups[ directionalitySoughtSpecified ] = []

# 	nextRow = 0

# 	rowIndices = range( nextRow, nChans )

# 	for row in rowIndices :

# 		rowVals = copy.deepcopy(candidateIndices[1][np.where(candidateIndices[0]==row)])
# 		nextRow += 1
# 		lastNeighbourRow = row + pars["NEIGHBOUR_SPAN"]

# 		for col in range(0,len(rowVals)) :

# 			possibleStartGroupIndex = rowVals[ col ]


# 			currentGroup = [ [row, possibleStartGroupIndex] ]

# 			directionalitySought = copy.deepcopy(directionalitySoughtSpecified)
# 			# print("directinoalitysought in check: ", directionalitySought)

# 			for adjacentRow in range( nextRow, lastNeighbourRow ) :
# 				if adjacentRow in rowIndices :
# 					foundMatchingAntegradeNeighbour = False
# 					foundMatchingRetrogradeNeighbour = False
# 					foundMatch = False

# 					adjacentRowVals = candidateIndices[ 1 ][ np.where( candidateIndices[0]==adjacentRow ) ]
# 					distMultplr = abs(adjacentRow - row) + 1
# 					delays = adjacentRowVals - possibleStartGroupIndex
# 					# print("delays: ", delays)

# 					if directionalitySought=="BiDirectional" or directionalitySought=="Antegrade" :
# 						antegradeIndices = np.where(( delays > pars["MIN_DELAY"]*distMultplr ) &  ( delays < pars["MAX_DELAY"]*distMultplr))
# 						if len( antegradeIndices[ 0 ] ) > 0 :
# 							foundMatchingAntegradeNeighbour = True


# 					if directionalitySought=="BiDirectional" or directionalitySought=="Retrograde" :					
# 						retrogradeIndices = np.where(( delays < pars["MIN_DELAY"]*(distMultplr*-1) ) &  ( delays > pars["MAX_DELAY"]*(distMultplr*-1)))
# 						if len( retrogradeIndices[ 0 ] ) > 0 :
# 							foundMatchingRetrogradeNeighbour = True

# 					if foundMatchingAntegradeNeighbour or foundMatchingRetrogradeNeighbour :

# 						if directionalitySought == "BiDirectional" :
# 							if foundMatchingAntegradeNeighbour :
# 								minAntegradeDelayIndex = np.min( np.abs( antegradeIndices[0] ) )

# 							else:
# 								minAntegradeDelayIndex = 9999

# 							if foundMatchingRetrogradeNeighbour :
# 								minRetrogradeDelayIndex = np.min( np.abs( retrogradeIndices[0] ) )
# 								# print("minRetrogradeDelayIndex: ", minRetrogradeDelayIndex)
# 							else :
# 								minRetrogradeDelayIndex = 9999								

# 							if minAntegradeDelayIndex < minRetrogradeDelayIndex :
# 								directionalitySought = "Antegrade"

# 							else:
# 								directionalitySought = "Retrograde"		

# 						# print("minAnt: ", minAntegradeDelayIndex, " minRetro: ", minRetrogradeDelayIndex, " directionalitySought: ", directionalitySought)								

# 						if directionalitySought == "Antegrade" and foundMatchingAntegradeNeighbour:
# 							foundMatch = True
# 							matchingEventIndex = np.min( adjacentRowVals[ antegradeIndices ] ) # Assumptions. Max of one index per channel can belong to a group.
# 							# print("Adding ANTE-grade index: ", matchingEventIndex)

# 						elif directionalitySought == "Retrograde" and foundMatchingRetrogradeNeighbour:
# 							foundMatch = True
# 							matchingEventIndex = np.min( adjacentRowVals[ retrogradeIndices ] ) # Assumptions. Max of one index per channel can belong to a group.
# 							# print("Adding RETRO-grade index: ", matchingEventIndex)

# 						# print("minAnt: ", minAntegradeDelayIndex, " minRetro: ", minRetrogradeDelayIndex, " directionalitySought: ", directionalitySought)								

# 						# print("distanceOfIndices: ", distanceOfIndices, " distanceMultiplier: ", distMultplr)

# 						if foundMatch :
# 							matchingEventRowAndColIndex = [ adjacentRow, int(matchingEventIndex) ]
# 								# print("matchingEventRowAndColIndex: ", matchingEventRowAndColIndex, " currentGroup: ", currentGroup)

# 							if matchingEventRowAndColIndex not in currentGroup :
# 								currentGroup.append( matchingEventRowAndColIndex )

# 			if len( currentGroup ) > pars["MIN_CHANS_TO_FORM_SUB_GROUP"] :

# 				if directionalitySought=="Antegrade" :
# 					reverseVal = False

# 				else:
# 					reverseVal = True

# 				sortedList = sorted( currentGroup, reverse=reverseVal )
# 				# print(directionalitySought, " sortedList: ", sortedList)
# 				possibleGroups[ directionalitySought ].append( sortedList )

# 	time_taken(startTime, inspect.stack()[0][3], shouldRun=True)

# 	return possibleGroups

def order_indices_by_max_distance_to_adjacent_neighbs( indices ) :

	nextNeighbDist = np.array( np.diff( indices ))
	previousNeighbDist = np.copy( nextNeighbDist )

	previousNeighbDist = np.insert( previousNeighbDist, 0, 99999 )
	nextNeighbDist = np.append( nextNeighbDist, 99999 )
	distsStacked = np.stack(( previousNeighbDist, nextNeighbDist )).T

	minDistArr = np.array([])

	for r in range(distsStacked.shape[0]) :
		minDist = int(np.min(distsStacked[r,:]))
		minDistArr = np.append(minDistArr, minDist)

	# print( "minDistArr: ", minDistArr )

	argSort = np.argsort( minDistArr.astype(int) )[::-1]

	return indices[ argSort ]



def get_nearest_valid_neighbour(  adjacentRowIndices, lastUpIndexAdded, lastDownIndexAdded, upsRowsTraversed, downsRowsTraversed, currRowDist, adjRow, priorDirection, params ):

	goForth = True

	if len(adjacentRowIndices) > 0 :
		if ( adjRow == -1  ) :
			distToNeighbs = adjacentRowIndices - lastUpIndexAdded

		elif ( adjRow == 1 ) :
			distToNeighbs = adjacentRowIndices - lastDownIndexAdded			

		nearestNeighbIndex = np.argmin( np.abs( distToNeighbs ) )
		nearestNeighbVal = adjacentRowIndices[ nearestNeighbIndex ]
		nearestNeighbDelay = distToNeighbs[ nearestNeighbIndex ] 

		goForth = True

		if params["MAX_DELAY"] :
			if np.abs( nearestNeighbDelay ) > (params["MAX_DELAY"]*currRowDist) :
				goForth = False

		if params["MIN_DELAY"] :
			if np.abs( nearestNeighbDelay ) < (params["MIN_DELAY"]*currRowDist) :
				goForth = False

		if np.abs( nearestNeighbDelay ) > params["MIN_DELAY_TO_CONCLUDE_DIRECTIONALITY"] :
			currDirection = get_direction( nearestNeighbDelay, adjRow )

		else:
			currDirection = ""

		if (len( currDirection ) > 0) and (len( priorDirection ) > 0):
			# If specified, check the point is going in the sme direction
			if params["CONSISTENT_DIRECTION"] :
				if not currDirection == priorDirection :
					goForth = False

			if params["DIRECTION_CHECK"] :
				if not currDirection == priorDirection :
					goForth = False		

		priorDirection = currDirection

	else:
		goForth = False

	if goForth:

		if adjRow == -1 :
			upsRowsTraversed = 0
			lastUpIndexAdded = nearestNeighbVal

		else :
			downsRowsTraversed = 0
			lastDownIndexAdded = nearestNeighbVal

	else :

		if adjRow < 0 :
			upsRowsTraversed += 1

		else :
			downsRowsTraversed += 1

		nearestNeighbVal = -1

	return nearestNeighbVal, lastUpIndexAdded, lastDownIndexAdded, upsRowsTraversed, downsRowsTraversed, priorDirection


def get_row_counters( row, upRowI, downRowI, adjRow, nChans ) :
	skip = False

	if ( adjRow == -1 ) :
		upRowI = upRowI + adjRow
		currRow = upRowI
		if upRowI < 0:
			skip = True		

	elif ( adjRow == 1 ) :
		downRowI = downRowI + adjRow
		currRow = downRowI		
		if downRowI >= nChans:
			skip = True			

	currRowDist = np.abs(row-currRow)

	return currRow, currRowDist, upRowI, downRowI, skip




def search_indices_for_groups( directionalitySoughtSpecified, pars, candidateIndices, nChans ) :

	startTime = time.time()
	elementaryGroups = {}


	nextRow = 0

	print("n reference indices: ", len(candidateIndices[0]))

	# rowIndices = order_row_indices_by_asc_order_of_non_x_items( candIndices[0] )
	rowIndicesInOrderOfnumOfIndices = Counter( candidateIndices[0] ).most_common( ) 
	print("rowIndicesInAscOrderOfnumOfIndices: ", rowIndicesInOrderOfnumOfIndices)
	rowIndicesInAscOrderOfnumOfIndicesAsc = sorted(rowIndicesInOrderOfnumOfIndices, key=lambda x: x[1])
	print("rowIndicesInAscOrderOfnumOfIndicesAsc: ", rowIndicesInAscOrderOfnumOfIndicesAsc)

	nGroupMult = 7

	maxNgroups = rowIndicesInAscOrderOfnumOfIndicesAsc[-1][1] * nGroupMult
	allocatedIndices = get_init_channel_array( nChans )

	params = {}

	maxNofRowsToSkip = pars[ "MAX_NEIGHBOUR_SPAN" ]
	groupingMethod = pars[ "SUB_METHOD" ]
	params["MIN_DELAY_TO_CONCLUDE_DIRECTIONALITY"] = pars["MIN_DELAY_TO_CONCLUDE_DIRECTIONALITY"]

	if "MIN_DELAY_TO_NEIGHBOUR" in pars.keys():
		params["MIN_DELAY"] = pars["MIN_DELAY_TO_NEIGHBOUR"]
	else:
		params["MIN_DELAY"]  = False

	if "MAX_DELAY_TO_NEIGHBOUR" in pars.keys() :
		params["MAX_DELAY"]  = pars[ "MAX_DELAY_TO_NEIGHBOUR" ]
	else:
		params["MAX_DELAY"] = False		

	if "CONSISTENT_DIRECTION" in pars.keys() :
		params["CONSISTENT_DIRECTION"] = pars[ "CONSISTENT_DIRECTION" ]
	else:
		params["CONSISTENT_DIRECTION"] = False	

	if "DIRECTION" in pars.keys() :
		params["DIRECTION_CHECK"] = pars[ "DIRECTION" ]
	else:
		params["DIRECTION_CHECK"] = False			

	groupsArr = np.full( shape=(nChans,maxNgroups), fill_value=-1 )
	traversedIndices = np.copy( groupsArr )

	groupI = 0
	traversedGroupI = 0

	# for each channel in asc order of number of indices
	for row in rowIndicesInAscOrderOfnumOfIndicesAsc :

		row = row[0]
		# Get the chan indices
		origRowIndices = copy.deepcopy( candidateIndices[1][ np.where( candidateIndices[0]==row ) ] )

		# If indices already allocated, remove
		rowIndices = np.setdiff1d( origRowIndices, traversedIndices[ row, :]  )
		if len(rowIndices) > 0:
			print("row: ", row , "groupI: ", groupI, " n: ", get_num_non_x_items(groupsArr, -1), "origRowIndices.shape: ", origRowIndices.shape, " rowIndices.shape: ", rowIndices.shape )

			orderedRowIndices = order_indices_by_max_distance_to_adjacent_neighbs( rowIndices )

			for possibleFoundingIndex in orderedRowIndices :

				traversedIndices = insert_index_into_arr( traversedIndices, row, possibleFoundingIndex, nChans ) 

				# print("row: ", row, " groupI: " , str(groupI), "index: ", possibleFoundingIndex, " n: ", get_num_non_x_items(groupsArr, -1))

				# Add possible founding index to groups
				groupsArr[ row, groupI ] = possibleFoundingIndex

				# initialise vals
				lastUpIndexAdded = possibleFoundingIndex
				lastDownIndexAdded = possibleFoundingIndex

				priorDirection = ""

				upsRowsTraversed = 0
				downsRowsTraversed = 0

				upRowI = row
				downRowI = row

				# search in alternating up and down directions for nearby indices and add to group if match criteria
				while ( downsRowsTraversed <= maxNofRowsToSkip ) and ( upsRowsTraversed <= maxNofRowsToSkip ) and ((upRowI >= 0) or ( downRowI < nChans )):

					# for up and down directions
					for adjRow in [-1, 1] :

						currRow, currRowDist, upRowI, downRowI, skip = get_row_counters( row, upRowI, downRowI, adjRow, nChans )

						if not skip :

							# Get possible group neighbours
							adjacentRowIndices = candidateIndices[ 1 ][ np.where( candidateIndices[0]==currRow ) ]
							adjacentRowIndices = np.setdiff1d( adjacentRowIndices, traversedIndices[ currRow, :]  )

							# print("currRow: ", "adjacentRowIndices.shape: ", adjacentRowIndices.shape)
						
							nearestNeighbVal, lastUpIndexAdded, lastDownIndexAdded, upsRowsTraversed, downsRowsTraversed, priorDirection = get_nearest_valid_neighbour( adjacentRowIndices, lastUpIndexAdded, lastDownIndexAdded, upsRowsTraversed, downsRowsTraversed, currRowDist, adjRow, priorDirection, params )
							groupsArr[ currRow, groupI ] = nearestNeighbVal

							traversedIndices = insert_index_into_arr( traversedIndices, currRow, nearestNeighbVal, nChans ) 



				nIndicesInGroup = get_num_non_x_items( groupsArr[:,groupI] )

				if nIndicesInGroup > pars["MIN_INDICES_TO_FORM_GROUP"] :
					groupI += 1

				else :
					groupsArr[:, groupI] = -1

	return groupsArr


def divy_groups_into_directions(inGroupIndices, pars, nChans):

	nGroups = inGroupIndices.shape[1]
	groupIs = np.arange(nGroups)

	antI = 0
	retI = 0

	directedIndices = {}
	emptArr = np.full( shape=(nChans, nGroups) , fill_value=-1 )

	directedIndices[ 'Antegrade' ] = {} 
	directedIndices[ 'Antegrade' ][ 'GroupIndex' ] = emptArr
	directedIndices[ 'Retrograde' ] = {} 
	directedIndices[ 'Retrograde' ][ 'GroupIndex' ] = emptArr

	for groupI in groupIs :
		# print( "inGroupIndices[:, groupI]: ", inGroupIndices[:, groupI] )
		groupIndices = get_non_x_items( inGroupIndices[:, groupI] )
		# print( groupIndices )

		if len(groupIndices) > 0:
			if groupIndices[0] < groupIndices[-1] :
				directedIndices[ 'Antegrade' ][ 'GroupIndex' ][ :, antI ] = np.copy(inGroupIndices[ :, groupI ])
				antI += 1

			else :
				directedIndices[ 'Retrograde' ][ 'GroupIndex' ][ :, retI ] = np.copy(inGroupIndices[ :, groupI ])
				retI += 1

	return directedIndices


def divy_groups_into_directions(inGroupIndices, pars, nChans):

	nGroups = inGroupIndices.shape[1]
	groupIs = np.arange(nGroups)

	antI = 0
	retI = 0

	directedIndices = {}
	emptArr = np.full( shape=(nChans, nGroups) , fill_value=-1 )

	directedIndices[ 'Antegrade' ] = {} 
	directedIndices[ 'Antegrade' ][ 'GroupIndex' ] = emptArr
	directedIndices[ 'Retrograde' ] = {} 
	directedIndices[ 'Retrograde' ][ 'GroupIndex' ] = emptArr

	for groupI in groupIs :
		# print( "inGroupIndices[:, groupI]: ", inGroupIndices[:, groupI] )
		groupIndices = get_non_x_items( inGroupIndices[:, groupI] )
		# print( groupIndices )

		if len(groupIndices) > 0:
			if groupIndices[0] < groupIndices[-1] :
				directedIndices[ 'Antegrade' ][ 'GroupIndex' ][ :, antI ] = np.copy(inGroupIndices[ :, groupI ])
				antI += 1

			else :
				directedIndices[ 'Retrograde' ][ 'GroupIndex' ][ :, retI ] = np.copy(inGroupIndices[ :, groupI ])
				retI += 1

	return directedIndices	
				

			# for col in range(0, len(rowVals)) :

			# 	possibleStartGroupIndex = rowIndices[ col ]
			# 	currentGroup = [ [row, possibleStartGroupIndex] ]
			# 	directionalitySought = copy.deepcopy(directionalitySoughtSpecified)





			# 		for adjacentRow in range( nextRow, lastNeighbourRow ) :
			# 			if adjacentRow in rowIndices :
			# 				foundMatchingAntegradeNeighbour = False
			# 				foundMatchingRetrogradeNeighbour = False
			# 				foundMatch = False

			# 				distMultplr = abs(adjacentRow - row) + 1
			# 				delays = adjacentRowVals - possibleStartGroupIndex
			# 				# print("delays: ", delays)

			# 				if directionalitySought=="BiDirectional" or directionalitySought=="Antegrade" :
			# 					antegradeIndices = np.where(( delays > pars["MIN_DELAY"]*distMultplr ) &  ( delays < pars["MAX_DELAY"]*distMultplr))
			# 					if len( antegradeIndices[ 0 ] ) > 0 :
			# 						foundMatchingAntegradeNeighbour = True


			# 				if directionalitySought=="BiDirectional" or directionalitySought=="Retrograde" :					
			# 					retrogradeIndices = np.where(( delays < pars["MIN_DELAY"]*(distMultplr*-1) ) &  ( delays > pars["MAX_DELAY"]*(distMultplr*-1)))
			# 					if len( retrogradeIndices[ 0 ] ) > 0 :
			# 						foundMatchingRetrogradeNeighbour = True

			# 				if foundMatchingAntegradeNeighbour or foundMatchingRetrogradeNeighbour :

			# 					if directionalitySought == "BiDirectional" :
			# 						if foundMatchingAntegradeNeighbour :
			# 							minAntegradeDelayIndex = np.min( np.abs( antegradeIndices[0] ) )

			# 						else:
			# 							minAntegradeDelayIndex = 9999

			# 						if foundMatchingRetrogradeNeighbour :
			# 							minRetrogradeDelayIndex = np.min( np.abs( retrogradeIndices[0] ) )
			# 							# print("minRetrogradeDelayIndex: ", minRetrogradeDelayIndex)
			# 						else :
			# 							minRetrogradeDelayIndex = 9999								

			# 						if minAntegradeDelayIndex < minRetrogradeDelayIndex :
			# 							directionalitySought = "Antegrade"

			# 						else:
			# 							directionalitySought = "Retrograde"		

			# 					# print("minAnt: ", minAntegradeDelayIndex, " minRetro: ", minRetrogradeDelayIndex, " directionalitySought: ", directionalitySought)								

			# 					if directionalitySought == "Antegrade" and foundMatchingAntegradeNeighbour:
			# 						foundMatch = True
			# 						matchingEventIndex = np.min( adjacentRowVals[ antegradeIndices ] ) # Assumptions. Max of one index per channel can belong to a group.
			# 						# print("Adding ANTE-grade index: ", matchingEventIndex)

			# 					elif directionalitySought == "Retrograde" and foundMatchingRetrogradeNeighbour:
			# 						foundMatch = True
			# 						matchingEventIndex = np.min( adjacentRowVals[ retrogradeIndices ] ) # Assumptions. Max of one index per channel can belong to a group.
			# 						# print("Adding RETRO-grade index: ", matchingEventIndex)

			# 					# print("minAnt: ", minAntegradeDelayIndex, " minRetro: ", minRetrogradeDelayIndex, " directionalitySought: ", directionalitySought)								

			# 					# print("distanceOfIndices: ", distanceOfIndices, " distanceMultiplier: ", distMultplr)

			# 					if foundMatch :
			# 						matchingEventRowAndColIndex = [ adjacentRow, int(matchingEventIndex) ]
			# 							# print("matchingEventRowAndColIndex: ", matchingEventRowAndColIndex, " currentGroup: ", currentGroup)

			# 						if matchingEventRowAndColIndex not in currentGroup :
			# 							currentGroup.append( matchingEventRowAndColIndex )

			# 		if len( currentGroup ) > pars["MIN_CHANS_TO_FORM_SUB_GROUP"] :

			# 			if directionalitySought=="Antegrade" :
			# 				reverseVal = False

			# 			else:
			# 				reverseVal = True

			# 			sortedList = sorted( currentGroup, reverse=reverseVal )
			# 			# print(directionalitySought, " sortedList: ", sortedList)
			# 			possibleGroups[ directionalitySought ].append( sortedList )

	time_taken(startTime, inspect.stack()[0][3], shouldRun=True)

	return possibleGroups	



def link_sub_groups(  subGroups,  pars, nGroups=0 ) :
	startTime = time.time()

	minGroupSize = pars["MIN_CHANS_TO_FORM_GROUP"]
	# print("link_sub_groups locals(): ", locals())
	subGroupIndex = 0
	uniqueGroups = {}
	groups = {}

	# print("linking sub groups of direction ", subGroups.keys(), " totalling: ")

	for direction, subGrps in subGroups.items():
		print("total sub groups for ", direction, " n is ", len(subGrps))

		groups[direction] = []
		uniqueGroups[ direction ] = {}
		uniqueGroups[ direction ]['GroupIndex'] = []
		for thisSubGroup in subGrps :
			subGroupIndex += 1

			# print("analysing subgroup: ", subGroupIndex)

			foundMatchingSubGroup = False
			uniqueItems = []

			for thatSubGroup in subGrps[ subGroupIndex:] :
				if should_we_link_these_lists(thisSubGroup, thatSubGroup, direction):
					thisSubGroup = copy.deepcopy( link_these_lists(thisSubGroup, thatSubGroup) )
					# print("Linking sub group index subGroupIndex: ", subGroupIndex)


			mergedWithExistingGroup = False

			for i in range( len( groups[direction] ) ) :
				if should_we_link_these_lists( thisSubGroup, groups[direction][i] ) :
					groups[ direction ][i] = copy.deepcopy( link_these_lists(thisSubGroup, groups[direction][i],  direction) )
					print("Linking main index subGroupIndex: ", subGroupIndex)
					mergedWithExistingGroup = True

			if not mergedWithExistingGroup :
				if len(thisSubGroup) > minGroupSize :
					groups[ direction ].append( copy.deepcopy( thisSubGroup ))

		uniqueGroups[ direction ][ 'GroupIndex' ] = copy.deepcopy( get_unique_items_in_list( groups[direction] ) )
		print( "len(uniqueGroups): ", len(uniqueGroups) )
	# print(uniqueGroups)
	# quit()
	# 	nUniqueGroups = len( uniqueGroups )
	# # if not nUniqueGroups == nGroups:
	# # 	uniqueGroups = link_sub_groups(uniqueGroups, minGroupSize, nGroups=nUniqueGroup

	return uniqueGroups



def get_grouped_event_properties( allSamples, indicesDict ) :

	startTime = time.time()

	groupEventPropDic = {}

	for direction in indicesDict.keys() :
		originalIndices = indicesDict[ direction][ 'GroupIndex' ]

		if 'InterpIndex' in indicesDict[ direction ].keys() :
			interpIndices = indicesDict[ direction ][ 'InterpIndex' ]
			interp = True

		else:
			interp = False
			interpIndices = np.array([])

		groupDict = {}
		groupI = 0

		for groupIndexPairs, interpIndexPairs in zip(originalIndices, interpIndices) :

			groupDict[ groupI ] = {}

			# if direction == "ANTEGRADE":
			# 	reverseVal = False				

			# else :
			# 	reverseVal = True	

			timeStart = groupIndexPairs[ 0 ][ 1 ]
			timeEnd =  groupIndexPairs[ -1 ][ 1 ]
			channelStart = groupIndexPairs[ 0 ][ 0 ]
			channelEnd = groupIndexPairs[ -1 ][ 0 ]

			# print("indices before sorting: ", groupedIndices)
			# print("indices after sorting: ", indices)
			groupDict[ groupI ][ 'Indices' ] = copy.deepcopy( groupIndexPairs )

			if interp:
				groupDict[ groupI ][ 'IntrpIndices' ] = copy.deepcopy( interpIndexPairs )

			duration = timeEnd - timeStart
			distanceInChannel =  channelEnd - channelStart
			speed = distanceInChannel / duration

			groupDict[ groupI ][ 'StartIndex' ] = timeStart
			groupDict[ groupI ][ 'EndIndex' ] = timeEnd
			groupDict[ groupI ][ 'Duration' ] = duration
			groupDict[ groupI ][ 'Distance' ] = distanceInChannel
			groupDict[ groupI ][ 'Speed' ] = np.round(speed, 2)
			groupDict[ groupI ][ 'Direction' ] = direction

			indicesValues = np.array([])

			for indexPair in groupIndexPairs :
				indicesValues = np.append( indicesValues, allSamples[ indexPair[0], indexPair[1] ] )

			maxIndex = groupIndexPairs[ np.argmax( indicesValues ) ]
			maxValue = np.max( indicesValues )

			minIndex = groupIndexPairs[ np.argmin( indicesValues ) ]
			minValue = np.min( indicesValues )

			groupDict[ groupI ][ 'Values' ] = indicesValues	
			groupDict[ groupI ][ 'MaxIndex' ] = maxIndex	
			groupDict[ groupI ][ 'MaxValue' ] = np.round(maxValue, 2)
			groupDict[ groupI ][ 'MinIndex' ] = minIndex
			groupDict[ groupI ][ 'MinValue' ] = np.round(minValue, 2)

			groupI += 1

		groupEventPropDic[direction] = copy.deepcopy( groupDict )

	return groupEventPropDic


def get_start_end_indices( groupedIndicesDict ) :

	for direction, groupIndices in groupedIndicesDict.items() :
		startIndices = []
		endIndices = []

		for indices in groupIndices[ 'GroupIndex' ] :
			startIndices.append( [indices[0]] )
			endIndices.append( [indices[-1]] )

		groupedIndicesDict[ direction ][ 'StartIndex' ] = startIndices
		groupedIndicesDict[ direction ][ 'EndIndex' ] = endIndices

	return groupedIndicesDict



def get_direction(delay, goingUp) :
	delay = delay*goingUp
	if delay < 0 :
		return "RETROGRADE"
	elif delay > 0 :
		return "ANTEGRADE"
	else :
		return "SYNCHED"



def check_lists_have_same_directionality( firstList, secondList ):

	firstListGroupDirection = get_group_direction(firstList)
	secondListGroupDirection = get_group_direction(secondList)

	if firstListGroupDirection==secondListGroupDirection:
		return True
	else:
		return False


def getItem(item):
	return item[1]


def get_group_direction( inList ) :

	sortedListByTime = sorted( inList, key=getItem)

	listStart = sortedListByTime[0][1]
	listEnd = sortedListByTime[-1][1]

	if (listEnd - listStart) > 0 :
		return "Antegrade"

	else:
		return "Retrograde"



def should_we_link_these_lists(firstList, secondList, checkDirection=False) :
	# startTime = time.time()

	if one_of_this_list_in_that_list( firstList, secondList ) :
		if checkDirection:
			if check_lists_have_same_directionality(firstList, secondList) :
				return True

		else:
			return True

	# time_taken(startTime, inspect.stack()[0][3], shouldRun=False)			

	return False



def link_these_lists( firstList, secondList, specifiedDirection=False ) :
	# startTime = time.time()

	combinedLists = copy.deepcopy(firstList) + copy.deepcopy(secondList)

	uniqueItems = get_unique_items_in_list( combinedLists )

	if not specifiedDirection:
		groupDirection = get_group_direction(uniqueItems)
	else:
		groupDirection = specifiedDirection

	if groupDirection=="Retrograde":
		reverseVal = True
	else:
		reverseVal = False


	sortedList = sorted(uniqueItems, reverse=reverseVal)
	# time_taken(startTime, inspect.stack()[0][3], shouldRun=True)			

	return sortedList



def interpolate_events_within_group( groupIndicesDict, pars ) :

	for direction, groupIndices in groupIndicesDict.items() :
		interpedGroups = []
		allIndicesGroups = []
		# nIndices = len( indices )
		# print("Direction: ", direction, " groupIndices: ", groupIndices )

		for indices in groupIndices['GroupIndex']:
			interpedGroup = []
			# print("len(indices): ", len(indices), " len(indices)-1: ", (len(indices[:-1])))
			indicesI = 0

			for indexPair in indices[ : -1 ] :
				indicesI += 1
				# print("len(indices): ", len(indices), " indicesI: ", indicesI)

				nextIndexPair = indices[ indicesI ]
			
				timeGap = nextIndexPair[1] - indexPair[1]
				chanGap = abs( nextIndexPair[0] - indexPair[0] )
					

				if chanGap > 1: #Interpolation must happen if changap is greater than one
					increment = round(timeGap / chanGap)
					incrementTotal = indexPair[1]
					chanRange = range( (indexPair[ 0 ]+1), nextIndexPair[0] )

					if direction=='Retrograde' :
						chanRange = range( (indexPair[ 0 ] - 1), (nextIndexPair[0]), -1)
					
					# print("chanRange: ", chanRange)
					for chan in chanRange :
							# print("chan: ", chan, "indexPair: ", indexPair, "nextIndexPair: ", nextIndexPair)								
							# print("chanRange: ", chanRange)						
						incrementTotal = int(incrementTotal + increment)
						interpedIndex = [ chan, incrementTotal ]
						interpedGroup.append ( interpedIndex )


			if len( interpedGroup ) > 0 :
				# print("indices: ", indices)
				# print("interpped indices: ", interpedGroup)
				# print("Start coords: ", indices[0], " interped group: ", interpedGroup, " end coord: ", indices[-1]  )

				interpedGroups.append( interpedGroup )
				allIndicesGroup = link_these_lists( indices, interpedGroup, specifiedDirection=direction )
				# print(direction," ", allIndicesGroup)
				allIndicesGroups.append( allIndicesGroup )
		# print("Direction: ",direction, " interpedgroups: ", interpedGroups)
		# print("Direction: ",direction, " allIndicesGroups: ", allIndicesGroups)

		groupIndicesDict[ direction ][ 'InterpIndex' ] = copy.deepcopy(interpedGroups)
		# groupIndicesDict[ direction ][ 'AllIndices' ] = copy.deepcopy(interpedGroups)

	return groupIndicesDict



