import numpy as np
import re
from pyutils.dict_utils import *
# from hierarchies.hierarchies_events import *
from hierarchies.hierarchies import *
from hierarchies.indexing_events import *
from pyutils.np_arr_samples import *


def get_win_range_from_str( inwinStr ) :
	regex = r"S[-]?[0-9]*E[0-9]"

	returnThisItem = True

	returnThisItem = re.search(regex, inwinStr.upper())

	if returnThisItem:
		winStr = inwinStr

	startwinPos = (inwinStr.upper().index("S") + 1)
	endwinPos = (inwinStr.upper().index("E") + 1)

	winVals = [ int(inwinStr[startwinPos:(endwinPos-1)]) , int(inwinStr[endwinPos:])]

	return winVals	


def get_win_range_from_hieranrchy( inHierListo ) :
	regex = r"WNS[-]?[0-9]*WE[0-9]"

	for it in inHierListo :
		returnThisItem = True

		returnThisItem = re.search(regex, it.upper())
		# print("re.search(regex, it): ",returnThisItem)

		if returnThisItem:
			winStr = it
			break
	startwinPos = (winStr.upper().index("WnS") + 1)
	endwinPos = (winStr.upper().index("WnE") + 1)
	# print(winStr)

	winVals = [ int(winStr[startwinPos:(endwinPos-1)]) , int(winStr[endwinPos:])]

	return winVals	


def get_num_neighbs_chans_from_str( neighbChanStr ) :
	regex = r"NNBCHNS"

	retItem = True

	retItem = re.search(regex, itemInList.upper())
	# print("itemInList: ", itemInList, "re.search(regex, itemInList): ",retItem)
	chanInt = 0

	if retItem :
		chanInt = int( itemInList[ 7 : ] )

	return chanInt, retItem



def get_num_neighbs_chans_from_hierarchy( chanInList ) :
	# print("inlist: ", chanInList)
	# print("Regex: ", regex)
	for itemInList in chanInList :
		chanInt, retItem = get_num_neighbs_chans_from_str( itemInList )
		if retItem :
			break

	return chanInt	



def create_hierarchLists_based_on_file_name(eventsFile, evtParams) :
	
	lvlTypes = evtParams['LEVELS_TYPE']
	lvlSpecifyingStrsAll = evtParams['LEVEL_SPECIFYING_STRINGS']

	hierFromFileName = get_event_types(eventsFile.attr['Name'], evtParams)
	formattedHierFromFileName = list_of_strings_from_caps_under_to_cap_lower(hierFromFileName)

	allHierarchies = []
	allHierarchies.append( formattedHierFromFileName )
	
	finalHierarchies = []
	# print("Start creating hierarchies")	

	synoDict = get_synonyms_dict()

	while len(allHierarchies) > 0:

		baseHierarchy = allHierarchies.pop()

		currHierarchy = []

		for lvlType, lvlSpecifyingStrs in zip(lvlTypes, lvlSpecifyingStrsAll) :

			specStrsWithSyns = expand_list_to_include_synonyms( lvlSpecifyingStrs, synoDict )

			lvlSpecifyingStrs = list_of_strings_from_caps_under_to_cap_lower( specStrsWithSyns )
			# print("expanded list with lvlSpecifyingStrs: ", lvlSpecifyingStrs)

			specifier = get_item_in_bothLists(baseHierarchy, lvlSpecifyingStrs)
		
			if specifier == None :

				divergingHierarchies = []
				hierList = []

				for specStr in lvlSpecifyingStrs :
					if check_if_string_in_file( specStr, eventsFile.attr['NameAndPath'] ) :
						divergingHierarchies.append( specStr )
						# there is a hierarchy in the file that diverges from the filename

				for divergeH in divergingHierarchies:
					hierList = []

					hierList = list(currHierarchy) + [ replace_synonyms( divergeH , synoDict ) ]

					if not hierList in allHierarchies :
						allHierarchies.append( hierList )

				if len(hierList) > 0 :
					pass
					# print("allHierarchies: " , allHierarchies)

				# get default if is required.
				elif check_level_is_required(lvlType) :

					specifiers = get_default_specifier(lvlSpecifyingStrs)

					for spec in specifiers:
						hierList = []

						hierList = list(currHierarchy) + [ replace_synonyms( spec , synoDict ) ]

						if not hierList in allHierarchies :
							allHierarchies.append( hierList )					

			else :

				currHierarchy.append( specifier )

		if check_new_hierarchy_matches_required_hierarchy(currHierarchy, lvlTypes, lvlSpecifyingStrsAll, synoDict) :
			finalHierarchies.append( currHierarchy )
	# print("eventsFile: ", eventsFile.attr['Name'])

	return finalHierarchies





def map_input_win_set_to_labelled_events_set( evntWnDic, eventDic, inputwinHierachy, labelEventHierarchy ) :

	inputWinHierachyFull = get_full_hierarchies( evntWnDic,  inputwinHierachy )
	evntWnMaps = get_vals_from_dict_with_this_hierarchy( evntWnDic, inputWinHierachyFull )
	# print("inputWinHierachyFull: ", inputWinHierachyFull)
	# print("EVNT_win_MAPS: ", evntWnMaps)
	# print("evntWnDic: ", evntWnDic)
	# quit()

	# print_keys_hierarchy(evntWnDic, "evntWnDic")
	# # print("eventDic: ", eventDic)

	# print_keys_hierarchy(eventDic, "eventDic")	
	labelEventHierarchiesFull = get_full_hierarchies( eventDic, labelEventHierarchy )

	if list_contains_list(labelEventHierarchiesFull) :
		labelEventHierarchyFullMinimal = get_smallest_list( labelEventHierarchiesFull )
	else:
		labelEventHierarchyFullMinimal = copy.deepcopy(labelEventHierarchiesFull)


	# print( "labelEventHierarchiesFull: ", labelEventHierarchiesFull )
	print( "labelEventHierarchyFullMinimal: ", labelEventHierarchyFullMinimal )
	# quit()

	
	trueEvntIndices = get_vals_from_dict_with_this_hierarchy( eventDic, labelEventHierarchyFullMinimal )
	# print("inputwinHierachy: ", inputwinHierachy)
	# print("labelEventHierarchy: ", labelEventHierarchy)	

	# print("evntWnMaps.shape: ", evntWnMaps.shape)
	# print("trueEvntIndices.shape: ", trueEvntIndices.shape)
	# print("trueEvntIndices: ", trueEvntIndices)

	if ( evntWnMaps.any() ) and ( trueEvntIndices.any() ) :

		nWins = evntWnMaps.shape[0]	

		# print("evntWnMaps.shape: ", evntWnMaps.shape)
		# print("evntWnMaps.shape[0]: ", evntWnMaps.shape[0])

		overlappingwins = np.zeros(shape=(nWins), dtype=int)

		iteratI = 0

		for evntWnMap in evntWnMaps :

			# print( "Is ", trueEvntIndices[ evntWnMap[0], : ], " ==  ", evntWnMap[1] )
			overlappingIndices = np.where( trueEvntIndices[ evntWnMap[0], : ] == evntWnMap[1] )
			# print( "evntWnMap: ", evntWnMap, " overlappingIndices: ", overlappingIndices )

			if len( overlappingIndices[0] ) > 0 :
				overlappingwins[ iteratI ] = 1

			iteratI += 1

	else:
		overlappingwins = None

	return overlappingwins	



def create_new_events_from_event_win_mapping( winsToInclude, evntWnHier, sourceHier, newHier, evntWnMapDic, evntDic, nChans ) :
	# print_keys_hierarchy(evntWnMapDic, "evntWnMapDic")
	# print( "sourceHier: ", sourceHier )
	# print(" newHier: ", newHier)

	inputWinHierachyFull = get_full_hierarchies( evntWnMapDic,  evntWnHier )
	# print( "inputWinHierachyFull: ", inputWinHierachyFull )
	evntWnMaps = get_vals_from_dict_with_this_hierarchy( evntWnMapDic, inputWinHierachyFull )
	# print( "Prior evntWnMaps: ", len(evntWnMaps) )
	# print( "wins to include: ", winsToInclude )
	# totalNwins = len(winsToInclude[0]) + len(winsToExclude[0])

	# print("winsToInclude.shape: ", winsToInclude.shape)
	# print(" len(winsToInclude[0]): ", len(winsToInclude[0]), " evntWnMaps.shape: ", evntWnMaps.shape)
	# print(" wins to include: ", winsToInclude)
	# print(" wins to exclude: ", winsToExclude)


	evntWnMaps = evntWnMaps[ winsToInclude]
	# print("After subset subsetEvEpocMaps: ", len(subsetEvEpocMaps))
	# quit()

	evntDic = transform_event_coord_list_to_channel_array( evntWnMaps, copy.deepcopy(evntDic), newHier, nChans )

	return evntDic


def transform_event_coord_list_to_channel_array( evntWnMaps, evntDic, newEvtHier, nChans ) :

	nWins = evntWnMaps.shape[0]

	evntDic = update_dict_with_a_new_initialised_hierarchy( evntDic, newEvtHier, nChans )
	evntChanArray = get_init_channel_array( nChans )

	for epocRow in range(0, nWins) :

		samplesChan = evntWnMaps[ epocRow ][0]
		samplesIndex = evntWnMaps[ epocRow ][1]

		# print( "samplesChan: ", samplesChan, " samplesIndex: ", samplesIndex )

		evntChanArray = insert_index_into_arr( evntChanArray, samplesChan, samplesIndex, nChans )

	evntDic = set_vals_in_dict( evntDic, newEvtHier, evntChanArray )

	return evntDic






