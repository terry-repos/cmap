import os
from global_imports import session_vars as g

import copy

from pyutils.io_utils import *
from pyutils.dict_utils import *

from hierarchies.hierarchies import *


def get_default_specifier(specList) :

	for spec in specList:
		if "*" in spec :
			return caps_under_to_cap_lower(spec)
	return spec







# def create_hierarchLists_based_on_file_name(eventsFile, evtParams) :
	
# 	lvlTypes = evtParams['LEVELS_TYPE']
# 	lvlSpecifyingStrsAll = evtParams['LEVEL_SPECIFYING_STRINGS']

# 	hierFromFileName = get_event_types(eventsFile.attr['Name'], evtParams)
# 	formattedHierFromFileName = list_of_strings_from_caps_under_to_cap_lower(hierFromFileName)

# 	allHierarchies = []
# 	allHierarchies.append( formattedHierFromFileName )
	
# 	finalHierarchies = []
# 	print("allHierarchies: ", allHierarchies)
# 	print("Start creating hierarchies: ")	

# 	synoDict = get_synonyms_dict()

# 	while len(allHierarchies) > 0 :

# 		if len(baseHierarchy) > 1 :
# 			if isinstance(baseHierarchy, list) :
# 				baseHierarchy = allHierarchies.pop()

# 		else:
# 			baseHierarchy = copy.deepcopy(allHierarchies)

# 		currHierarchy = []

# 		for lvlType, lvlSpecifyingStrs in zip(lvlTypes, lvlSpecifyingStrsAll) :

# 			specStrsWithSyns = expand_list_to_include_synonyms( lvlSpecifyingStrs, synoDict )

# 			lvlSpecifyingStrs = list_of_strings_from_caps_under_to_cap_lower( specStrsWithSyns )

# 			specifier = get_item_in_bothLists(baseHierarchy, lvlSpecifyingStrs)

		
# 			if specifier == None :

# 				divergingHierarchies = []
# 				hierList = []

# 				for specStr in lvlSpecifyingStrs :
# 					if check_if_string_in_file( specStr, eventsFile.attr['NameAndPath'] ) :
# 						divergingHierarchies.append( specStr )
# 						# there is a hierarchy in the file that diverges from the filename

# 				for divergeH in divergingHierarchies:
# 					hierList = []

# 					hierList = list(currHierarchy) + [ replace_synonyms( divergeH , synoDict ) ]
# 					print("hierlist: ", hierList, " baseHierarchy: ", baseHierarchy, " expanded list with lvlSpecifyingStrs: ", lvlSpecifyingStrs, " specifier: ", specifier)

# 					if not hierList in allHierarchies :
# 						allHierarchies.append( hierList )

# 				if len(hierList) > 0 :
# 					pass
# 					# print("allHierarchies: " , allHierarchies)

# 				# get default if is required.
# 				elif check_level_is_required(lvlType) :

# 					specifiers = get_default_specifier(lvlSpecifyingStrs)

# 					for spec in specifiers:
# 						hierList = []

# 						hierList = list(currHierarchy) + [ replace_synonyms( spec , synoDict ) ]

# 						if not hierList in allHierarchies :
# 							allHierarchies.append( hierList )					

# 			else :

# 				currHierarchy.append( specifier )

# 		if check_new_hierarchy_matches_required_hierarchy(currHierarchy, lvlTypes, lvlSpecifyingStrsAll, synoDict) :
# 			finalHierarchies.append( currHierarchy )
# 	# print("eventsFile: ", eventsFile.attr['Name'])

# 	return finalHierarchies



def create_hierarchLists_based_on_file_name_contents__and_event_params(eventsFile, evtParams) :
	
	lvlTypes = evtParams['LEVELS_TYPE']
	lvlSpecifyingStrsAll = evtParams['LEVEL_SPECIFYING_STRINGS']

	hierFromFileName = get_event_types(eventsFile.attr['Name'], evtParams)
	formattedHierFromFileName = list_of_strings_from_caps_under_to_cap_lower(hierFromFileName)

	allHierarchies = []
	allHierarchies.append( formattedHierFromFileName )
	
	finalHierarchies = []
	print("Start creating hierarchies")	

	synoDict = get_synonyms_dict()
	print("synoDict: ", synoDict)
	print("allHierarchies: ", allHierarchies)

	while len( allHierarchies ) > 0 :

		baseHierarchy = allHierarchies.pop()
		print(" baseHierarchy: ", baseHierarchy)
		currHierarchy = []
		iterator = 0

		for lvlType, lvlSpecifyingStrs in zip(lvlTypes, lvlSpecifyingStrsAll) :

			iterator += 1
			specStrsWithSyns = expand_list_to_include_synonyms( lvlSpecifyingStrs, synoDict )

			lvlSpecifyingStrs = list_of_strings_from_caps_under_to_cap_lower( specStrsWithSyns )

			specifier = get_item_in_bothLists(baseHierarchy, lvlSpecifyingStrs)
			print( str(iterator), " expanded list with lvlSpecifyingStrs: ", lvlSpecifyingStrs, " baseHierarchy: ", baseHierarchy, " specifier: ", specifier)
		
			if specifier == None :

				divergingHierarchies = []
				hierList = []

				for specStr in lvlSpecifyingStrs :
					# print("eventsFile.attr['NameAndPath']: ", eventsFile.attr['NameAndPath'], " specStr: ", specStr)
					if check_if_string_in_file( specStr, eventsFile.attr['NameAndPath'] ) :
						# print("Match! we have a diverging hierarchy.")
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

		if check_new_hierarchy_matches_required_hierarchy( currHierarchy, lvlTypes, lvlSpecifyingStrsAll, synoDict) :
			finalHierarchies.append( currHierarchy )
	# print("eventsFile: ", eventsFile.attr['Name'])

	return finalHierarchies




def create_event_hierarchy_from_params_and_input_file(eventsFile, evtParams , nChans) :

	lvls =  evtParams['LEVELS_TYPE'] 
	lvlsStrs =  evtParams['LEVEL_SPECIFYING_STRINGS'] 
	eventTypesFromFileName = get_event_types(eventsFile.attr['Name'], evtParams)

	eventHierarchies = create_hierarchLists_based_on_file_name_contents__and_event_params(eventsFile, evtParams)
	print("eventsFile: ", eventsFile.attr['Name'])

	print("eventHierarchies orig: ", eventHierarchies)

	eventHierarchies = subset_to_exclude_file_name_hierarchy(eventHierarchies, eventTypesFromFileName)
	print("eventHierarchies without filename hierarch: ", eventHierarchies)

	eventHierarchies = subset_only_maximal_hierarchies(eventHierarchies)
	print("eventHierarchies with only maximal filename hierarchs: ", eventHierarchies)

	evtDic = {}

	for eventHierarchy in eventHierarchies :
		tmpDic = stack_list_as_hierarchical_dict(eventHierarchy, {})
		tmpDic = copy.deepcopy( set_nones_from_hierarchical_dict_to(tmpDic, get_init_channel_array(nChans) ) )

		evtDic = merge_dicts(evtDic, tmpDic)

	print("Initial event hierarchy: ")
	print_keys_hierarchy(evtDic)

	return evtDic
			


def get_default_units(evParams) :
	return list_of_strings_from_caps_under_to_cap_lower(evParams["LEVEL_SPECIFYING_STRINGS"][-1])






def event_xml_processing(inFile, xmlData, eventParams) :

	g.command_history.add('',os.path.basename(__file__),g.inspect.stack()[0][3], locals())
	xmlRoot = xmlData.getroot()
	print(xmlRoot.tag, " : ", xmlRoot.attrib)

	nChans = None

	if 'ChanConfig' in g.dat[g.cur['DAT_ID']].keys():
		if 'NUM_CHANS' in g.dat[g.cur['DAT_ID']]['ChanConfig'].keys() :
			if isinstance(g.dat[g.cur['DAT_ID']]['ChanConfig']['NUM_CHANS'], int) :
				nChans = copy.copy(g.dat[g.cur['DAT_ID']]['ChanConfig']['NUM_CHANS'])

	fileSpecifiedEventsHierarchy = get_event_types(inFile.attr['Name'], eventParams)


	newEvtsDic = create_event_hierarchy_from_params_and_input_file( inFile, eventParams, nChans )

	print("	print_keys_hierarchy(newEvtsDic)")
	print_keys_hierarchy(newEvtsDic)
	
	if 'parentKys' in locals() or 'parentKys' in globals():
		del parentKys

	parentKys, unitKys = get_child_and_parent_keys(newEvtsDic, list(), list())

	print("fileSpecifiedEventsHierarchy: ",fileSpecifiedEventsHierarchy)
	print("parentKeys: ", parentKys)
	print("unitKys: ", unitKys)

	synonymsDict = get_synonyms_dict()


	print_keys_hierarchy(newEvtsDic)
	hierArch = []

	for branch in xmlData.iter():

		# print(branch.tag, " : ", branch.attrib)
		thisChan = -1
		newEvsDicToAdd = {}


		for possKey, possVal in branch.attrib.items() :

			if isinstance(possVal, str) :
				possVal = possVal[0].upper() + possVal[1:]

				possValFormatted = replace_synonyms( possVal, synonymsDict )

				if possValFormatted in parentKys :
					if possValFormatted not in hierArch:
						# print("len(hierArch): ", len(hierArch), " len(fileSpecifiedEventsHierarchy): ", len(fileSpecifiedEventsHierarchy))
						hierArch = [possValFormatted]


			if possKey.upper() == 'CHANNEL':

				thisChan = int(possVal)

				if eventParams['SUBTRACT_ONE_FROM_CHAN_NUM']:
					thisChan += 1

				if eventParams['INVERT_CHANS']:
					thisChan = nChans - thisChan

			possKey = possKey[0].upper() + possKey[1:]

			if isinstance(possKey, str) :
				possKeyFormatted = replace_synonyms(possKey, synonymsDict)
				# print( "possKey: ", possKey, "possKeyFormatted:  ", possKeyFormatted)

			if possKeyFormatted in parentKys :

				if possKeyFormatted not in hierArch :

					# if len(hierArch) == 0 :		
					# 	# print("len(hierArch): ", len(hierArch), " len(fileSpecifiedEventsHierarchy): ", len(fileSpecifiedEventsHierarchy))
					# 	hierArch.append(possKeyFormatted)	
					# else:
					hierArch = [possKeyFormatted]



			elif possKeyFormatted in unitKys :
				newEvsDicToAdd[possKeyFormatted] = int(possVal)
			
			# print("possVal: ", possVal, "hierArch: ", hierArch, " possKeyFormatted: ", possKeyFormatted, " unitKys: ", unitKys, " parentKys: ", parentKys)


		# print("hierArch:", hierArch)

		if thisChan > -1 :

			if not newEvsDicToAdd == None :
				# print(" hierArch: ",hierArch)
				newEvtsDic = copy.copy(update_hierarchical_dict_with_flat_dict(dict(newEvtsDic), newEvsDicToAdd, hierArch, thisChan, nChans))
	
	eventHierarchies = copy.deepcopy( get_hierarchies_from_dict( newEvtsDic ) )
	print(eventHierarchies)
	completeHierarchies = []
	for evHier in eventHierarchies:
		tempNewHier = copy.deepcopy(fileSpecifiedEventsHierarchy)
		print("tempNewHier: ", tempNewHier, " evHier: ", evHier)
		tempNewHier+=evHier
		completeHierarchies.append( tempNewHier )
	print("completeHierarchies: ", completeHierarchies)

	return newEvtsDic, completeHierarchies



def get_event_params_from_data_params() :
	return g.params['DATA']['EVENTS']



def get_event_types(inFileName, eventParams):

	evFileMatchParams = eventParams['EVENT_TYPE_MATCHING_BY_FILENAME']

	valsBetween = get_sub_str_between_strs( inFileName,  evFileMatchParams['EVENT_TYPE_PREFACE'] ,  evFileMatchParams['EVENT_TYPE_SUFFIX'] )
	print("valsBetween: ", valsBetween)
	eventHierarchy = valsBetween.split( g.params['SEP'] )
	eventHierarchy = remove_empty_strs_from_list(eventHierarchy)
	eventHierarchy = [ caps_under_to_cap_lower(item) for item in eventHierarchy ]

	print("inFileName: ", inFileName, "eventHierarchy: ", eventHierarchy)
	# quit()
	return eventHierarchy



def subset_dict_to_contain_specified_hierarchy( hiDict, hiList, hierListkeysInDict=[], ouDic=[] ) :

	if isinstance(hiDict, dict) :

		for ki, vi in hiDict.items() :

			kysInDict.append( ki )
			print( "hiList: ", hiList, " kysInDict: ", kysInDict )

			if isinstance(vi, dict) :
				ouDic = copy.copy( subset_dict_to_contain_specified_hierarchy( vi, hiList, kysInDict, ouDic ) )

			elif 'ndarray' in str(type(vi)) :
			
				indsOfInds = [np.where(vi > -1)]
				# print("indsOfInds: ", indsOfInds)					
				# if there are valid indices here ...
				if len(indsOfInds) > 0 :
					if len(indsOfInds[0]) > 0 :

						print("kysInDict: ", kysInDict, " hiList: ", hiList)

						missingHierarchs = copy.copy(remove_this_list_from_that_list( kysInDict, hiList ))
						print("missingHierarchs: ", missingHierarchs)

						if len( missingHierarchs ) == 0 :
							ouDic[ki] = vi


			if len( kysInDict ) > 0 :
				kysInDict.pop()

	return ouDic



def get_styles_for_levels_in_hierarchy( evtDataParams, mkrParams,  mkrhierarchiesL ) :

	lvlTypes = evtDataParams['LEVELS_TYPE']
	lvlSpecifyingStrsAll = evtDataParams['LEVEL_SPECIFYING_STRINGS']

	synoDict = get_synonyms_dict()

	# for each level type if there is a match in any of the hierarchies add the 
	levelPosDict = {}	

	for lvlType, lvlSpecifyingStrs in zip(lvlTypes, lvlSpecifyingStrsAll) :

		specStrsWithSyns = expand_list_to_include_synonyms( lvlSpecifyingStrs, synoDict )
		# print("expanded list with specStrsWithSyns: ", specStrsWithSyns)

		lvlSpecifyingStrs = list_of_strings_from_caps_under_to_cap_lower( specStrsWithSyns )
		# print("expanded list with lvlSpecifyingStrs: ", lvlSpecifyingStrs)

		posCounter = 0
		for hierLi in mkrhierarchiesL :
			oldPosCounter = posCounter

			for room in hierLi :
				# print("room: ", room, " lvlType: ", lvlType, " lvlSpecifyingStrs: ", list_as_str( lvlSpecifyingStrs))

				if list_has_substr_of_str( lvlSpecifyingStrs, room) :

					lvlType = lvlType.replace("*","")

					if not mkrParams[lvlType]["BY"] == None :
						room = caps_under_to_cap_lower( room )

						if not room in levelPosDict.keys() :
							levelPosDict[room] = {}
							levelPosDict[room]['styler'] = str(mkrParams[lvlType]["BY"]).lower()					
							# levelPosDict[room]['styler'] = str(levelPosDict[room]['styler'].lower())
							# print(hierLi," ",levelPosDict)
							# print(posCounter)
							# print(mkrParams[lvlType]["VALUES"][posCounter])

							levelPosDict[room]['val'] = list( mkrParams[lvlType]["VALUES"] )[ posCounter ]

							posCounter += 1
	

	# print("mkrhierarchiesL: ", mkrhierarchiesL)
	# print("levelPosDict: ", levelPosDict)
	return levelPosDict










