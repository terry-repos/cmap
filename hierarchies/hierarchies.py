import os

import copy

from pyutils.io_utils import *
from pyutils.dict_utils import *


from pyutils.list_utils import *
from pyutils.string_utils import *


def get_all_hierarchies_lists_containing_valid_indices(aHierDic, currHierarchy=[], outHierarchies=[]) :

	if isinstance(aHierDic, dict) :
		for K, V in aHierDic.items() :
			currHierarchy.append(str(K))

			if isinstance(V, dict) :

				outHierarchies = get_all_hierarchies_lists_containing_valid_indices(V, currHierarchy, outHierarchies)

			elif 'ndarray' in str(type(V)) :

				indsOfInds = np.where(V > -1)
				# print("currHierarchy: ", list_as_str(currHierarchy), " inds of inds: ", indsOfInds)
				if len(indsOfInds) > 0 :
					if len(indsOfInds[0]) > 0 :
						outHierarchies.append( list(currHierarchy) )

			currHierarchy.pop()

	return outHierarchies






def get_ndarrays_that_contains_all_specified_hierarchies_casual( hierarDictionary, hierarchLst, slic=None, keysInDict=[], outSamples=np.array([]) ) :
	# print_keys_hierarchy(hierarDictionary)
	# print("hierarchLst: ", hierarchLst)
	if isinstance(hierarDictionary, dict) :

		for cke, cva in hierarDictionary.items() :

			keysInDict.append( cke )
			# print( "hierarchLst: ", hierarchLst, " keysInDict: ", keysInDict )

			if isinstance(cva, dict) :
				outSamples = copy.deepcopy( get_ndarrays_that_contains_all_specified_hierarchies( cva, hierarchLst, slic, keysInDict, outSamples ))

			elif 'ndarray' in str(type(cva)) :

				missingHiers = copy.deepcopy( remove_this_list_from_that_list( list( keysInDict ), list( hierarchLst ) ))

				if len( missingHiers ) == 0 :
					# if reached valid hiearchy:

					if len( np.where((cva > -1) | (cva < -1))[0] ) > 0 :
						print( "Reached valid hierarchy ", hierarchLst, " with keysInDict: ", keysInDict, " (", list_as_str(cva.shape,"_"), ")" )
						return copy.deepcopy(cva)

				else :
					print("Hierarchy missing: ", missingHiers, " for ", hierarchLst)

			if len( keysInDict ) > 0 :
				keysInDict.pop()

	return outSamples


def get_intersecting_win_indices_for_this_hierarchy_from_that_hierarchy( hierarDictionary, thisHierarchy, thatHierarchy, keysInDict=[], outSamples=np.array([]) ) :

	theseIndices = get_vals_from_dict_with_this_hierarchy( hierarDictionary, thisHierarchy )
	thoseIndices = get_vals_from_dict_with_this_hierarchy( hierarDictionary, thatHierarchy )

	overlappingIndices = np.array( [] )

	for theseInds in theseIndices :
		if theseInds in thoseIndices :
			overlappingIndices = np.append(overlappingIndices, theseInds )

	# print("theseIndices: ", theseIndices)
	# print("thoseIndices: ", thoseIndices)
	# print("overlappingIndices: ", overlappingIndices)
	# # quit()


	return overlappingIndices	


def get_complementing_win_indices_for_this_hierarchy_from_that_hierarchy( hierarDictionary, thisHierarchy, thatHierarchy, keysInDict=[], outSamples=np.array([]) ) :

	theseIndices = get_vals_from_dict_with_this_hierarchy( hierarDictionary, thisHierarchy )
	thoseIndices = get_vals_from_dict_with_this_hierarchy( hierarDictionary, thatHierarchy )

	complementingIndices = np.array( [] )

	for theseInds in theseIndices :
		if not theseInds in thoseIndices :
			complementingIndices = np.append(complementingIndices, theseInds )

	for thoseInds in thoseIndices :
		if not thoseInds in complementingIndices :
			if not thoseInds in theseIndices :
				complementingIndices = np.append(complementingIndices, thoseInds )				

	print("theseIndices: ", theseIndices)
	print("thoseIndices: ", thoseIndices)
	print("overlappingIndices: ", overlappingIndices)

	return complementingIndices	



def get_intersect_of_all_hierarchies_and_specified( allFullHiers, specHiers ) :

	outSpecs = []
	
	for specHier in specHiers :

		if list_contains_list( specHiers ) :
			formatSpecHier = list_of_strings_from_caps_under_to_cap_lower( specHier )

		else:
			formatSpecHier = list_of_strings_from_caps_under_to_cap_lower( specHiers )	

		# print("formatSpecHier: ", formatSpecHier)
		for fullHier in allFullHiers :
			if all_of_this_list_in_that_list(formatSpecHier, fullHier) :
				if not fullHier in outSpecs:
					outSpecs.append(fullHier)

	return outSpecs



def get_full_hierarchies(inDictiona, partialHierarchyList) :

	allHierarchies = get_all_hierarchies_lists_containing_valid_indices( inDictiona )
	fullHierarchies = copy.deepcopy(get_intersect_of_all_hierarchies_and_specified( allHierarchies, partialHierarchyList ))

	return fullHierarchies

	

def check_hierarchy_in_dict_explicit( hierarchyDct, definedHierarchy ) :
	allHierarchies = get_all_hierarchies_lists_containing_valid_indices( hierarchyDct )
	print("allHierarchies: ", allHierarchies, " definedHierarchy: ", definedHierarchy)
	if definedHierarchy in allHierarchies:
		return True
	else:
		return False	



def check_new_hierarchy_matches_required_hierarchy( inHiArch, lvlTyps, lvlSpecStrsAll, synoDic ) :

	for lvlType, lvlSpecStrs in zip(lvlTyps, lvlSpecStrsAll) :

		specStrsWithSyns = expand_list_to_include_synonyms( lvlSpecStrs, synoDic )
		lvlSpecifyingStrs = list_of_strings_from_caps_under_to_cap_lower( specStrsWithSyns )

		specifier = get_item_in_bothLists(inHiArch, lvlSpecifyingStrs)
		if (specifier==None) :
			if check_level_is_required(lvlType)	:
				return False

	return True
	




def check_level_is_required(levelTypeStr) :
	if "*" in levelTypeStr:
		return True
	else:
		return False




def subset_to_exclude_file_name_hierarchy(inHierarchies, fileNameHierArchies) :

	outHierarchies = []
	for hierL in inHierarchies :
		outL = remove_this_list_from_that_list(fileNameHierArchies, hierL)
		outHierarchies.append(outL)

	return outHierarchies



def subset_only_maximal_hierarchies(inHierarchies) :

	maxLengthHierarchy = 0
	for inHierarchy in inHierarchies :
		if len(inHierarchy) > maxLengthHierarchy :
			maxLengthHierarchy = len(inHierarchy)

	outHierarchies = [ outHierarchy for outHierarchy in inHierarchies if len(outHierarchy) == maxLengthHierarchy ]		

	return outHierarchies	


def get_synonyms_dict( ) :

	if 'MISC_PARAMS' in g.params['GEN'].keys() :

		if isinstance(g.params['GEN']['MISC_PARAMS'], dict) :

			if 'CONVERT_PARAMS_FROM_TO' in g.params['GEN']['MISC_PARAMS'].keys() :

				synonymsDict = g.params['GEN']['MISC_PARAMS']['CONVERT_PARAMS_FROM_TO']

	if not 'synonymsDict' in locals():
		synonymsDict = None

	return synonymsDict




def expand_list_to_include_synonyms( strLst , synDic ) :

	synonymsDic = get_synonyms_dict( )
	lstWithSyns = []

	for strIt in strLst :
		synonymStrs = add_synonyms( strIt, synDic )
		lstWithSyns += synonymStrs 

	uniqueStrs = set(lstWithSyns)

	return list(set(lstWithSyns))






