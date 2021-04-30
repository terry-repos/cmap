import inspect
import time

import numpy as np
from pyutils.dict_utils import *
from pyutils.list_utils import *
from pyutils.time_utils import *
from collections import Counter


def get_start_end_event_windows( evts, sli, pars ) :

	# print("	print_keys_hierarchy(evts):")
	# print_keys_hierarchy(dict(evts))

	startEvts = get_indices_that_contains_all_specified_hierarchies( dict(evts), list(pars["START_EVENTS_HIERARCHY"]), sli, [] )
	endEvts = get_indices_that_contains_all_specified_hierarchies( dict(evts), list(pars["END_EVENTS_HIERARCHY"]), sli, [] )
	# print("startEvts: ", startEvts)	
	# print("endEvts: ", endEvts)


	return startEvts, endEvts



def create_events_for_this_hierarchy_synched_across_chans( evntDc, hierarchy, eventIndex, nRepeats ) :

	evntDc = copy.deepcopy( update_dict_with_a_new_initialised_hierarchy( evntDc, hierarchy )	)
	indicesArray = np.full( shape=(nRepeats), fill_value=eventIndex  )
	evntDc = copy.deepcopy( set_vals_in_dict( copy.deepcopy(evntDc), hierarchy, indicesArray  ))

	return evntDc


	
def get_indices_that_contains_all_specified_hierarchies( hierarDicti, hierarchList, slic=None, explicit=True, indicesRange=[], keysInDict=[], outIndices=np.array([]) ) :

	if isinstance(hierarDicti, dict) :

		for ck, cv in hierarDicti.items() :

			keysInDict.append( ck )
			# print( "ck: ", ck, "hierarchList: ", hierarchList, " keysInDict: ", keysInDict )

			if isinstance(cv, dict) :
				outIndices = copy.deepcopy( get_indices_that_contains_all_specified_hierarchies( cv, hierarchList, slic, explicit, indicesRange, keysInDict, outIndices ))

			elif 'ndarray' in str(type(cv)) :

				if not list_contains_list( hierarchList ) :
					hierarchList = [list(hierarchList)]
				# print(" Post hierarchList: ", hierarchList, " keysInDict: ", keysInDict )


				for hierLst in hierarchList :

					if explicit:
						if all_of_this_list_in_all_of_that_list( hierLst, keysInDict ) :
							missingHiers = []
						else:
							missingHiers = ["Not mutually equal"]

					else:
						missingHiers = copy.deepcopy( remove_this_list_from_that_list( list(keysInDict), list(hierLst) ))


					if len( cv ) > 0 :
						if len( missingHiers ) == 0 :
							# print("len missingHiers = 0 ")
							# if there are valid indices here ...
							# print("slic: ", slic)
							# print("cv: ", cv)
							sourceIndicesArray = cv[ slic ]

							# print("indicesRange: ", indicesRange)

							if len(indicesRange) > 1 :
								matchingIndices =  np.where(( sourceIndicesArray >= indicesRange[0] ) & ( sourceIndicesArray < indicesRange[1] )) 
							else:
								matchingIndices =  np.where( sourceIndicesArray > -1) 
							# print(" Going through hierLst: ", hierLst, " keysInDict: ", keysInDict , "matchingIndices: ", matchingIndices, " sourceIndicesArray.shape: ", sourceIndicesArray)

							if len(matchingIndices) > 0:

								if len(matchingIndices) > 1 :
									colIndices = np.array([])
									for rowIndex, colIndex in zip(matchingIndices[0], matchingIndices[1]) :
										if not colIndices.any() :
											colIndices = sourceIndicesArray[rowIndex, colIndex]
										else:
											colIndices = np.append(colIndices, sourceIndicesArray[rowIndex, colIndex])	

									outIndices = np.append( outIndices , [ np.array(matchingIndices[0]), colIndices ] )									
									# outIndices = [ np.array(matchingIndices[0]), colIndices ]
								else:
									indices = sourceIndicesArray[matchingIndices].astype( int )
									outIndices = np.append(outIndices, indices)					


			if len( keysInDict ) > 0 :
				keysInDict.pop()

	return outIndices


# def get_copy_of_array_from_this_hierarchy( hierarDicti, hierarchList ) :

# 	if isinstance(hierarDicti, dict) :

# 		for ck, cv in hierarDicti.items() :

# 			keysInDict.append( ck )
# 			# print( "ck: ", ck, "hierarchList: ", hierarchList, " keysInDict: ", keysInDict )

# 			if isinstance(cv, dict) :
# 				outIndices = copy.deepcopy( get_indices_that_contains_all_specified_hierarchies( cv, hierarchList, slic, explicit, indicesRange, keysInDict, outIndices ))

# 			elif 'ndarray' in str(type(cv)) :

# 				# print(" hierarchList: ", hierarchList, " keysInDict: ", keysInDict )

# 				if explicit:
# 					if all_of_this_list_in_all_of_that_list( hierarchList, keysInDict ) :
# 						missingHiers = []
# 					else:
# 						missingHiers = ["Not mutually equal"]

# 				else:
# 					missingHiers = copy.deepcopy( remove_this_list_from_that_list( list(keysInDict), list(hierarchList) ))

# 				if len(cv) > 0 :
# 					if len( missingHiers ) == 0 :
# 						# print("len missingHiers = 0 ")
# 					# if there are valid indices here ...
# 						# print("slic: ", slic)
# 						# print("cv: ", cv)
# 						# print( "outindices: ", get_num_non_x_items( outindices ) )
# 						# print( "cv: ", get_num_non_x_items( cv ) )

# 						sourceIndicesArray = cv[slic]

# 						if len(indicesRange) > 1 :
# 							matchingIndices =  np.where(( sourceIndicesArray >= indicesRange[0] ) & ( sourceIndicesArray < indicesRange[1] )) 
# 						else:
# 							matchingIndices =  np.where( sourceIndicesArray > -1) 

# 						if len(matchingIndices) > 0:

# 							if len(matchingIndices) > 1 :
# 								colIndices = np.array([])

# 								for rowIndex, colIndex in zip(matchingIndices[0], matchingIndices[1]) :

# 									if not colIndices.any() :
# 										colIndices = sourceIndicesArray[rowIndex, colIndex]
# 									else:
# 										colIndices = np.append(colIndices, sourceIndicesArray[rowIndex, colIndex])	

# 								outIndices = [ np.array(matchingIndices[0]), colIndices ] )
# 								# print( "outIndices inner: ", get_num_non_x_items( outIndices ) )

# 							else:
# 								indices = sourceIndicesArray[matchingIndices].astype(int)
# 								outIndices = np.append(outIndices, indices)					


# 			if len( keysInDict ) > 0 :
# 				keysInDict.pop()

# 	# print("outindices outer: ", get_num_non_x_items(outindices))

# 	return outIndices	


def map_npwhere_to_event_structured_indices( npWhereArray, nChans ) :
	startTime = time.time()

	chanWithMostEvents, nEvents = Counter(npWhereArray[0]).most_common(1)[0] 
	# print("counter time: ", (time.time()-startTime))
	eventsArr = np.full( shape=( nChans, nEvents ), fill_value=-1 )

	for row in set(npWhereArray[0]) :
		sampleIndices = np.where( npWhereArray[0]==row )[0]
		nSampleIndices = len(sampleIndices)
		eventsArr[ row, 0 : nSampleIndices ] = npWhereArray[1][ sampleIndices ]

	time_taken(startTime, inspect.stack()[0][3])
	return eventsArr


def map_npwhere_to_event_structured_indices_by_row( npWhereArray, row ) :

	sampleIndices = np.where( npWhereArray[0]==row )[0]
	return npWhereArray[ 1 ][ sampleIndices ]



def transform_event_coords_to_channel_array(eventCoordsList, nChans):

	eventIndicesArr = np.copy(get_init_channel_array( nChans ))

	for eventCoords in eventCoordsList :
		eventIndicesArr = insert_index_into_arr( eventIndicesArr, eventCoords[0], eventCoords[1], nChans ) 

	return eventIndicesArr











# def get_ndarrays_that_contains_all_specified_hierarchies( hierarDictionary, hierarchLst, slic=None, keysInDict=[], outSamples=np.array([]) ) :

# 	if isinstance(hierarDictionary, dict) :

# 		for cke, cva in hierarDictionary.items() :

# 			keysInDict.append( cke )
# 			print( "hierarchLst: ", hierarchLst, " keysInDict: ", keysInDict )

# 			if isinstance(cva, dict) :
# 				outSamples = copy.copy( get_ndarrays_that_contains_all_specified_hierarchies( cva, hierarchLst, slic, keysInDict, outSamples ))

# 			elif 'ndarray' in str(type(cva)) :

# 				missingHiers = copy.copy( remove_this_list_from_that_list( list(keysInDict), list(hierarchLst) ))
# 				print("missingHiers: ", missingHiers)

# 				if len( missingHiers ) == 0 :
# 					# if reached valid hiearchy:
# 					print(cva)
# 					return cva

# 			if len( keysInDict ) > 0 :
# 				keysInDict.pop()

# 	return outSamples




def keep_event_indices( inEvDic, rangeToKeep=None, axisToDelete=1, outEvDic={} ) :
	print("rangeToKeep: ", rangeToKeep)
	if isinstance(inEvDic, dict) :

		for hk, hv in inEvDic.items() :

			if isinstance(hv, dict) :

				outEvDic[hk] = {}
				outEvDic[hk] = copy.copy( keep_event_indices( hv, rangeToKeep, axisToDelete, outEvDic[hk] ))

			elif 'ndarray' in str(type(hv)) :

				if axisToDelete==0 :
					# rowRange = range( rangeToKeep[0], rangeToKeep[1] )
					hv = hv[rangeToKeep[0] : rangeToKeep[1], :]

				else :
					eventLocsToRemove = np.where( (hv < rangeToKeep[0]) | (hv > rangeToKeep[1]))
					eventLocsToKeep = np.where( (hv >= rangeToKeep[0]) & (hv <= rangeToKeep[1]))

					# print("eventLocsToRemove: ", len(hv[ eventLocsToRemove ]))
					# print("hv[ eventLocsToRemove ]: ", hv[ eventLocsToRemove ])
					hv[ eventLocsToRemove ] = -1
					hv[ eventLocsToKeep ] -= rangeToKeep[0]


				outEvDic[hk] = np.copy(hv)

	return outEvDic	





def delete_event_indices( inEvDic, rangeToDelete=None, axisToDelete=1, outEvDic={} ) :
	print("rangeToDelete: ", rangeToDelete)

	if isinstance(inEvDic, dict) :

		for hk, hv in inEvDic.items() :

			if isinstance(hv, dict) :

				outEvDic[hk] = {}
				outEvDic[hk] = copy.copy( delete_event_indices( hv, rangeToDelete, axisToDelete, outEvDic[hk] ))

			elif 'ndarray' in str(type(hv)) :

				if axisToDelete==0 :
					hvIndices = np.s_[ rangeToDelete[0] : rangeToDelete[1] ]
					hv = np.delete(hv, hvIndices, axis=0)

				else :
					eventLocsToRemove = np.where( (hv >= rangeToDelete[0]) & (hv <= rangeToDelete[1]) )

					# print("eventLocsToRemove: ", len(hv[ eventLocsToRemove ]))
					# print("hv[ eventLocsToRemove ]: ", hv[ eventLocsToRemove ])
					hv[ eventLocsToRemove ] = -1
					hv[ np.where( hv >= rangeToDelete[1] ) ] -= (rangeToDelete[1] - rangeToDelete[0]) 



				# if len(eventLocsToKeep) > 1 :
				# 	if len(eventLocsToKeep[1]) > 1:
				
				# 		maxNindices = np.max(eventLocsToKeep[1]) + 1

				# 		initArray = np.full(shape=(hv.shape[0], maxNindices), fill_value=-1)
				# 		# print( "hv.shape: ", hv.shape )

				# 		# print(initArray.shape)
				# 		initArray[eventLocsToKeep] = hv[eventLocsToKeep]
				# 		initArray[eventLocsToKeep] -= rangeToKeep[0]

				# 	else :
				# 		initArray = np.full(shape=(hv.shape[0], 1), fill_value=-1)

				# else :
				# 	initArray = np.full(shape=(hv.shape[0], 1), fill_value=-1)
				
				# print("hv: ", hv)
				# print("initArray: ", initArray)
				# print("n hv: ", (hv.shape[0] * hv.shape[1]), " n found: ", len(eventLocsToKeep[0]), " orig shape: ", hv.shape,  ", init.shape: ", initArray.shape )
				# # eventLocsToKeep = np.where( (initArray >= rangeToKeep[0])  & (initArray <= rangeToKeep[1]) )
				# print("hv: ", len(hv[np.where(hv>-1)]) )				
				# print("initArray: ", len(initArray[np.where(initArray>-1)]) )

				outEvDic[hk] = np.copy(hv)

	return outEvDic		