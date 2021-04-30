import sys
import copy 
import six.moves.cPickle as pickle
import shutil

import json
import csv

import numpy as np

import filecmp
           
import scipy

sys.path.append('..')

from global_imports import external_modules
from global_imports import session_vars as g

from pyutils.io_utils import *
from pyutils.dict_utils import *

np.set_printoptions(suppress=True)


def save_data() :
	g.command_history.add('', os.path.basename(__file__),g.inspect.stack()[0][3], locals())
	
	print_keys_hierarchy(g.dat)

	saveParams = g.cur['STEP']['OUTPUT']['SAVE_DATA']['PARAMS']

	ID = g.cur['DAT_ID']

	print("Attempting to save data for: ", str(ID), " ", g.cur['STEP']['OUTPUT']['DIR'])


	# if saveParams['SAVE_DATA_IN_ORIGINAL_FOLDER'] :
	# 	baseOutputFolder = g.loadDataParams['DEFAULT_DATA_ROOT']
	# else:
	# 	baseOutputFolder = g.params['OUT_ROOT']	

	baseInputFolder = g.loadDataParams[ 'DEFAULT_DATA_ROOT' ]
	baseOutputFolder = g.params[ 'OUT_ROOT' ]	

	if saveParams[ 'ADD_STEP_AS_NEW_SUB_FOLDER' ] :
		saveOutputDir = check_output_path_str( [ baseOutputFolder, saveParams['SUB_FOLDER'], g.cur['STEP']['OUTPUT']['DIR'].upper() ] )
	else:
		saveOutputDir = check_output_path_str( [ baseOutputFolder, saveParams['SUB_FOLDER'] ] )

	saveInputDir = check_output_path_str( [ baseInputFolder, saveParams['SUB_FOLDER'] ] )	

	if saveParams['ADD_STEP_TO_FILENAME_SUFFIX'] :
		baseFilename = create_filename( [ g.cur['DAT_ID'],  caps_under_to_cap_lower(g.cur['STEP']['OUTPUT']['DIR']), "" ] ) 
	
	else :
		baseFilename = create_filename( [ g.cur['DAT_ID'], "" ] )

	if g.params['GEN']['FIRST_SAVE'] :

		copy_user_defined_paradigm_files()
		g.params['GEN']['FIRST_SAVE'] = False

	# loadDataParamsFile = g.params['PIPELINE_STEPS'][0]['LOAD_DATA']['PARAMS_FILE']
	# loadDataParams = g.loadDataParams['LOAD_DATA_PARAMS']

	# save_data_to_file(g.params['OUT_ROOT'], g.loadDataFile.attr['NameNoExt'], loadDataParams, 'JSON')


	# print_keys_hierarchy(g.dat, "G.DAT hierarchy")

	traverse_data_dict_and_save(g.dat[ID], saveOutputDir, saveInputDir, baseFilename, saveParams['SAVE_INPUT_DATA'], baseClassParams=[], valuesHierarchy=[])


def copy_user_defined_paradigm_files() :

	shutil.copy(g.paradigmFile.attr['NameAndPath'], g.params['OUT_ROOT'])
	shutil.copy(g.loadDataFile.attr['NameAndPath'], g.params['OUT_ROOT'])

# def save_nested_dict_to_structured_folders_and_flat_file(outPath, filename, inDict, dataClass ) :

# 	for key, val in inDict.items():

# 		# check if you've found the gold (a value that is a dict, and its keys are ints (channel nums))
# 		# then iterate through and add
# 		if isinstance(val, dict):

# 			if isinstance(get_first_key_in_dict(val), int):
# 				nChans = len(val.keys())
# 				maxNsamples = 0

# 				for tempVal in val.values():
# 					print("tempval: ", tempVal)
# 					if len(tempVal) > maxNsamples :
# 						maxNsamples = len(tempVal)

# 				outArray = np.zeros(shape=(nChans, maxNsamples))
# 				print("outArray.shape:" , outArray.shape)

# 				for channel, indices in val.items() :
# 					print(indices)
# 					outArray[channel, 0:len(indices)] = np.array(indices).astype(int)

# 				print(outArray)
# 				outArray = outArray[np.where(outArray!=0)]

# 				fileNameAndPath = create_dir([outPath, key]) + filename
# 				np.savetxt(fileNameAndPath, outArray, delimiter=',', fmt='%12d')
#
# 		# else keep looking
# 			else:
# 				save_nested_dict_to_structured_folders_and_flat_file(outPath, filename, val, dataClass)



def	traverse_data_dict_and_save( dat, saveOutputDir, saveInputDir,  baseFileName, shouldSaveInputData, baseClassParams=[], valuesHierarchy=None, hierArchicalOutputDir=None, outFileName=None, currentBaseDataClass=None ) :

	# print("originalOutputDir: ", originalOutputDir, " baseFileName: ", baseFileName, " hierArchicalOutputDir: ", hierArchicalOutputDir, " outfilename: ", outFileName)

	for dataClass, value in dat.items() :

		if dataClass in g.loadDataParams[ 'DATA' ].keys() :
			baseClassParams = g.loadDataParams[ 'DATA' ][ dataClass ]
			currentBaseDataClass = dataClass

		if not shouldSaveInputData :
			if not 'IS_OUTPUT_DATA' in baseClassParams.keys() :
				# print("Don't save ", currentBaseDataClass)
				continue


		# print( "Trying to save dataclass: ", dataClass , " of base class: ", baseClassParams)

		if ( not(dataClass.upper() in [ 'SAMPLES', 'TIMESTAMPS', 'CHANCONFIG' ]) or ( not g.cur[ 'STEP' ][ 'STEP' ].upper() in ['DETECT', 'OPERATENN', 'OPERATE_NN'] )) :

			if not 'IS_OUTPUT_DATA' in baseClassParams.keys() :
				originalDir = str( create_dir( [ saveInputDir ] ) )
					# print("originalDir: ", originalDir)

			else:
				originalDir = str( create_dir( [ saveOutputDir ] ) )


			hierArchicalOutputDir = str( create_dir( [ originalDir, currentBaseDataClass ] ))

			if ( outFileName == None ) or ( baseFileName == outFileName ) :
				outFileName = str(create_filename([baseFileName, dataClass, ""]))	

			else :
				outFileName = str(create_filename([outFileName, dataClass, ""]))	
			# print("dataClass: ", dataClass, " g.loadDataParams.keys(): ", g.loadDataParams['DATA'].keys())

			if ( baseClassParams['EXTENSIONS']['OUTPUT'] == 'JSON' ) or ( 'ndarray' in str(type(value)) ) or 'HAS_HEADERS' in baseClassParams.keys():
				# print("__", dataClass.upper(), "__")
				# if isinstance(dat[dataClass], dict) :
				# 	print_keys_hierarchy(dat[dataClass])

				save_data_to_file( hierArchicalOutputDir, outFileName, dat[dataClass], baseClassParams, dataClass )



			elif isinstance(value, dict) :
				# print_keys_hierarchy(value, "value")
				# print("dataclass: ", dataClass)
				# print("dat[ dataClass ]: ", g.dat)
				traverse_data_dict_and_save( dat[ dataClass ], saveOutputDir, saveInputDir, baseFileName, shouldSaveInputData, baseClassParams, valuesHierarchy, hierArchicalOutputDir, outFileName, currentBaseDataClass )

			if 'SAVE_HIERARCHY_IN_OUTPUT_FOLDERS' in baseClassParams.keys():
				if baseClassParams['SAVE_HIERARCHY_IN_OUTPUT_FOLDERS']:
					hierArchicalOutputDir = remove_childmost_level_from_path(hierArchicalOutputDir)

			if check_if_more_than_two_chunks(outFileName, g.params['SEP']):
				outFileName = remove_last_chunk_from_str(outFileName, g.params['SEP'])






def save_data_to_file(outPath, filename, datToSave, datParams, dataClass) :

	g.command_history.add('', os.path.basename(__file__),g.inspect.stack()[0][3], locals())
	extensionToSave = datParams['EXTENSIONS']['OUTPUT'].upper()

	if dataClass.upper()=='SAMPLES' :
		filename = str(create_filename([filename, caps_under_to_cap_lower(g.cur['STEP']['STEP']), ""]))

	if len(filename) > 200 :
		filename=filename.replace("_","").replace("e","").replace("a","").replace("i","").replace("o","").replace("Chns","").replace("d","")

	if len(filename) > 180 :
		filename=filename[ : 180 ]

	filenameWithExt = create_filename([filename, extensionToSave.lower()])
	filenameAndPath = outPath + filenameWithExt

	if extensionToSave == 'CSV' :
		# print("datTosave for CSV: ", datToSave)

		if isinstance(datToSave, dict) :
			# print( "dataClass: ", dataClass, "extensionToSave: ", extensionToSave )
			# print_keys_hierarchy( datToSave )			
			# print("found ourselves a dict!")
			## Dict of format {RowDescriptor: {colHeader, rowVal},{colHeader, rowVal}, ... 

			for keyz, valz in datToSave.items() :
				filenameWithExt = create_filename([filename, keyz, extensionToSave.lower()])
				filenameAndPath = outPath + filenameWithExt	

				row = ""			

				with open(filenameAndPath, 'w') as outFile :
					csvWriter = csv.writer( outFile )					
					if isinstance(valz, dict) :
						serializabledValz = make_dict_serializable(valz)

						# print("row! ", row)
						valuesAsList = list(serializabledValz.values())

						if 'ndarray' in str(type(valuesAsList[0])) :
							iteratorVal = 0
							for valNDarray in valuesAsList :
								valNDarray2D = np.array([np.round(valNDarray, 4).astype(float)]).reshape(-1,1)
								if iteratorVal == 0 :
									concatenatedArrays = np.copy(valNDarray2D)
								else :
									concatenatedArrays = np.hstack(( concatenatedArrays, np.copy(valNDarray2D) ))
								# print("concatenatedArrays: ", concatenatedArrays.shape)
								iteratorVal += 1


							if np.issubdtype(concatenatedArrays.dtype, np.integer) :
								formt = '%8d'
							else :
								formt = '%.4f'		

							hdrs = list_as_str(list(valz.keys()),",")

							np.savetxt( filenameAndPath, concatenatedArrays.astype(float),  delimiter=",", fmt=formt, header=hdrs )

						else :
							if row=="" :
								row = [ "Desc" ] + list( serializabledValz.keys() )
								csvWriter.writerow( row )		
							row = [keyz] + round_items_in_list( valuesAsList, 4)
							csvWriter.writerow( row )

					elif isinstance(valz, int) or isinstance(valz, float) :
						if row=="" :
							row = list(datToSave.keys())
							csvWriter.writerow( row )

						row = round_items_in_list( list( datToSave.values( ) ), 4)

						# print("row: ", row)
						csvWriter.writerow( row )	
						break					

		# print("np.issubdtype(datToSave.dtype, np.integer): ", np.issubdtype(datToSave.dtype, np.integer))
					elif isinstance(valz, list) :
						print( "dataClass: ", dataClass, "extensionToSave: ", extensionToSave )
						print( datToSave )			
						row = ""			
				
						for item in valz :
							if isinstance(item, dict) :
								if row=="" :
									row = list( item.keys() )
									csvWriter.writerow( row )

								row = round_items_in_list( list(item.values( )), 4)

								print("row: ", row)
								csvWriter.writerow( row )							
				
		else: #is numpy
			if np.issubdtype(datToSave.dtype, np.integer) :
				formt='%8d'

			else:
				formt='%.4f'

			existsNonNegOnes = np.where((datToSave < -1) | (datToSave > -1))
			# print("existsNonNegOnes: ",len(existsNonNegOnes[0]))
			if len(existsNonNegOnes[0]) > 0:
				np.savetxt(filenameAndPath, datToSave,  delimiter=",", fmt=formt)
	
	elif extensionToSave == 'MAT' :
		# To comply with scipy io request for a dict -- wrap np array in a dict.
		if 'ndarray' in str(type(datToSave)) :
			matFromScipy = {}
			matFromScipy['npArray'] = datToSave
			datToSave = matFromScipy

		scipy.io.savemat(filenameAndPath, datToSave, appendmat=False)

	elif extensionToSave == 'BIN' :
		with open(filenameAndPath, 'wb') as outFile :
			pickle.dump(datToSave, outFile)		

	elif extensionToSave == 'JSON' :
		# print("prior to serialization: ", datToSave)
		serializabledDict = make_dict_serializable(datToSave)
		# print("post serialization: ", serializabledDict)

		with open(filenameAndPath, 'wb') as outFile :
			jsonStr = json.dumps(serializabledDict, sort_keys=False, indent=4)
			outFile.write(jsonStr.encode('utf8'))







