import sys
import copy 

import json
import xml.etree.ElementTree as ET


import numpy as np

sys.path.append('..')
from global_imports import external_modules
from global_imports import session_vars as g

from pyutils.io_utils import *
from pyutils.dict_utils import *
from pyutils.string_utils import *

from hierarchies.hierarchies_events import *

from files.file_attributes import FileAttributes


def load_data() :
	
	g.command_history.add('', os.path.basename(__file__),g.inspect.stack()[0][3], locals())

	#Simplify variable names from globals

	ID = str(g.cur['DAT_ID'])

	# Initialise data dictionary
	g.dat[ ID ] = {}	
	g.dat[ ID ][ 'ChanConfig' ] = copy.copy(g.params['GEN']['DEFAULT_CConfig'])

	# print("LOADING DATA ACCORDING TO LINKED FILE: ")
	# print_keys_hierarchy(g.cur['LINKED_FILES']['DATA'])

	g.dat[ID] = traverse_linked_files_and_load(g.dat[ID], g.cur['LINKED_FILES']['DATA'], [])

	if not 'ChanConfig' in g.dat[ID].keys() :
		g.dat[ID]['ChanConfig'] = copy.copy(g.params['GEN']['DEFAULT_CConfig'])

	if 'Samples' in g.dat[ID].keys() :

		g.dat[ID]['ChanConfig']['NUM_CHANS'] = g.dat[ID]['Samples'].shape[0]
		g.dat[ID]['ChanConfig']['NUM_SAMPLES'] = g.dat[ID]['Samples'].shape[1]		

	g.cur['ChanConfig'] = copy.copy(g.dat[ID]['ChanConfig'])	

	if 'NUM_SAMPLES' in g.dat[ID]['ChanConfig']:

		if not 'SAMPLE_RANGE' in g.dat[ID]['ChanConfig']:
			g.dat[ID]['ChanConfig']['SAMPLE_RANGE'] = [0, g.dat[ID]['ChanConfig']['NUM_SAMPLES']]

		if not 'Timestamps' in g.dat[ID].keys():
			g.dat[ID]['Timestamps'], g.dat[ID]['ChanConfig']['TIME_BETWEEN_SAMPLES'] = generate_timestamps(g.dat[ID]['ChanConfig']['NUM_SAMPLES'])

	if 'Windows' not in g.dat[ID].keys():
		g.dat[ID]['Windows'] = {}

	# print('DATA STRUCTURE: ')
	# print_keys_hierarchy(g.dat[ID])
	# quit()
	# print(g.dat[ID]['Timestamps'])


def traverse_linked_files_and_load(dat, linkedFilesDict, datClassHierarchy) :

	# print("datClassHierarchy: ", datClassHierarchy, "linkedFilesDict: ", linkedFilesDict, "outDat: ", dat)

	for dataClass, value in linkedFilesDict.items() :

		datClassVal = caps_under_to_cap_lower(dataClass)

		# print("datClassVal: ", datClassVal)


		if isinstance(value, FileAttributes):

			print("Found me a file  ", value.attr['Name'], " for dataclass: ", dataClass)

			dat[ datClassVal ] = copy.copy(load_file(value, datClassVal))
			# print_keys_hierarchy(dat)


		elif isinstance(value, dict):

			datClassHierarchy.append(datClassVal) 

			if not datClassVal in dat.keys():
				dat[ datClassVal ] = {}

			dat[ datClassVal ] = copy.copy( traverse_linked_files_and_load(dat[ datClassVal ], value, datClassHierarchy) )

	return dat



def load_file(inputFile, dataClass) :

	g.command_history.add('', os.path.basename(__file__),g.inspect.stack()[0][3], locals())
	print("Loading ... ", inputFile.attr['Name'], " dataClass: ", dataClass)

	if inputFile.attr['Extension'] == 'MAT':
		loadedData = sio.loadmat(inputFile.attr['NameAndPath'], matlab_compatible=False)

		if isinstance(loadedData, dict) :
			# print_keys_hierarchy(loadedData)
			firstKey = get_first_key_in_dict(loadedData)

			for key in loadedData.keys() :
				if 'ndarray' in str(type(loadedData[key])) :
					loadedData = loadedData[key]
					break

	elif inputFile.attr['Extension'] in ['TXT'] :
		# print("dataClass: ", dataClass)
		loadedData = np.loadtxt(inputFile.attr['NameAndPath'])

		# print("loadedData.shape: ", loadedData.shape)


	elif inputFile.attr['Extension'] in ['CSV'] :
		loadedData = np.loadtxt(inputFile.attr['NameAndPath'], delimiter=",")		



	elif inputFile.attr['Extension'] in ['JSON'] :
		with open(inputFile.attr['NameAndPath']) as jsonInputFile :
			loadedData = json.load(jsonInputFile)			



	elif inputFile.attr['Extension'] in ['XML', 'SEQ'] :

		with open(inputFile.attr['NameAndPath']) as xmlInputFile:		
			loadedData = ET.parse(xmlInputFile)


	else:

		print("Unsupported file extension ", inputFile.attr['Extension'], " for file ",inputFile.attr['Name'],". Convert the file to .txt or .mat first.")
		return None 

	if 'Events' in inputFile.datClass :
		eventsParams = g.loadDataParams['DATA']['Events']

		if inputFile.attr['Extension'] in ['XML', 'SEQ'] :
			loadedData, eventHierarchies = event_xml_processing(inputFile, loadedData, eventsParams)
		else: # probably csv
			eventHierarchies = get_event_types(inputFile.attr['Name'], eventsParams)

			# print_keys_hierarchy(loadedData)
		
		# print("g.cur['STEP']['OUTPUT']['PLOT_CHANS']['EVENT_MARKER_PARAMS']['EVENT_HIERARCHIES_TO_PLOT']: ", g.cur['STEP']['OUTPUT']['PLOT_CHANS']['EVENT_MARKER_PARAMS']['EVENT_HIERARCHIES_TO_PLOT'])
		if 'EVENT_MARKER_PARAMS' in g.cur['STEP']['OUTPUT']['PLOT_CHANS'].keys() :

			if not 'EVENT_HIERARCHIES_TO_PLOT' in g.cur['STEP']['OUTPUT']['PLOT_CHANS'].keys() :
				g.cur['STEP']['OUTPUT']['PLOT_CHANS']['EVENT_MARKER_PARAMS']['EVENT_HIERARCHIES_TO_PLOT'] = [copy.deepcopy(eventHierarchies)]
				
			else:
				g.cur['STEP']['OUTPUT']['PLOT_CHANS']['EVENT_MARKER_PARAMS']['EVENT_HIERARCHIES_TO_PLOT'] += copy.deepcopy(eventHierarchies)

		# print("g.cur['STEP']['OUTPUT']['PLOT_CHANS']['EVENT_MARKER_PARAMS']['EVENT_HIERARCHIES_TO_PLOT']: ", g.cur['STEP']['OUTPUT']['PLOT_CHANS']['EVENT_MARKER_PARAMS']['EVENT_HIERARCHIES_TO_PLOT'])

			
		# print(" EVENT FILE : ", inputFile.datClass)

	if dataClass.upper()=='Samples'.upper() :

		if isinstance(loadedData, dict) :
			if dataClass in ['Samples', 'Timestamps'] :
				if len(loadedData.keys()) == 1 :
					loadedData = loadedData[get_first_key_in_dict(loadedData)]

		if loadedData.shape[0] > loadedData.shape[1] :
			loadedData = np.transpose(loadedData)	


	return loadedData	



def get_ChanConfig(inputFile, dataClass=None, dataType=None, isMain=False) :

	pipeline = g.params['PIPELINE_PROCESSING']



def generate_timestamps(NUM_SAMPLES):
	g.command_history.add('',os.path.basename(__file__),g.inspect.stack()[0][3], locals())

	sampleRate = g.dat[g.cur['DAT_ID']]['ChanConfig']['SAMPLE_RATE']
	timeBetweenSamples = round((1 / sampleRate), 4)
	timeStamps = np.linspace(0,NUM_SAMPLES,NUM_SAMPLES)
	timeStamps = timeStamps*timeBetweenSamples

	return timeStamps, timeBetweenSamples

	





