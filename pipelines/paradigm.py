import sys
import os

import copy 

from os import listdir, sep

upTwoLevels = '..' + os.path.sep + '..'
sys.path.append(upTwoLevels)
sys.path.append('..')

import glob
import json

from files.file_attributes import FileAttributes
from pyutils.io_utils import *


# from global_imports import external_modules

class Paradigm():

	def __init__(self, filename, rootPyPath) :

		self.paradigmSubFolder = 'user_defined_paradigms'

		self.rootPyPath = rootPyPath

		self.paradigmFileName = filename

		self.params = self.load_params_file(self.paradigmFileName)

		self.paradigmFile = copy.copy(self.fileAttributes)

		self.loadDataParams = self.get_load_params()
		self.loadDataFile = copy.copy(self.fileAttributes)


		self.miscParams = self.get_misc_params()

		self.create_extra_params()



	def load_params_file(self, file):

		paramsFileAndPath = self.verify_paradigm_file_and_path_location(file)

		self.fileAttributes = FileAttributes(paramsFileAndPath)

		return load_json_file(self.fileAttributes.attr['NameAndPath'])



	def get_load_params(self):

		if 'PARAMS_FILE' in self.params['PIPELINE_STEPS'][0]['LOAD_DATA'].keys() :
			paramsFile = self.params['PIPELINE_STEPS'][0]['LOAD_DATA']['PARAMS_FILE']
			print("paramsFile: ", paramsFile)
			if len(paramsFile) > 0:
				loadDataParams = self.load_params_file(paramsFile)['LOAD_DATA_PARAMS']
				self.params['PIPELINE_STEPS'][0]['LOAD_DATA']['PARAMS'] = loadDataParams
			else:
				loadDataParams = self.params['PIPELINE_STEPS'][0]['LOAD_DATA']['PARAMS']

		return loadDataParams



	def get_misc_params(self):

		
		if 'MISC_PARAMS_FILE' in self.params.keys() :
			paramsFile = self.params['MISC_PARAMS_FILE']
			print("paramsFile: ", paramsFile)
			if len(paramsFile) > 0:
				miscParams = self.load_params_file(paramsFile)['MISC_PARAMS']

		return miscParams		



	def verify_paradigm_file_and_path_location(self, fileToVerify):

		if not '.' in fileToVerify :
			fileToVerify += '.json'
		if not '.JSON' in fileToVerify.upper() :
			fileToVerify += '.json'

		cwdDefaultPathAndFile = os.getcwd() + os.path.sep + self.paradigmSubFolder + os.path.sep + fileToVerify
		cwdRootPathAndFile = os.getcwd() + os.path.sep + fileToVerify

		pyRootDefaultPathAndFile = self.rootPyPath + os.path.sep + self.paradigmSubFolder + os.path.sep + fileToVerify
		pyRootPathAndFile = self.rootPyPath + os.path.sep + fileToVerify
		
		FileNameAndPath = fileToVerify

		if os.path.exists(cwdDefaultPathAndFile):
			return cwdDefaultPathAndFile

		elif os.path.exists(cwdRootPathAndFile):
			return cwdRootPathAndFile

		if os.path.exists(pyRootDefaultPathAndFile):
			return pyRootDefaultPathAndFile

		elif os.path.exists(pyRootPathAndFile):
			return pyRootPathAndFile			

		elif os.path.sep in fileToVerify:
			return fileToVerify

		else:
			print("Could not find the specified paradigm file, ", fileToVerify ,". Please make sure it is a .json file and is located in:")
			print(self.rootPyPath + os.path.sep + self.paradigmSubFolder)
			raise ValueError



	def create_extra_params(self) :
		print(self.params)
		print(self.loadDataParams)

		self.params['GEN'] = {}

		if isinstance(self.miscParams, dict) :
			self.params['GEN']['MISC_PARAMS'] = self.miscParams

		self.params['GEN']['PARADIGM_FILE_NAME'] = self.paradigmFile.attr['Name']	
		self.params['GEN']['DEFAULT_OUTPUT_ROOT'] = create_dir(pathList=([self.loadDataParams['DEFAULT_OUTPUT_ROOT'], self.paradigmFile.attr['NameNoExt']]), addDayTime=True)
		self.params['OUT_ROOT'] = self.params['GEN']['DEFAULT_OUTPUT_ROOT']
		self.params['GEN']['LOAD_PROCESSED_LEVEL'] = self.loadDataParams['DATA']['Samples']['LOAD_PROCESSED_LEVEL']	
		self.params['GEN']['FIRST_SAVE'] = True

		# CHAN CONFIG
		if 'DEFAULT_CConfig' in self.loadDataParams['DATA']['ChanConfig'].keys() :
			self.params['GEN']['DEFAULT_CConfig'] = self.loadDataParams['DATA']['ChanConfig']['DEFAULT_CConfig']

		# OUTPUT SEPARATOR
		if self.params['DEFAULT_OUTPUT_PARAMS_FOR_EACH_STEP']['OUTPUT']['SEPARATOR_CHAR'] == None:
			self.params['SEP'] = self.loadDataParams['FILENAME_IDS_MATCHING']['ID_SEPARATOR_CHAR']
			self.params['DEFAULT_OUTPUT_PARAMS_FOR_EACH_STEP']['OUTPUT']['SEPARATOR_CHAR'] = self.params['SEP']
		else:
			self.params['SEP'] = self.params['DEFAULT_OUTPUT_PARAMS_FOR_EACH_STEP']['OUTPUT']['SEPARATOR_CHAR']












