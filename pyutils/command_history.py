import sys
import os
from os import listdir, sep

upTwoLevels = '..' + os.path.sep + '..'
sys.path.append(upTwoLevels)
sys.path.append('..')

import glob
import json

from files.file_attributes import FileAttributes
from global_imports import session_vars as g

from pyutils.dict_utils import *



# from global_imports import external_modules

class CommandHistory():

	def __init__(self) :
		self.commands = []
		self.messages = []


	def add(self, message, className, command, localVars=None):
		currID = ''

		if 'DAT_ID' in g.cur.keys():
			if not g.cur['DAT_ID'] == None:
				currID = g.cur['DAT_ID']

		if localVars == None:
			localVars = ""

		classNameAndCommand =  " " + className + ": " + command + " " + currID + message

		if hasattr(g, 'params'):

			if not g.params['GEN']['MISC_PARAMS']['SUPPRESS_COMMAND_OUTPUT']:
				print('---------')
				print(classNameAndCommand)
				if isinstance(localVars, dict) :
					print_keys_hierarchy(localVars, "Arguments:")
		else:
			print('---------')
			print(classNameAndCommand)
			if isinstance(localVars, dict) :
				print_keys_hierarchy(localVars, "Arguments:")

		self.commands.append(classNameAndCommand)
		self.messages.append(message)



	def verify_paradigm_file_and_path_location(self):
		if not '.' in self.paradigmFileName :
			self.paradigmFileName += '.json'

		cwdDefaultPathAndFile = os.getcwd() + os.path.sep + self.paradigmSubFolder + os.path.sep + self.paradigmFileName
		cwdRootPathAndFile = os.getcwd() + os.path.sep + self.paradigmFileName

		pyRootDefaultPathAndFile = self.rootPyPath + os.path.sep + self.paradigmSubFolder + os.path.sep + self.paradigmFileName
		pyRootPathAndFile = self.rootPyPath + os.path.sep + self.paradigmFileName
		
		FileNameAndPath = self.paradigmFileName

		if os.path.exists(cwdDefaultPathAndFile):
			return cwdDefaultPathAndFile

		elif os.path.exists(cwdRootPathAndFile):
			return cwdRootPathAndFile

		if os.path.exists(pyRootDefaultPathAndFile):
			return pyRootDefaultPathAndFile

		elif os.path.exists(pyRootPathAndFile):
			return pyRootPathAndFile			

		elif os.path.sep in self.paradigmFileName:
			return fileNameAndPath

		else:
			print("Could not find the specified paradigm file, ", self.paradigmFileName ,". Please make sure it is a .json file and is located in:")
			print(self.rootPyPath + os.path.sep + self.paradigmSubFolder)
			raise ValueError











