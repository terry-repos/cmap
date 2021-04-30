import sys
import os
from os import listdir, sep
import copy

upTwoLevels = '..' + os.path.sep + '..'
sys.path.append(upTwoLevels)
sys.path.append('..')

from global_imports import external_modules
from global_imports import session_vars as g

import glob

from files.file_attributes import FileAttributes
from pyutils.io_utils import *
from pyutils.list_utils import *
from hierarchies.hierarchies_events import *

# from global_imports import external_modules

class DataSelection():

	def __init__(self, inDataParams) :

		self.dataParams = inDataParams
		self.fileSelectionParams = self.dataParams['OVERARCHING_FILE_SELECTION']

		self.files = []

		self.load_data_classes()

		self.allDataClasses = []

		self.load_files()

		self.fileIDs = self.get_unique_file_ids()

		print(self.fileIDs)

		self.link_fileIDs_and_dataClasses()



	def load_data_classes(self) :

		self.dataClasses = []

		for dataClass in self.dataParams['DATA'].keys():
			print("Loading files for DATACLASS  ", dataClass)
			if self.dataParams['DATA'][dataClass]['LOAD']  :
				self.dataClasses.append(dataClass)

			if 'IS_MAIN_DATA_CLASS' in self.dataParams['DATA'][dataClass].keys()  :

				if self.dataParams['DATA'][dataClass]['IS_MAIN_DATA_CLASS']  :
					self.mainDataClass = dataClass
		print("dataClass: ", self.dataClasses)



	def link_fileIDs_and_dataClasses(self) :

		print("LINKING FILES")
		print("==============")
		self.linkedFilesGroup = []
		filesToIterate = copy.copy(self.files)

		for fileID in self.fileIDs:

			linkedFile = {}
			linkedFile['DAT_ID'] = fileID
			linkedFile['DATA'] = {}

			for file in filesToIterate:
				if file.ids['ALL'] == fileID:
					# print("file.datClass: ", file.datClass)
					datClassAsDict = stack_list_as_hierarchical_dict(file.datClass, {})
					# print("HIERARCH DICT? ", datClassAsDict)
					datClassAsDict = set_childmost_value_from_hierarchical_dict(datClassAsDict, file)
					# print("PRIOR: ", linkedFile['DATA'])

					priorDict = linkedFile['DATA'].copy()
					# print("priorDict: ", priorDict)

					priorDict.update(datClassAsDict)	

					linkedFile['DATA'] = merge_dicts(linkedFile['DATA'].copy(), datClassAsDict)	
					# print("linkedFile: ", linkedFile ,"linkedFile: ", file.attr['Name'])

			self.linkedFilesGroup.append(linkedFile)

		for linkedFile in self.linkedFilesGroup:
			print_filenames_in_hier_dic( linkedFile['DATA'] )

		print(self.linkedFilesGroup)



	def load_files(self):

		self.files = []
		for dataClass in self.dataClasses:
			print("RETRIEVING FILES FOR DATA CLASS :  ", dataClass)
			self.files += self.get_data_selection(dataClass)
		print("============================")
		print("Files selected for loading: ")
		print("============================")

		for file in self.files:
			print(file.attr['Name'])

		print("============================")



	def get_unique_file_ids(self):

		uniqueIDs = []
		print("Getting unique ids for files: ")		
		for file in self.files:
			print(file.attr['Name'])
			uniqueIDs.append(file.ids['ALL'])

		return list(set(uniqueIDs))



	def get_data_selection(self, dataClass) :
		
		dataClassParams = self.dataParams['DATA'][dataClass]

		dataRoot = self.get_data_root( dataClassParams )

		subPaths, subFolderItemsList = self.get_subPaths( dataRoot, dataClassParams )


		if 'DATA_TYPE' in dataClassParams.keys():
			dataType = dataClassParams['DATA_TYPE']

		else:
			dataType = None

		files = []

		for subPath, subFolderItems in zip(subPaths, subFolderItemsList) :

			pathAndWildCard = subPath + "*.*"
			print(pathAndWildCard)
			filesInSubPath = glob.glob(pathAndWildCard)
			print(filesInSubPath)
			for fileAndPath in filesInSubPath :

				file = FileAttributes(fileAndPath, dataClass, dataType)

				file.datClass = [ dataClass ]

				file = self.get_data_identifiers(file)

				if self.match_containing_str( dataClassParams, file) :

					if self.match_containing_extension(dataClassParams, file) :

						if self.file_conforms_to_overarching_file_selection(file) :

							if dataClass.upper() in ['EVENTS', 'WINDOWS', 'EVNTWNMAPS'] :
		
								# automationType = get_automation_type( file.attr['Name'], dataClassParams)
								eventTypes = get_event_types( file.attr['Name'], dataClassParams)
								print(file.attr['Name'], " ", eventTypes)
								# file.datClass.append( automationType )
								file.datClass += eventTypes
								print(file.datClass)

							# else:
							# 	if len(subFolderItems) > 0 :
							# 		file.datClass += subFolderItems		
													
							files.append(file)

							if not file.datClass in self.allDataClasses:
								self.allDataClasses.append(file.datClass)

		if len(subPaths) < 1 :
			print("Could not produce the folders for iteration")

		elif len(files) == 0 :
			print("No files were found. Check the paths specified and the'ONLY_ANALYSE_FILES' parameters in your paradigm configuration.")
		
		else :
			print("Returning ", len(files),  " files.")

		return files



	def get_data_root(self, dataClassParams) :

		print("dataClassParams: ", dataClassParams)
		defaultDataRoot = self.dataParams['DEFAULT_DATA_ROOT'] 
		print("dataClassParams['DATA_ROOT']: ", dataClassParams['DATA_ROOT'])
		if dataClassParams['DATA_ROOT'] is None:
			return defaultDataRoot
		else:
			if os.path.exists(dataClassParams['DATA_ROOT']) :
				return dataClassParams['DATA_ROOT']

			else:
				defaultDataRoot = check_output_path_str([defaultDataRoot, dataClassParams['DATA_ROOT']])
		self.dataParams['DEFAULT_DATA_ROOT'] = defaultDataRoot
		
		return defaultDataRoot



	def get_subPaths(self, rootPath, dataClassParams ) :

		if not self.fileSelectionParams['IN_THESE_SUB_FOLDERS'] == None:
			onlyInclTheseSubDirs = self.fileSelectionParams['IN_THESE_SUB_FOLDERS']

		if 'onlyInclTheseSubDirs' in locals():
			if "*" in onlyInclTheseSubDirs :
				subFolders = [''.join([pathContents[0],os.sep]) for pathContents in os.walk(rootPath)]
			else:
				subFolders = [''.join([pathContents[0],os.sep])for pathContents in os.walk(rootPath) if pathContents[0] in onlyInclTheseSubDirs]
		else:
			subFolders = [''.join([pathContents[0],os.sep]) for pathContents in os.walk(rootPath)]
		print()

		subFolderItems = []
		subFoldersLevels = []

		for subFolder in subFolders :
			subFolderItems.append(get_sub_path_items(rootPath, subFolder ) )	

		if 'EXCLUDING_THESE_SUB_FOLDERS' in self.fileSelectionParams.keys() :

			subFoldersToExcl = self.fileSelectionParams['EXCLUDING_THESE_SUB_FOLDERS']
			print("subFoldersToExcl: ", subFoldersToExcl)
			if not subFoldersToExcl == None :
				if len(subFoldersToExcl) > 0 :
					if subFoldersToExcl == '*' :
						subFolders = []

					else :
						outSubFolders = []
						outSubItems = []
						for sub, item in zip(subFolders, subFolderItems) :
							doNotAddThisFolder = False
							for folderToExclude in subFoldersToExcl:
								if folderToExclude in item:
									doNotAddThisFolder = True

							if not doNotAddThisFolder:
								outSubFolders.append(sub)
								outSubItems.append(item)

						subFolders = copy.deepcopy(outSubFolders)
						subFolderItems = copy.copy(outSubItems)

		return subFolders, subFolderItems



	def get_data_identifiers(self, file) :

		file.ids = {}
		separator = self.dataParams['FILENAME_IDS_MATCHING']['ID_SEPARATOR_CHAR']
		for idClass, idPos in self.dataParams['FILENAME_IDS_MATCHING']['IDS_POSITION'].items():

			if not idPos == None:
				file.ids[idClass] = self.match_ID(file.attr['NameNoExt'], idClass, idPos, separator )

				print("file: ", file.attr['Name'], " file.ids[idClass]: ", file.ids[idClass])

		allIDs = ""
		idI = 0

		for idStr in file.ids.values() :
			idI += 1
			if idI > 1:
				allIDs +=  g.params['SEP']
			allIDs += idStr 
		file.ids['ALL'] = allIDs

		return file



	def match_containing_str(self, dataClassParams, file):

		foundStr = False
		if not "*" in dataClassParams['CONTAINING_STRINGS'] and not "" in dataClassParams['CONTAINING_STRINGS']:

			for subStr in dataClassParams['CONTAINING_STRINGS']:
				if subStr.upper() in file.attr['Name'].upper():
					foundStr = True

		else:
			foundStr = True

		if foundStr :
			print("File ",file.attr['Name'] ," MATCHES  search string ", dataClassParams['CONTAINING_STRINGS'])

		else :
			print("File ",file.attr['Name'] , "does not match search string ", dataClassParams['CONTAINING_STRINGS'])

		return foundStr



	def match_ID(self, filenameNoExt, idClass, idPos, sep) :
		print("filenameNoExt: ", filenameNoExt)
		sepPoses = [pos for pos, char in enumerate(filenameNoExt) if char == sep]
		if len(sepPoses) > 0:
			if idPos == 0 :
				idStr = filenameNoExt[0:sepPoses[0]]

			elif len(sepPoses) > 1 :
				idStr = filenameNoExt[(sepPoses[0]+1):sepPoses[1]]
			else:
				idStr = filenameNoExt[(sepPoses[0]+1):]
		else:
			idStr = ""

		return idStr.upper()



	def match_containing_extension(self, dataClassParams, file):

		foundExtension = False
		if not "*" in dataClassParams['EXTENSIONS']['INPUT'] and not "" in dataClassParams['EXTENSIONS']['INPUT']:

			for extension in dataClassParams['EXTENSIONS']['INPUT']:
				if extension.upper() in file.attr['Extension'].upper() or extension.upper()==file.attr['Extension'].upper():
					foundExtension = True

		if foundExtension :
			print("File  extension ", file.attr['Extension'] ," MATCHES extension(s) ", dataClassParams['EXTENSIONS']['INPUT'])

		else:
			print("File extension ", file.attr['Extension'] ," does not match extension(s) ", dataClassParams['EXTENSIONS']['INPUT'])

			foundExtension = True

		return foundExtension



	def file_conforms_to_overarching_file_selection(self, file) :

		print("---------------")
		print("Checking whether ", file.attr['Name'], " conforms with file matching requests:")
		fileConforms = True

		for dataClassToLoadKey, dataClassToLoadParams in self.fileSelectionParams.items() :

			if dataClassToLoadKey == 'WITH_THESE_SUBJECT_IDS':

				if not "*" in dataClassToLoadParams and not "" in dataClassToLoadParams:
					if not file.ids['SUBJECT'].upper() in dataClassToLoadParams:
						print("This files subject id ", file.ids['SUBJECT'], " does not conform with the requested subject ID(s) ", file.ids['SUBJECT'])
						fileConforms = False

			elif dataClassToLoadKey == 'WITH_THESE_RECORDING_IDS':

				if not "*" in dataClassToLoadParams and not "" in dataClassToLoadParams:
					if not file.ids['RECORDING'].upper() in dataClassToLoadParams:
						print("This file ", file.ids['RECORDING'] ," does not conform with the requested recording ID(s) ", file.ids['RECORDING'])
						fileConforms = False

			elif dataClassToLoadKey == 'WITH_PREFACES':

				if not "*" in dataClassToLoadParams and not "" in dataClassToLoadParams:
					found_matching = False
					for preface in dataClassToLoadParams :
						if file.attr['NameNoExt'].upper().startswith(preface.upper()) :
							found_matching = True
						print("This file does not conform with the requested file prefaces.")

					if not found_matching :
						fileConforms = False

			elif dataClassToLoadKey == 'EXCLUDING_THESE_SUBJECT_IDS' :
				if file.ids['SUBJECT'].upper() in dataClassToLoadParams :
					print("This files subject id ", file.ids['SUBJECT'], " has been requested to be excluded")
					fileConforms = False

		return fileConforms
	


	def match_companion_data_selection(self, mainFile) :

		g.command_history.add(mainFile.attr['Name'], self.__class__.__name__,g.inspect.stack()[0][3])		
		companionFilesToReturn = dict()
		for companionDataClass in self.companionDataClasss :

			companionParams = self.dataParams['COMPANION_DATA_CLASSES'][companionDataClass]
			print("companionParams: ", companionParams, " companionDataClass: ", companionDataClass)
			if companionParams['PER_DATASET_COMPANION_FILE'] :
				companionFiles = self.get_data_selection( companionParams, companionDataClass, isMainDataClass=False)
			else:
				companionFiles = self.get_general_file(companionParams)

			for companionFile in companionFiles :

				if not companionFile.attr['NameAndPath'] == mainFile.attr['NameAndPath'] :
					if companionFile.ids['SUBJECT'] == mainFile.ids['SUBJECT'] :
						if companionFile.ids['RECORDING'] == mainFile.ids['RECORDING'] :
							companionFilesToReturn[companionDataClass] = companionFile

		return companionFilesToReturn
