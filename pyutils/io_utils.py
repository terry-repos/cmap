import sys

sys.path.append('..')

from global_imports import external_modules
from global_imports import session_vars as g

import scipy.io as sio

from pathlib import Path, PurePath
import glob

from os import listdir, sep
import os

import pwd 
import grp

import datetime

import json

from pyutils.list_utils import *
from pyutils.string_utils import *


def create_dir(pathList=None, addDayTime=False, dontCreateDir=False) :

	g.command_history.add('', os.path.basename(__file__),g.inspect.stack()[0][3], locals())
	
	outpath = str(check_output_path_str(pathList, addDayTime) + os.path.sep)
	purePath = None

	purePath = PurePath(outpath)

	purePathsList = list(purePath.parts)

	pathBuilder = ''

	if not os.path.exists(outpath):

		for pathI in purePathsList:
		
			if not pathI == os.path.sep:
				pathBuilder += os.path.sep + pathI

				if not os.path.exists(pathBuilder) :

					set_os_permission_to_own(pathBuilder)

					try:
						os.makedirs(pathBuilder, mode=0o777)

					except Exception as e :
						print(e)

	return add_trailing_separator_in_path(outpath)



def remove_childmost_level_from_path(inPath) :
	purePath = PurePath(inPath)

	purePathsList = list(purePath.parts)
	del purePathsList[-1]
	pathBuilder = ''

	for pathI in purePathsList :
	
		if not pathI == os.path.sep:
			pathBuilder += os.path.sep + pathI

	# print("pathBuilder: ",pathBuilder)
	return add_trailing_separator_in_path(pathBuilder)



def get_most_recently_modified_directory(inRootPath) :
	return max(glob.glob(os.path.join(inRootPath, '*/')), key=os.path.getmtime)



def get_most_recently_modified_file(inRootPath, includeSubFolders=True, filename=None ) :
	filesFullPaths = []
	for path, subdirs, files in os.walk(inRootPath) :
		for name in files :
			if not filename==None:
				if name.upper() == filename.upper():
					filesFullPaths.append(os.path.join(path, name))
			else:
				filesFullPaths.append(os.path.join(path, name))

	# print("filesFullPaths: ", filesFullPaths)

	return max(filesFullPaths, key=os.path.getctime)



def check_output_path_str(inPath, addDayTime=False, addSecs=False) :
	
	g.command_history.add('', os.path.basename(__file__),g.inspect.stack()[0][3], locals())

	if isinstance(inPath, (list, tuple)):
		joinedInPath = ''
		for pathI in inPath:
			pathIstr = str(pathI)			
			if len(pathIstr) > 0 :
				joinedInPath += remove_trailing_separator_in_path(pathIstr) + os.path.sep 
		inPath = str(joinedInPath)

	else:
		inPath = str(inPath)

	doubleSeparator = os.path.sep + os.path.sep
	inPath.replace(doubleSeparator, os.path.sep)
	inPath = remove_trailing_separator_in_path(inPath)

	if addDayTime==True:
		now = datetime.datetime.now()
		inPath += "__" + str(now.month).zfill(2) + "-" + str(now.day).zfill(2) + "__" + str(now.hour).zfill(2) + "h" + str(now.minute).zfill(2) + 'm'
		if addSecs:
			inPath += str(now.second).zfill(2) + 's'


	return inPath



def get_relative_sub_path_level(rootPath, rootPathAndSubPath) :
	rootPath = PurePath(rootPath)
	rootPathLevels = len(list(rootPath.parts))
	
	rootPathAndSubPath = PurePath(rootPathAndSubPath)
	rootPathAndSubPathLevels = len(list(rootPathAndSubPath.parts))

	return rootPathAndSubPathLevels - rootPathLevels



def get_files_parent_directory(fileNameAndPath) :
    self.attr['Root'] = str(purePath.parents[0]) + os.path.sep



def get_sub_path_items(rootPath, rootPathAndSubPath) :
	rootPath = PurePath(rootPath)
	rootPathParts = list(rootPath.parts)
	
	rootPathAndSubPath = PurePath(rootPathAndSubPath)
	rootPathAndSubPathParts = list(rootPathAndSubPath.parts)

	return remove_this_list_from_that_list(rootPathParts, rootPathAndSubPathParts)



def create_filename(itemsToConcat) :
	if isinstance(itemsToConcat, (list, tuple)) :

		joinedFilename = ''
		itemI = 0
		nItemsToJoin = len(itemsToConcat)

		for item in itemsToConcat :

			itemAsStr = str(item)
			itemI += 1

			if itemAsStr.upper() in ['MAT','TXT','PNG','MP4','BIN','CSV','CFG','JSON', 'XML', 'BDF']:
				itemAsStr = '.' + itemAsStr

			joinedFilename += (itemAsStr)
			if (itemI < (nItemsToJoin-1)) :
				joinedFilename += g.params['SEP']

	else:
		joinedFilename = str(itemsToConcat)

	return joinedFilename



def remove_trailing_separator_in_path(inputPath):
	if inputPath.endswith(os.path.sep) :
		return inputPath[:-1]
	else:
		return inputPath



def add_trailing_separator_in_path(inputPath):
	if not inputPath.endswith(os.path.sep) :
		return (inputPath + os.path.sep)
	else:
		return inputPath		



def set_os_permission_to_own(path):
	try :
		uid = pwd.getpwnam('nobody').pw_uid
		gid = grp.getgrnam('nogroup').gr_gid
		os.chown(path, uid, gid)

	except Exception as e :
		print(e)	

	try:
		os.chmod(path, 0o777)
	except Exception as e :
		print(e)


def load_json_file(fileAndPath) :

	with open(fileAndPath, "r") as jsonFile:
			return json.load(jsonFile)	


def check_if_string_in_file( strToCheck, fileNameAndPath ) :

	with open(fileNameAndPath) as myfile :
		readFile = myfile.read()
		if strToCheck in readFile :
			return True
		else:
			strToCheck = caps_under_to_cap_lower(strToCheck)
			if strToCheck in readFile:
				return True
			else:
				strToCheck = make_first_letter_lower_case(strToCheck)
				if strToCheck in readFile:
					return True
				else:
					if strToCheck.lower() in readFile:
						return True
					else:
						if strToCheck.upper() in readFile:
							return True						

	# print("Could not find: ", strToCheck, " in file.")

	return False
