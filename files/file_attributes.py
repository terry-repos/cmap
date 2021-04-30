import os
import numpy as np

from pathlib import Path, PurePath
import copy

class FileAttributes() :

    def __init__(self, fileNameAndPath, dataClass=None, ProcessedClass=None, dataType=None) :

        self.attr = {}
        self.attr['NameAndPath'] = fileNameAndPath
        if not dataClass == None :
            self.attr['DataClass'] = dataClass

        if not ProcessedClass == None :
            self.attr['Processed'] = ProcessedClass

        if not dataType == None :
            self.attr['dataType'] = dataType            

        purePath = PurePath(fileNameAndPath)
        purePathsList = list(purePath.parts)

        if len(purePathsList) > 0:

            self.attr['Name'] = purePathsList.pop()
            self.attr['Root'] = str(purePath.parents[0]) + os.path.sep
            fileNameSplit = self.attr['Name'].rsplit('.', 1)
            self.attr['Extension'] = fileNameSplit.pop().upper()
            self.attr['NameNoExt'] = copy.copy(fileNameSplit)[0]
            self.attr['NameAndPathNoExt'] = self.attr['Root'] + self.attr['NameNoExt']
            self.attr['FileSize'] = os.path.getsize(self.attr["NameAndPath"])

        else:
            
            self.attr['Name'] = ""
            self.attr['Root'] = ""
            self.attr['Extension'] = ""
            self.attr['NameNoExt'] = ""
            self.attr['NameAndPathNoExt'] = ""
            self.attr['FileSize'] = 0


    def is_valid(self, extensionsToCompare):
        
        if (extensionsToCompare[0] == "*.*") or (extensionsToCompare == "*.*") :
            return True
        if (self.attr['Extension'] in extensionsToCompare) :
            return True
        else:
            return False

