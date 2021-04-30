import sys
sys.path.append('..')
from global_imports import session_vars as g
from global_imports import external_modules

from importlib import import_module
import copy

from pipelines.paradigm import Paradigm
from pipelines.pipeline_steps import PipelineSteps
import steps 

from files.data_selection import DataSelection

from steps.plot_channels import PlotChannels
from steps.detect import *

from steps.update_samples_info import *
from steps.save_data import *

class AnalysisPipeline() :

	def __init__(self, paradigmFilePath, rootPyPath) :

		self.paradigmFilePath = paradigmFilePath
		self.rootPyPath = rootPyPath

		g.cur = {}


	def load_paradigm(self) :

		g.command_history.add('Passing ' +  self.paradigmFilePath + ' to ParadigmFile class.', self.__class__.__name__, g.inspect.stack()[0][3])
	
		self.paradigmFile = Paradigm( self.paradigmFilePath, self.rootPyPath )
		self.paradigmFile.create_extra_params()
		g.paradigmFile = self.paradigmFile.paradigmFile
		g.loadDataFile = self.paradigmFile.loadDataFile

		g.params = self.paradigmFile.params
		g.loadDataParams = self.paradigmFile.loadDataParams



	def build_pipeline(self) :

		g.command_history.add('Building pipeline... ', self.__class__.__name__,g.inspect.stack()[0][3])
		self.pipelineSteps = PipelineSteps(g.params)



	def prepare_files(self) :

		g.command_history.add('Preparing files... ', self.__class__.__name__,g.inspect.stack()[0][3])
		g.DataSelection = DataSelection(g.loadDataParams)

		for file in g.DataSelection.files :
			print(file.attr['Name'])



	def run_pipeline(self) :

		if g.params['PIPELINE_PROCESSING']['ANALYSIS_SEQUENCE'] == 'ONE_BY_ONE' :

			for linkedFiles in g.DataSelection.linkedFilesGroup :

				g.cur = {}
				g.cur['DAT_ID'] = linkedFiles['DAT_ID']
				g.cur['LINKED_FILES'] = copy.copy(linkedFiles)

				for pipeStep in self.pipelineSteps.steps :

					g.cur['STEP'] = copy.deepcopy(pipeStep)
					self.run_step()



	def run_step(self) :
		# print("g.cur['STEP']['PARAMS']['PER_CHANNEL']: ", g.cur['STEP']['PARAMS']['PER_CHANNEL'])
		if not g.cur['STEP']['PARAMS']['CHAN_RANGE'] :

			if g.cur['DAT_ID'] in g.dat.keys() :

				g.cur['STEP']['PARAMS']['CHAN_RANGE'] = g.dat[ g.cur[ 'DAT_ID' ] ][ 'ChanConfig' ][ 'CHANNEL_ORDER' ]


		if g.cur['STEP']['PARAMS']['SAMPLES_PROCESSING'] :

			if g.cur['STEP']['PARAMS']['PER_CHANNEL'] :

				for chan in g.cur['STEP']['PARAMS']['CHAN_RANGE'] : 
					if 'SLICE' in g.cur.keys() :
						del g.cur['SLICE']						

					g.cur['SLICE'] = np.s_[chan,::]
					# print("Current indices: ", g.cur['SLICE'])

					if g.cur['STEP']['PARAMS']['CHAN_RANGE'][0] == g.cur['SLICE'][0] :
						g.cur['IN_FIRST_CHANNEL'] = True

					else:
						g.cur['IN_FIRST_CHANNEL'] = False

					# if g.cur['STEP']['PARAMS']['CHAN_RANGE'][-1] == g.cur['SLICE'][0] :
					# 	g.cur['IN_LAST_CHANNEL'] = True

					# else:
					# 	g.cur['IN_LAST_CHANNEL'] = False						

					self.process_step()

			else:
				g.cur['SLICE'] = copy.copy( np.s_[::] )
				self.process_step()

		else:
			g.cur['SLICE'] = copy.copy( np.s_[::] )			
			self.process_step()

		update_samples_info()
		self.output_steps()		
		


	def process_step(self) :
		# print("g.cur['STEP']['STEP']: ", g.cur['STEP']['STEP'])
		stepModule = import_module('steps.' + g.cur['STEP']['STEP'])
		g.command_history.add(str(g.cur['STEP']), stepModule.__name__, g.cur['STEP']['STEP'], locals())		

		stepModule	
		stepFunc = stepModule.__name__ + '.' + g.cur['STEP']['STEP'] + '()'

		eval(stepFunc)



	def output_steps(self):

		g.command_history.add('', self.__class__.__name__, g.inspect.stack()[0][3], locals())

		if g.params['DEFAULT_OUTPUT_PARAMS_FOR_EACH_STEP']['OUTPUT']['PRINT_DATA_HIEARCHY_AFTER_EACH_STEP'] :
			print_keys_hierarchy( g.dat[ g.cur['DAT_ID'] ], ("Data hierarchy after " + g.cur['STEP']['STEP']) )

		if 'MATCH_EVENTS_WITH_REF' in g.cur['STEP']['PARAMS'].keys():
			
			detect( shouldMatchEvents=True )
			print_keys_hierarchy( g.dat[ g.cur['DAT_ID'] ], ("Data hierarchy after matching for step " + g.cur['STEP']['STEP']) )


		# print( "g.cur['STEP']['OUTPUT']['SAVE_DATA']: ", g.cur['STEP']['OUTPUT']['SAVE_DATA'], " for ",  g.cur['STEP']['STEP'] )

		if g.cur['STEP']['OUTPUT']['SAVE_DATA']['PERFORM_SAVE_DATA'] :
			if 'ONLY_ON_FINAL_STEP' in g.cur['STEP']['OUTPUT']['SAVE_DATA'].keys() :
				if 'IS_FINAL_STEP' in g.cur['STEP'].keys() :
					save_data()
			else :
				save_data()	

		if g.cur['STEP']['OUTPUT']['PLOT_CHANS']['PERFORM_PLOT_CHANS'] :
			PlotChannels()








