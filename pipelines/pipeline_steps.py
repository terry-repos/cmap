import sys
import copy
sys.path.append('..')
from global_imports import external_modules
from global_imports import session_vars as g

from steps import *
from pyutils.dict_utils import *
from pyutils.list_utils import *


class PipelineSteps():

	def __init__(self, params):

		self.params = params
		self.create_pipeline_steps()

		print(self.params)



	def create_pipeline_steps(self) :

		g.command_history.add('Creating pipeline steps.',self.__class__.__name__, g.inspect.stack()[0][3])

		self.defaultParams = copy.deepcopy(self.params['DEFAULT_OUTPUT_PARAMS_FOR_EACH_STEP']['OUTPUT'])
		pipelineParams = copy.deepcopy(self.params['PIPELINE_STEPS'])

		self.steps = []
		self.pipeI = 0

		g.paradigmUsed = {}
		g.paradigmUsed['PIPELINE_STEPS'] = {}		
		self.nPipelineSteps = len(pipelineParams)

		for stepParams in pipelineParams :

			self.add_pipeline_steps(stepParams)

		

	def add_pipeline_steps(self, pipelineStepParams, possiblePipeName=None) :

		pipelineStepParamsItems = pipelineStepParams.items()
		print( "pipelineStepParamsItems: ", pipelineStepParamsItems )

		for pipeKey, pipeParams in pipelineStepParamsItems :
			print( "pipeKey: ", pipeKey, "pipeParams: ", pipeParams )

			if 'PERFORM_THIS_STEP' in pipeParams:
				if pipeParams['PERFORM_THIS_STEP'] :
					print("Adding step ", pipeKey, " to pipeline steps.")
					self.pipeI += 1
					newFormattedStep = dict()
					newFormattedStep = self.format_step(pipeKey, dict(pipeParams))
					self.steps.append(newFormattedStep)
					g.paradigmUsed[ 'PIPELINE_STEPS' ][ pipeKey ] = pipeParams
			
			# If child is a dict, then check if it contains pipeline steps there 
			else:
				if isinstance(pipeParams, dict) :
					self.add_pipeline_steps(pipeParams, pipeKey)



	def format_step(self, step, inParams):
		
		if 'formattedStep' in locals():
			del formattedStep

		formattedStep = dict()
		formattedStep['NUM'] = self.pipeI

		formattedStep['STEP'] = step.lower()
		formattedStep['PARAMS'] = dict(inParams['PARAMS'])

		if self.pipeI == self.nPipelineSteps :
			formattedStep['PARAMS']['IS_FINAL_STEP'] = True

		if not 'METHODS' in inParams.keys() :
			formattedStep['PARAMS']['METHODS'] = copy.copy(formattedStep['STEP'])

		else:
			formattedStep['PARAMS']['METHODS'] = copy.copy(inParams['METHODS'])


		if not 'PER_CHANNEL' in formattedStep['PARAMS'].keys() :
			formattedStep['PARAMS']['PER_CHANNEL'] = True


		if not 'CHAN_RANGE' in formattedStep['PARAMS'].keys() :
			formattedStep['PARAMS']['CHAN_RANGE'] = False		


		if 'OUTPUT_EVENTS_HIERARCHY' in	formattedStep['PARAMS'].keys() or (formattedStep['STEP'].upper() in ['DETECT']) :

			# Check if should perform matching
			if 'MATCH_EVENTS_WITH_REF' in formattedStep[ 'PARAMS' ].keys() :
				if formattedStep['PARAMS']['MATCH_EVENTS_WITH_REF'] :
					if not 'REFERENCE_EVENTS_HIERARCHIES' in formattedStep[ 'PARAMS' ].keys() :
						formattedStep[ 'PARAMS' ][ 'REFERENCE_EVENTS_HIERARCHIES' ] = copy.deepcopy( self.defaultParams['DEFAULT_REFERENCE_EVENTS_HIERARCHIES'] )
				else :
					formattedStep[ 'PARAMS' ].pop('REFERENCE_EVENTS_HIERARCHIES', None)

			if not 'MATCH_EVENTS_WITH_REF' in formattedStep[ 'PARAMS' ].keys() :
				if self.defaultParams[ 'DEFAULT_REFERENCE_EVENTS_HIERARCHIES' ]:
					formattedStep[ 'PARAMS' ][ 'MATCH_EVENTS_WITH_REF' ] = copy.deepcopy( self.defaultParams[ 'MATCH_EVENTS_WITH_REF_AFTER_EACH_STEP' ] )
					if not 'REFERENCE_EVENTS_HIERARCHIES' in formattedStep[ 'PARAMS' ].keys() :
						formattedStep[ 'PARAMS' ][ 'REFERENCE_EVENTS_HIERARCHIES' ] = copy.deepcopy( self.defaultParams[ 'DEFAULT_REFERENCE_EVENTS_HIERARCHIES' ] )


			# print("ormattedStep['PARAMS']: ", formattedStep['PARAMS'])
			# if step is marked as a param
			if 'OUTPUT_EVENTS_HIERARCHY' not in	formattedStep['PARAMS'].keys():
	 			formattedStep['PARAMS']['OUTPUT_EVENTS_HIERARCHY'] = [ ]

			if step in formattedStep['PARAMS'].keys() :
				print("step in param keys!")
				if isinstance(formattedStep['PARAMS'][step], list) :
					insertProp = list_as_str(formattedStep['PARAMS'][step])

				else:
					insertProp = str(formattedStep['PARAMS'][step])

				if list_contains_list(formattedStep['PARAMS']['OUTPUT_EVENTS_HIERARCHY']) :
					nListsInList = len(formattedStep['PARAMS']['OUTPUT_EVENTS_HIERARCHY'])

					for listsI in range(0,(nListsInList-1)) :
						formattedStep['PARAMS']['OUTPUT_EVENTS_HIERARCHY'][listsI].insert(1, caps_under_to_cap_lower(insertProp))
				else:
					formattedStep['PARAMS']['OUTPUT_EVENTS_HIERARCHY'].insert(1, caps_under_to_cap_lower(insertProp))


			if list_contains_list(formattedStep['PARAMS']['OUTPUT_EVENTS_HIERARCHY']) :
				nListsInList = len( formattedStep['PARAMS']['OUTPUT_EVENTS_HIERARCHY'] )
				print("list contains list!")

				# if step is marked as a param 
				for listsI in range( 0, (nListsInList) ) :
					formattedStep['PARAMS']['OUTPUT_EVENTS_HIERARCHY'][listsI].insert(0, caps_under_to_cap_lower( formattedStep['STEP'] ))					
					formattedStep['PARAMS']['OUTPUT_EVENTS_HIERARCHY'][listsI].insert(1, caps_under_to_cap_lower( formattedStep['PARAMS']['METHODS'] ))

			else :
				formattedStep['PARAMS']['OUTPUT_EVENTS_HIERARCHY'].insert(0, caps_under_to_cap_lower( formattedStep['STEP'] ))					
				formattedStep['PARAMS']['OUTPUT_EVENTS_HIERARCHY'].insert(1, caps_under_to_cap_lower(formattedStep['PARAMS']['METHODS']))



		if formattedStep['PARAMS']['METHODS'].upper() in ['WINDOW_AT_EVENTS', 'LOAD_DATA' ] :

			formattedStep['PARAMS']['SAMPLES_PROCESSING'] = True
			formattedStep['PARAMS']['PER_CHANNEL'] = False


		if not 'SAMPLES_PROCESSING' in formattedStep['PARAMS'].keys() :

			if formattedStep['STEP'].upper() in ['CLEAN_SAMPLES','SUBSET_SAMPLES', 'DETECT'] :
				formattedStep['PARAMS']['SAMPLES_PROCESSING'] = True

			else :
				formattedStep['PARAMS']['SAMPLES_PROCESSING'] = False

	
 
		formattedStep['OUTPUT'] = copy.copy(self.generate_step_output_params(dict(inParams)))
		formattedStep['OUTPUT']['DIR'] = str(formattedStep['NUM']).zfill(2) + g.params['SEP'] + formattedStep['PARAMS']['METHODS'].upper()

		if 'SUB_METHOD' in formattedStep['PARAMS'].keys() :
			formattedStep['OUTPUT']['DIR'] +=  g.params['SEP'] + formattedStep['PARAMS']['SUB_METHOD']
		# print("formattedStep: ", formattedStep)

		return formattedStep



	def generate_step_output_params(self, stepParams) :

		newOutputParams = dict(g.params['DEFAULT_OUTPUT_PARAMS_FOR_EACH_STEP']['OUTPUT'])
		copyStepParams = copy.copy(stepParams)

		if 'SAVE_DATA' in copyStepParams :
			saveDataParams = {}
			saveDataParams['SAVE_DATA'] = dict(copyStepParams['SAVE_DATA'])
			newOutputParams = copy.copy(merge_dicts( saveDataParams, dict(newOutputParams) ))
			newOutputParams = dict_collections_update( newOutputParams, dict(saveDataParams) )

		if 'PLOT_CHANS' in copyStepParams :
			plotChansParams = {}
			plotChansParams['PLOT_CHANS'] = dict(copyStepParams['PLOT_CHANS'])
			newOutputParams = copy.copy(merge_dicts( plotChansParams, dict(newOutputParams) ))	
			newOutputParams = dict_collections_update( newOutputParams, dict(plotChansParams) )

			if 'EVENT_HIERARCHIES_TO_PLOT' not in newOutputParams[ 'PLOT_CHANS' ][ 'EVENT_MARKER_PARAMS' ].keys() :
				newOutputParams['PLOT_CHANS']['EVENT_MARKER_PARAMS']['EVENT_HIERARCHIES_TO_PLOT'] = []


		print("stepParams: ", stepParams)
		print("newOutputParams: ", newOutputParams)

		return newOutputParams








