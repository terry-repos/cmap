import sys
import os
import copy 
import inspect

import numpy as np
import math

from scipy import signal

sys.path.append('..')
from global_imports import external_modules
from global_imports import session_vars as g
from global_imports.simplify_vars import *

from pyutils.dict_utils import *


from algorithms.stats import *

from pyutils.io_utils import *

from steps.subset_samples import *
from steps.classification_helpers import *


from hierarchies.hierarchies_events import *
from hierarchies.indexing_events import *

from hierarchies.indexing_wins import *

import tensorflow as tf
from tensorflow.contrib import predictor



# Notes:
# =======
# If you are initializing a tensor with a constant value, you can pass a NumPy array to the tf.constant() op, and the result will be a TensorFlow tensor with that value.


class operate_nn() :

	def __init__( self ) :

		g.command_history.add('', os.path.basename(__file__),g.inspect.stack()[0][3], locals())
		self.nnParams = make_params_readable()
		self.ID = get_curr_ID()
		self.nChans = get_nChans()
		print("In operate nn for ", self.ID)

		self.init_data_structure()

		self.dataDirectories = copy.copy( self.get_base_directories( ) )

		self.paramsVariations = build_nn_param_variations( self.nnParams )

		for self.currentParams in self.paramsVariations :

			self.currentParams = get_classification_params( self.ID, self.currentParams )

			self.hierarchies = get_classification_data_hierarchies( self.ID, self.currentParams, self.nnParams, self.dataDirectories )
			self.get_or_make_input_wins( ) # instantiates self.allInputDataIndices, self.inputWins, self.inputWinLbls 

			if 'TRAIN' in self.nnParams['METHODS'].upper( ) :		
				self.train_nn( )

			elif 'PREDICT' in self.nnParams['METHODS'].upper( ) :
				self.init_predict_data_structure( )
				self.predict_with_nn( )

			self.tfSession.close()
			if hasattr( self, 'tfSession' ) :
				del self.tfSession			




	def get_base_directories( self ) :

		dataDetails = {}
		dataDetails[ 'nnRootDir' ] = create_dir( [ g.loadDataParams['DEFAULT_DATA_ROOT'], "NeuralNets" ] )

		return dataDetails



	def init_data_structure( self ) :

		# if not 'EVENT_MARKER_PARAMS' in g.cur['STEP']['OUTPUT']['PLOT_CHANS']:
		# 	g.cur['STEP']['OUTPUT']['PLOT_CHANS']['EVENT_MARKER_PARAMS'] = {}

		# if not 'EVENT_HIERARCHIES_TO_PLOT' in g.cur['STEP']['OUTPUT']['PLOT_CHANS']['EVENT_MARKER_PARAMS'] :
		# 	g.cur['STEP']['OUTPUT']['PLOT_CHANS']['EVENT_MARKER_PARAMS']['EVENT_HIERARCHIES_TO_PLOT'] = []


		if 'cvEvaluation' not in g.dat[ self.ID ].keys() :
			g.dat[ self.ID ][ 'cvEvaluation' ] = {}

		if 'cvAvg' not in g.dat[ self.ID ].keys() :
			g.dat[ self.ID ][ 'cvAvg' ] = {}		

		if 'TFevaluation' not in g.dat[ self.ID ].keys() :
			g.dat[ self.ID ][ 'TFevaluation' ] = {}		




	def init_predict_data_structure( self ) :

		if 'nnPredictions' not in g.dat[ self.ID ].keys() :
			print(self.ID, ' nnPred empty!')
			g.dat[ self.ID ][ 'nnPredictions' ] = {}

		else :
			print( "found nnPred for: ", self.ID )

		for lblHier in self.hierarchies['labelHierarchies'] :
			g.cur[ 'STEP' ] = copy.deepcopy( alter_step_params_according_to_wins( self.currentParams, lblHier, g.cur[ 'STEP' ] ) )

			predHier = list_as_str(lblHier,"")
			print("pred hier: ", predHier)
			g.dat[ self.ID ][ 'nnPredictions' ][ predHier ] = 0


	
	def train_nn( self ) :
		self.dataDirectories['nnFullDir'] = create_dir([ self.dataDirectories['nnRootDir'], self.currentParams['nnName'] ])



		self.folds = get_folds( self.currentParams[ 'nFolds' ], self.inputWinLbls )
		foldI = 0

		for trainIndices, testIndices in self.folds :
			self.tfSession = tf.Session()

			foldI += 1

			self.train( trainIndices, testIndices, foldI )
			del self.tfSession

		self.tfSession = tf.Session()

		self.train( self.allInputwinIndices )
		self.save_model( )





	def predict_with_nn(self) :
		self.tfSession = tf.Session()



		if 'TRAIN' in self.currentParams[ 'METHODS' ].upper( ) :

			self.predictWithinSession = True
			self.predict_batch_of_wins()

		else :

			self.predict_load_init( )
			self.predictWithinSession = False


		predictions = np.zeros(shape=(2))
		eventDictTitle = self.ID + " events" 

		for inputWin, inputEvntWnMap in zip(self.inputWins, self.inputEvntWnMap) :
			# print("inputWin.shape: ", inputWin.shape, " inputEvntWnMap.shape: ", inputEvntWnMap.shape)
			prediction = self.predict_data_from_loaded_model( inputWin )
			predictions[prediction] += 1
			predHier = copy.deepcopy( self.hierarchies[ 'labelHierarchies' ][ prediction ] )
			predHierAsStr = list_as_str(predHier, "")

			chanIndex, unlabelledEventIndex = inputEvntWnMap[0], inputEvntWnMap[1]

			g.dat[ self.ID ][ 'nnPredictions' ][ predHierAsStr ] += 1
			# print("g.dat[ self.ID ][ 'nnPredictions' ]: ", g.dat[ self.ID ][ 'nnPredictions' ])
			g.dat[ self.ID ][ 'Events' ] = insert_index_into_dict_with_hier_list( g.dat[ self.ID ][ 'Events' ], predHier, chanIndex, unlabelledEventIndex, get_nChans() )	

		print(self.ID, " ", g.dat[ self.ID ][ 'nnPredictions' ])

		print("predictions: ", predictions)




	def train( self, trainIndices, testIndices=np.array([]), foldI=None ) :

		self.trainingFunc = self.build_training_func( trainIndices )
		self.featureCols = [ tf.feature_column.numeric_column("x", shape=[ self.currentParams['featureSize'] ]) ]

		self.nn = self.build_nn_config( )

		self.nn.train( input_fn=self.trainingFunc, steps=self.currentParams[ 'NUM_TRAINING_STEPS' ] )

		if self.currentParams[ 'nnName' ] not in g.dat[ self.ID ]['cvEvaluation'].keys() :
			g.dat[ self.ID ][ 'cvEvaluation' ][ self.currentParams[ 'nnName' ] ] = []

		if self.currentParams[ 'nnName' ] not in g.dat[ self.ID ]['cvAvg'].keys() :
			g.dat[ self.ID ][ 'cvAvg' ][ self.currentParams[ 'nnName' ] ] = { }

		if self.currentParams[ 'nnName' ] not in g.dat[ self.ID ]['TFevaluation'].keys() :
			g.dat[ self.ID ][ 'TFevaluation' ][ self.currentParams[ 'nnName' ] ] = { }

		if testIndices.any() :
			self.evaluate_predictions( testIndices, foldI )

		else :
			self.testFunc = self.build_test_func()

			self.evaluate = self.nn.evaluate( input_fn = self.testFunc )
			self.evaluateCompatible = make_dict_serializable( self.evaluate )			
			print("self.evaluateCompatible: ", self.evaluateCompatible)
			g.dat[ self.ID ][ 'TFevaluation' ][ self.currentParams[ 'nnName' ] ] = self.evaluateCompatible



	def get_or_make_input_wins( self ) :

		self.inputWins, self.inputWinLbls = np.array( [] ), np.array( [] )
		lblI = 0

		for labelsHier in  self.hierarchies['inputHiers'] :
			essntlHierarchies, winLblsHier = get_win_hierarchies( self.currentParams, labelsHier )

			wins = get_vals_from_dict_with_this_hierarchy( g.dat[ self.ID ][ 'Windows' ] , winLblsHier )


			g.cur['STEP'] = copy.deepcopy( alter_step_params_according_to_wins( self.currentParams, labelsHier, g.cur[ 'STEP' ] ) )

			if len( wins ) == 0 :

				print( "wins were empty, attempting own windowing for ... ", winLblsHier )

				subset_samples( )
				wins = get_vals_from_dict_with_this_hierarchy( g.dat[ self.ID ][ 'Windows' ] , winLblsHier )


			if not len( wins ) == 0 :

				evntWnMap = get_vals_from_dict_with_this_hierarchy( g.dat[ self.ID ][ 'EvntWnMaps' ] , winLblsHier )

				if self.inputWins.shape[0] == 0 :

					self.inputWins = np.copy(wins)
					self.inputEvntWnMap = np.copy( evntWnMap )					
					self.inputWinLbls = np.full( shape=( wins.shape[0] ), fill_value=lblI)

				else :

					self.inputWins = np.vstack(( self.inputWins, wins ))
					self.inputEvntWnMap = np.vstack(( self.inputEvntWnMap, evntWnMap ))
					self.inputWinLbls = np.hstack(( self.inputWinLbls, np.full( shape=( wins.shape[0] ), fill_value=lblI) ))


			else :
				print(" Could not find wins with ", list_as_str( winLblsHier ), " criteria. ")

			lblI += 1

		print("self.hierarchies['inputHiers']: ", self.hierarchies['inputHiers'] )
		print("inputWins.shape: ", self.inputWins.shape, " wins.shape: ", wins.shape)


		self.allInputwinIndices = np.linspace( 0, self.inputWins.shape[0], 1, dtype=int )



	def save_model( self ) :

		self.featureSpec = tf.feature_column.make_parse_example_spec(self.featureCols)
		self.exportInputFn = tf.estimator.export.build_parsing_serving_input_receiver_fn( self.featureSpec )
		self.nn.export_savedmodel( self.dataDirectories['nnFullDir'], self.exportInputFn )	



	def evaluate_predictions( self, testIndices, foldI ) :

		trueLabels = self.inputWinLbls[testIndices]

		predictedLabels = self.predict_batch_of_wins(testIndices)
		
		self.create_prediction_events(predictedLabels, testIndices)
	
		cvRawResults, cvSummary = compute_classification_results( trueLabels,  predictedLabels, foldI )

		g.dat[ self.ID ][ 'cvEvaluation' ][ self.currentParams[ 'nnName' ] ].append( copy.deepcopy(cvSummary) )
		# print_keys_hierarchy(g.dat[ self.ID ]['cvEvaluation'], "g.dat[ self.ID ]['cvEvaluation']")
		print("g.dat[ self.ID ]['cvEvaluation']: ", g.dat[ self.ID ]['cvEvaluation'])

		if foldI == self.currentParams[ 'nFolds' ]  :
			g.dat[ self.ID ][ 'cvAvg' ][ self.currentParams[ 'nnName' ] ] = compute_classification_avg( copy.deepcopy(g.dat[ self.ID ][ 'cvEvaluation' ][ self.currentParams[ 'nnName' ] ]) )



	def create_prediction_events(self, predictions, testIndices) :

		evntWnMapSubset = self.inputEvntWnMap[ testIndices, : ]
		# print("predictions.shape: ", predictions.shape, " testIndices.shape: ", testIndices.shape, "self.inputEvntWnMap.shape: ", self.inputEvntWnMap.shape, "evntWnMapSubset.shape: ", evntWnMapSubset.shape)

		predI = 0
		for prediction, evntWnMap in zip( predictions, evntWnMapSubset ) :
			chanIndex, unlabelledEventIndex = evntWnMap[0], evntWnMap[1]
			predHier = copy.deepcopy( self.hierarchies[ 'labelHierarchies' ][ prediction ] )			

			# print("g.dat[ self.ID ][ 'nnPredictions' ]: ", g.dat[ self.ID ][ 'nnPredictions' ])
			print(str(predI), " predHier: ", predHier, " evntWnMap: ", evntWnMap)
			g.dat[ self.ID ][ 'Events' ] = insert_index_into_dict_with_hier_list( g.dat[ self.ID ][ 'Events' ], predHier, chanIndex, unlabelledEventIndex, get_nChans() )	
			predI+=1



	def predict_batch_of_wins( self, indices=None ) :
		self.predictFuncWithinSession = self.build_predict_func( self.inputWins[indices])

		predGenerator = list( self.nn.predict( input_fn=self.predictFuncWithinSession ) )

		return np.array( [ p['class_ids'][0] for p in predGenerator ] ).astype( int )
		


	def predict_load_init( self ) :

		self.get_nn_dir()
		self.load_saved_session()

		tf.saved_model.loader.load( self.tfSession, [ tf.saved_model.tag_constants.SERVING ], self.modelLoadDir ) 

		self.inputHolder = self.tfSession.graph.get_operation_by_name("input_example_tensor").outputs[0]
	
		self.predsOperation = self.tfSession.graph.get_operation_by_name("dnn/head/predictions/probabilities").outputs[0]

		self.predDict = {}		



	def predict_data_from_loaded_model(self, unlabelledWin) :

		inputAsTFrecord = tf.train.Feature( float_list=tf.train.FloatList( value=unlabelledWin ))

		example = tf.train.Example( features=tf.train.Features( feature = {"x" : inputAsTFrecord } ) )

		return int( self.tfSession.run([ self.predsOperation ], { self.inputHolder : [ example.SerializeToString() ]} )[0][0][0] )



	def predict_data_within_training_session(self, unlabelledwin) :

		self.predictFuncWithinSession = self.build_predict_func([[ unlabelledwin ]])

		predGenerator = list( self.nn.predict( input_fn=self.predictFuncWithinSession ))

		return predGenerator[0]['class_ids'][0]



	def get_nn_name(self) :

		lblI = 0

		for lbl in self.labels :
			if not lblI==self.nLbls:
				nnName+= 'v'

		return nnName



	def build_nn_config(self) :

		# print("HIDDEN_UNITS: ", self.currentParams['HIDDEN_UNITS'], "n_classes: ", self.hierarchies['nLbls'], " dir: ", self.dataDirectories['nnFullDir'] )
		return tf.estimator.DNNClassifier( feature_columns=self.featureCols,
			hidden_units = self.currentParams['HIDDEN_UNITS'],
			n_classes = self.hierarchies['nLbls'],
			model_dir = self.dataDirectories['nnFullDir'] )



	def get_nn_dir(self) :

		if 'NN_MODEL_TO_LOAD' in self.currentParams.keys() :
			if len(self.currentParams['NN_MODEL_TO_LOAD']) > 0 :
				nnDir = self.currentParams['NN_MODEL_TO_LOAD']
				self.dataDirectories['nnFullDir'] = create_dir([self.dataDirectories['nnRootDir'], nnDir])
				self.dataDirectories['nnFullDir'] = get_most_recently_modified_directory(self.dataDirectories['nnFullDir'])



		print("self.dataDirectories['nnFullDir']: ", self.dataDirectories['nnFullDir'])
		modelFile = get_most_recently_modified_file( self.dataDirectories['nnFullDir'], includeSubFolders=True, filename="saved_model.pb" )
		# print("modelFile: ", modelFile)
		self.modelLoadDir = remove_childmost_level_from_path(modelFile)



	def build_summary_hook( self ) :

		return tf.train.SummarySaverHook(
			10,
			output_dir=self.dataDirectories['nnTrainingDir'],
			summary_op=tf.summary.scalar("accuracy", self.evaluateCompatible['accuracy']) )



	def load_saved_session(self) :

		self.configProto = tf.ConfigProto(allow_soft_placement=True)
		self.tfSession = tf.Session(config=self.configProto)



	def build_training_func( self, trainingIndices ) :

		return tf.estimator.inputs.numpy_input_fn(
			x={"x": np.array( self.inputWins[ trainingIndices ])},
			y=np.array( self.inputWinLbls[ trainingIndices ] ),
			num_epochs=None,
			shuffle=True )		



	def build_test_func(self) :

		return tf.estimator.inputs.numpy_input_fn(
			x={"x": np.array( self.inputWins )},
			y=np.array(self.inputWinLbls),
			num_epochs=1,
			shuffle=False)



	def build_predict_func(self, unlabelledwin) :

		return tf.estimator.inputs.numpy_input_fn(
			x={"x": np.array(unlabelledwin)},
			num_epochs=1,
			shuffle=False)		

