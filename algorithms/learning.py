import sys
import numpy as np

from sklearn.decomposition import FastICA, PCA



def run_fast_ica(samples, params):
	print("Beginning Fast ICA")
	if not params['NUM_COMPONENTS'] == None:
		params['NUM_COMPONENTS'] = nChans
	samples = np.rot90(samples)
	print(samples.shape)
	ica = FastICA(n_components=params['NUM_COMPONENTS'], max_iter=params['MAX_ITER'], tol=params['TOLERANCE'])
	print("ica.get_params(): ", ica.get_params())
	print("type(ica): ", type(ica))
	S_ = ica.fit_transform(samples)  # Reconstruct signals
	print("S_", S_.shape)	
	A_ = ica.mixing_  # Get estimated mixing matrix
	print("A_", A_.shape)	

	combinedSignals = np.dot(S_, A_.T)
	print("Combined signals shape: ", combinedSignals.shape)

	print("ica.mean_ ", ica.mean_.shape)
	combinedSignalsICAmean = combinedSignals + ica.mean_
	print("combined + mean: ",  combinedSignalsICAmean)
	return np.rot90(S_)
