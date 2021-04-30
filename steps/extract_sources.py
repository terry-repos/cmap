import sys
import os
import copy 

import numpy as np

from scipy import signal
import sklearn
from sklearn import preprocessing

sys.path.append('..')
from global_imports import external_modules
from global_imports import session_vars as g
from global_imports.simplify_vars import *

from algorithms.stats import *
from algorithms.learning import *


def extract_sources() :

	g.command_history.add('', os.path.basename(__file__),g.inspect.stack()[0][3], locals())

	extractParams = make_params_readable()
	ID = get_curr_ID()
	sl = get_curr_sl()
	
	if extractParams['METHODS'] == 'FAST_ICA' :
		sources = run_fast_ica(g.dat[ID]['Samples'][sl], extractParams)

	g.dat[ID]['Samples'][sl] = np.copy(sources)

