import numpy as np
from global_imports import session_vars as g


def get_std_dev_from_median(chanSlice, scalar) :

	print(chanSlice)
	lowerLimit = np.median(g.dat[g.cur['DAT_ID']]['Samples'][chanSlice]) - \
				scalar * np.std(g.dat[g.cur['DAT_ID']]['Samples'][chanSlice])

	upperLimit = np.median(g.dat[g.cur['DAT_ID']]['Samples'][chanSlice]) + \
				scalar * np.std(g.dat[g.cur['DAT_ID']]['Samples'][chanSlice])

	return (lowerLimit, upperLimit)