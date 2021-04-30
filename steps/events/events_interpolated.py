import sys
sys.path.append('..')
import pyutils.global_imports
from pyutils.np_arr_flipping import *
from algorithmic.calc_statistics import *
from interpolation.interp_indexing import *
from interpolation.interp_wavefront import *
import numpy as np


class EventsInterpolated():

	def __init__(self, gridData, interpParams):

		self.gridData = gridData
		self.interpParams = interpParams

		self.resX = interpParams['resX']
		self.resXrange = range(2, self.resX)

		self.nInterpedWaves = gridData.shape[2]

		self.waveRange = range(0, self.nInterpedWaves)

		self.interpShape = calc_interp_shape(self.resX, gridData.shape)

		self.interpolate_main()



	def interpolate_main(self):
		print("interpshape: ", self.interpShape)
		self.interpWaves = np.full(shape=self.interpShape, fill_value=np.nan, dtype=float16)  
		print(self.interpWaves.nbytes) 

		for self.waveI in self.waveRange:
			self.currX = 1     
			self.map_original_data_to_interp()

			# Now upsample
			self.upsample()


	def map_original_data_to_interp(self):
		
		beg, jump, cEnd, rEnd = get_interp_indexing(self.currX, self.resX, self.interpShape)

		self.interpWaves[beg:rEnd:jump, beg:cEnd:jump, self.waveI] = self.gridData[:, :, self.waveI]



	def upsample(self) :

		for self.currX in self.resXrange:

			# First interpolate missing recording points. Try twice.
			if self.currX == 1 :
				for i in range(0,2) :
					self.interpolate_missing_vals()

			self.interpolate_missing_vals()



	def interpolate_missing_vals(self) :
		
		rs, cs = self.get_interp_coords()
		
		while True:

			rs, cs = self.order_NaNs_to_traverse_by_valid_neighbours(rs, cs)
			
			if len(rs)==0: #returned no valid elements
				break

			interpCoordI = 0

			for r, c in zip(rs, cs):

				if self.interpParams['interpMethods'].upper() == 'WAVEFRONT_ORIENTATION':

					linearNbrsCoords = get_linear_neighbour_pairs_coords(r, c)
					interpVal = find_min_diff_from_neighbours(linearNbrsCoords)

				if ~np.isnan(interpVal):
					self.interpWaves[r, c, self.waveI] = interpVal
				else:
					del rs[interpCoordI]
					del cs[interpCoordI]

				interpCoordI += 1



	def order_NaNs_to_traverse_by_valid_neighbours(self, rs, cs):

		validNeighbours = self.get_num_valid_neighbours(rs, cs)
		maxNofValidNeighbours = np.nanmax(validNeighbours)
		ix = []

		# iterate until it has found indices where interpolation is possible
		while True:
			if (maxNofValidNeighbours <= minValidNeighbours):
				break        

			ix = np.where(((np.isnan(inGridData))==nanValue) & (ValidNeighbours == maxNofValidNeighbours))

			if (ix[0].size > 0):
				break
			else:
				maxNofValidNeighbours -= 1

		try:
			rs = ix[0]
			cs = ix[1]

		except Exception as e:
			print(e)
			rs = []
			cs = []

		return rs, cs



	def get_interp_coords(self):
		beg, jump, cEnd, rEnd = get_interp_indexing(self.currX, self.resX, self.interpShape)
		return np.where(np.isnan((self.interpWaves[beg:rEnd:jump, beg:cEnd:jump, self.waveI])))




	def get_num_valid_neighbours(self, rs, cs): #including self!

		## Convert ATs to 1s and nans to 0s to enable counting of "Valid" neighbours
		nansAsZerosATsAsOnes = make_nans_zero_ATs_one(self.interpWaves[[rs], [cs], self.waveI])

		nValidNeighbrsArr = signal.convolve2d(nansAsZerosATsAsOnes, np.ones((3, 3)), mode="same").astype(dtype="int")
	 
		return nValidNeighbrsArr            