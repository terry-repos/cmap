import sys
sys.path.append('..')
from global_imports import external_modules
from events.events_interpolated import EventsInterpolated

class EventsSet():

	def __init__(self, files, dataParams):
		self.ETs = [self.read_ETs_from_file(file.attr['NameAndPath'], dataParams) for file in files]
		print(self.ETs)

	def read_ETs_from_file(self, inputEventsFileAndPath, dataParams):
		print("File: ", inputEventsFileAndPath)
		eventStringMarker = dataParams['eventMarker']
		reachedData = False
		waveI = 0
		addedSomeWaveData = False

		with open(inputEventsFileAndPath,"r") as f:
			nLines = len(f.readlines())			


		with open(inputEventsFileAndPath,"r") as f:
			waveEventsSet = np.array([])
			tsvReader = csv.reader(f, delimiter='\t') 

			lineI = 0

			for row in tsvReader:
				lineI +=1

				print(lineI)

				try:
					cleanedRow = [item.replace("'   '","") for item in row]	
					cleanedRow = [item.replace(" ","") for item in cleanedRow]
					cleanedRow = list(filter(None, cleanedRow)) # fastest]
					cleanedRow = [item for item in cleanedRow if len(item) > 0]   # in case blank tabs have been picked up.

					if reachedData:
						# add wave to list if next wave reached or end of file reached

						if addedSomeWaveData:
							if len(cleanedRow)==0:
								print("Found an empty row, so stopping reading data for this wave.")
								reachedData = False

							elif (eventStringMarker.upper() in cleanedRow[0].upper()):
								reachedData = False
								print("Found the next wave, so stopping reading data for this wave.")



						if not reachedData:
							waveI += 1

							if 'ROT90' in inputEventsFileAndPath.upper():
								waveEvents = np.rot90(waveEvents)
							print(waveEvents.shape)

							print(waveEventsSet.shape)

							if waveEventsSet.size == 0:
								waveEventsSet=np.full(shape=(waveEvents.shape[0], waveEvents.shape[1], 1), fill_value=np.nan, dtype=float)
								waveEventsSet[:,:,0] = waveEvents
							else:
								waveEventsSet = np.dstack((waveEventsSet, waveEvents))
							
							waveEvents = np.array([])

						else:
							if waveEvents.size == 0:
								waveEvents = np.array(cleanedRow).astype(float)

							else:
								waveEvents = np.vstack((waveEvents, np.array(cleanedRow).astype(float)))	
							addedSomeWaveData = True

							if lineI==nLines:
								reachedData = False
								print("Reached the end of the file, so stopping reading data for this wave.")														
							

					if len(cleanedRow) > 0:
						if eventStringMarker.upper() in cleanedRow[0].upper():
							waveEvents = np.array([])	
							addedSomeWaveData = False

							reachedData = True


				except Exception as e:
					print(e)
					raise


		print(waveEventsSet.shape)
		return waveEventsSet



	def interpolate(self, interpParams) :
		self.interpedEvents = [WaveEventsInterpolated(ETs, interpParams) for ETs in self.ETs]



	def get_elementary_wave_properties(self, interpParams) :
		pass
		# self.interpedEvents = InterpolatedEventSet(self.ETs, interpParams)		





