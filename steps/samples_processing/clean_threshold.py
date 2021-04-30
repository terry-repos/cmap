# # -*- coding: utf-8 -*-
# """
# Author : Shameer Sathar
# """
# from __future__ import division # Correct Python 2 division

# import numpy as np
# import scipy.io as sio

# import config_global as sp

# def warn(*args, **kwargs):
# 	pass
# import warnings
# warnings.warn = warn

# import sklearn


# def norm_first_clip(inData, clipLimits) :

# 	outData = np.clip(inData, clipLimits[0], clipLimits[1])

# 	return outData


# def norm_second_clip(secData, clipLims) : 

# 	#Shameer's secret sauce
# 	normedData = np.copy(secData)

# 	minLocs = np.where(secData < clipLims[0])
# 	maxLocs = np.where(secData > clipLims[1])

# 	# normedData[ = clipLims
# 	normedData[minLocs] = clipLims[0]
# 	normedData[maxLocs] = clipLims[1]


# 	# print("Hello: ")
# 	secData = (secData - np.min(secData)) / (np.max(secData) - np.min(secData))

# 	return secData


# def get_dev_with_median(inData):

# 	lowerLimit = np.median(inData) - 4 * np.std(inData);
# 	upperLimit = np.median(inData) + 4 * np.std(inData);

# 	return (lowerLimit, upperLimit)


# def normalise_chan_data(chanData) :
# 	nChans = chanData.shape[0]
# 	normalisedData = np.copy(chanData)

# 	# Clip between -2, 2 mv and perform Shameers normalisation
# 	clipLimits = (-2000, 2000)
# 	clippedData = norm_first_clip(chanData, clipLimits)

# 	for chan in range(0,nChans) :

# 		chanData = clippedData[chan,:]
# 		clipLimits = get_dev_with_median(chanData)
# 		chanDataSecondClip = norm_second_clip(chanData, clipLimits)
# 		normalisedData[chan, :] = sklearn.preprocessing.MinMaxScaler().fit_transform(chanDataSecondClip)
# 		# Classic normalisation [0,1]

# 	return normalisedData


# def indices_to_timestamps(indices):

# 	return indices * timeBetweenSamples



# def preprocess(chanData) :

# 	return normalise_chan_data(chanData)



# def get_extreme_chan_and_val(inData) :

# 	maxVal, minVal = np.max(inData), np.min(inData)
# 	maxChan, minChan = np.where(inData==maxVal)[0][0], np.where(inData==minVal)[0][0]

# 	#print("MaxVal: ", maxVal, "MaxChan", maxChan)
# 	#print("MinVal: ", minVal, "MinChan", minChan)

# 	if (abs(maxVal) > abs(minVal)) :
# 		return maxChan, inData[maxChan, :], maxVal 

# 	else :
# 		return minChan, inData[minChan, :], minVal 



# def find_pacing_threshold(inData) :

# 	extremeChanNum, extremeChanValues, extremeVal = get_extreme_chan_and_val(inData)

# 	percentileNum = 9; # for the whisker ends

# 	# whiskerEdges = (np.mean(extremeChanValues) - np.std(extremeChanValues), np.mean(extremeChanValues) + np.std(extremeChanValues))

# 	whiskerEdges = np.percentile(extremeChanValues, [percentileNum, 100-percentileNum])

# 	if (extremeVal < 0) : 
# 		quartile = np.min(whiskerEdges)
# 		outliers = extremeChanValues[np.where(extremeChanValues < quartile)]

# 	else:
# 		quartile = np.max(whiskerEdges)
# 		outliers = extremeChanValues[np.where(extremeChanValues > quartile)]

# 	thresholdVal = np.mean(outliers)

# 	#print("WhiskerEdges: ", whiskerEdges)
# 	#print("Extreme val: ", extremeVal)
# 	#print("Quartile: ", quartile)
# 	#print("ThresholdVal: ", thresholdVal)

# 	return extremeChanValues, thresholdVal



# def get_indices_above_threshold(inData, threshold) :

# 	if threshold < 0 :
# 		sampleIndices = np.where(inData < threshold)

# 	else :
# 		sampleIndices = np.where(inData > threshold)

# 	return sampleIndices[0]



# def find_start_end_intercepts_of_pacing_events(indicesAboveThreshold) :

# 	startIndices = np.array([])
# 	endIndices = np.array([])
# 	InAPacingEvent = False

# 	timeDiffOfNeighboursPairs = np.diff(indicesAboveThreshold)

# 	indexI = 0

# 	indicesAboveThresholdList = list(indicesAboveThreshold)


# 	for indexI in range(0, (len(indicesAboveThresholdList)-1)) :

# 		index = indicesAboveThresholdList[indexI]

# 		if (timeDiffOfNeighboursPairs[indexI] > 30) :
# 			endIndices = np.append(endIndices, index)
# 			InAPacingEvent = False

# 		elif not InAPacingEvent :
# 			startIndices = np.append(startIndices, index)
# 			InAPacingEvent = True


# 	# If was in a pacing event when for loop ended, add the next index to the end indices
# 	if InAPacingEvent :

# 		index = indicesAboveThresholdList[indexI+1]
# 		endIndices = np.append(endIndices, index)

# 	print("Startindices.shape: ", startIndices.shape, " endIndices.shape: ", endIndices.shape)

# 	return (startIndices.astype(int), endIndices.astype(int))



# def make_indices_plot_friendly(sampledIndices, nChans) :

# 	chanIndices = range(0, (nChans-1))
# 	combinedSamples = list(np.hstack((sampledIndices[0], sampledIndices[1])))

# 	nSamples = len(combinedSamples)

# 	#print("nSamples: ", nSamples)
# 	#print("combinedSamples: ", combinedSamples)


# 	indicesX = np.array([])
# 	indicesY = np.array([])

# 	for chan in chanIndices :

# 		indicesX = np.append(indicesX, [combinedSamples])
# 		indicesY = np.append(indicesY, [[chan]] * nSamples)

# 	#print("indicesX.shape: ", indicesX.shape, " indicesY.shape ", indicesY.shape)
# 	#print("indicesX: ", indicesX)
# 	#print("indicesY: ", indicesY)

# 	return [indicesY.astype(int), indicesX.astype(int)]



# def smooth_segment(inChan, startMarker, endMarker, extraPoints=0) :
# 	''' Linearly smooth data '''

# 	# Create start and end point
# 	startIndex = startMarker - extraPoints
# 	endIndex = endMarker + extraPoints

# 	if endIndex >= inChan.shape[0] :
# 		endIndex = inChan.shape[0] - 1

# 	# y = mx + c
# 	m = (inChan[endIndex] - inChan[startIndex]) / (endIndex - startIndex)
# 	c = inChan[startIndex]

# 	for index in range(startIndex, endIndex) :
# 		x = (index - startIndex)
# 		inChan[index] = m * x + c

# 	return inChan



# def clean_pacing_at_pacing_events(inData, pacingMarkers, nChans) :

# 	''' Clean the pacing components '''
# 	nMarkers = len(pacingMarkers)

# 	cleanedData = np.empty(shape=inData.shape)
	
# 	#print("pacingMarkers: ", pacingMarkers)
# 	#print("Starting cleaning of pacing events . . . ")
# 	for chan in range(0, nChans) :

# 		currentChan = inData[chan,:]

# 		for startPaceEventMarker, endPaceEventMarker in zip(pacingMarkers[0], pacingMarkers[1]) :

# 			# print("startPaceEventMarker: ", startPaceEventMarker, "endPaceEventMarker: ", endPaceEventMarker )
# 			currentChan = smooth_segment(inData[chan], startPaceEventMarker, endPaceEventMarker, extraPoints=20)

# 		cleanedData[chan,:] = currentChan

# 	print("Finished preprocessing of pacing data . . . ")

# 	return cleanedData	



# def clean_pacing(chanData) :
# 	#[data, threshold] = findThreshold(bdfdat);
# 	extremeChanVals, pacingThreshold = find_pacing_threshold(chanData)

# 	nChans = chanData.shape[0]

# 	indicesAboveThresh = get_indices_above_threshold(extremeChanVals, pacingThreshold)

# 	pacingMarkers = find_start_end_intercepts_of_pacing_events(indicesAboveThresh)

# 	pacingMarkersForPlotting = make_indices_plot_friendly(pacingMarkers, nChans)

# 	pacingCleanedData = clean_pacing_at_pacing_events(chanData, pacingMarkers, nChans)

# 	return pacingMarkersForPlotting, pacingCleanedData




# # %% Process the channel to be taken as reference
# # [data, threshold] = findThreshold(bdfdat); 

# # %% Find the point of stimulus based on threshold
# # x_threshold = 1:length(data)-1:length(data);
# # y_threshold = [threshold threshold];
# # figure;
# # plot(data);
# # hold on;
# # plot(x_threshold, y_threshold, 'r');
# # [x0, y0] = intersections(1:length(data), data, x_threshold, y_threshold);
# # hold on;
# # plot(x0, y0, 'og');



# # %% Save processed files
# # toapp.filtdata = bdfdat;
# # processed_file_name = strcat(PathName,'processed_', FileName);
# # if exist(processed_file_name, 'file') ==  2
# #     disp('File exists')
# # else
# #     save(processed_file_name, 'bdfdef', 'toapp')
# # end

# # %% Plot processed file name
# # pause('on');
# # figure;
# # plot_end=size(data,2);
# # plot_pace = x0;
# # plot_pace(find(x0 > plot_end)) = [];
# # for j = 32:64
# #     channel = j;
# #     ax1 = subplot(2,1,1);
# #     plot(bdfdat(channel,:));
# #     hold on;plot(plot_pace, 0, 'og');hold off;
# #     ax2 = subplot(2,1,2);
# #     plot(copy_bdf(channel,:));
# #     linkaxes([ax1,ax2],'xy');
# #     pause(2);
# # end	