# Philosophies ... one datafile per datatype.
#
# Specified order and whether processing steps are conducted

# First write a pipeline for an ideal dataset. But. Make the 'ideal' dataset as general
# as possible.

# An event is an index

# To do: automatically add output events hierchy to event hierarchies to plot.
# Have a singular 'data' folder that doesn't save over things that has had the same steps performed

# paramsSummaryStrs:  nghSpn7nChnSub2nChnGrp5delay5-20corPer11.48rat1.83


import sys
import os
 
from pipelines.analysis_pipeline import AnalysisPipeline

def main() :
	rootPyPath = os.path.dirname( os.path.realpath(__file__) )
	
	# Hard coding pipeline type -- can be entered as a parameter.
	if len(sys.argv) == 1 :
		sys.argv.append( 'PressureWavesPipeline' )

	print( "Loading ", sys.argv[1], " paradigm file." )

	ap = AnalysisPipeline( sys.argv[1], rootPyPath )
 
	# Load paradigm fileinputArr.shape
	ap.load_paradigm()

	# Build pipeline
	ap.build_pipeline()
 
	# Prepare files to load
	ap.prepare_files()
 
	# Run pipeline
	ap.run_pipeline()


if __name__ == '__main__' :
	main()