#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = mpower-gait-analysis
PYTHON_INTERPRETER = python

# source volume to volume in docker should be the same as 
# cached file in cache map is based on absolute path to the file itself
# this is subject to improvement
SOURCE_VOLUME = ~/.synapseCache
TARGET_VOLUME = /home/ec2-user/.synapseCache



#########################
# CREDENTIALS			#
#########################

# rule in making .synapseConfig
credentials:
	$(PYTHON_INTERPRETER) src/config_wrapper.py 
	
#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Make container using jupyter image, exposed to port for notebook access
## and using mounted volume for 
run_docker:
	docker run --rm -p 8888:8888 \
	-v $(SOURCE_VOLUME):$(TARGET_VOLUME) \
	-it gait-analysis-jupyter-image /bin/bash

## Make Dataset
data: 
	$(PYTHON_INTERPRETER) src/pipeline/build_gait_features.py 

## Make Demographics
demographics:
	$(PYTHON_INTERPRETER) src/pipeline/build_demographics.py

## Make aggregation
aggregate:
	$(PYTHON_INTERPRETER) src/pipeline/build_aggregation.py --group recordId
	$(PYTHON_INTERPRETER) src/pipeline/build_aggregation.py --group healthCode

## Update dataset only based on new recordId
update:
	$(PYTHON_INTERPRETER) src/pipeline/build_gait_features.py --update
	
## Delete all compiled Python files and files in synapseCache
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
	find ~/.synapseCache/. -type f -name "MPOWER_V1_gait_features.csv" -delete
	find ~/.synapseCache/. -type f -name "MPOWER_V2_gait_features.csv" -delete
	find ~/.synapseCache/. -type f -name "MPOWER_PASSIVE_gait_features.csv" -delete
	find ~/.synapseCache/. -type f -name "ELEVATE_MS_gait_features.csv" -delete
	
