#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = mpower-gait-analysis
PYTHON_INTERPRETER = python
	
#################################################################################
# COMMANDS                                                                      #
#################################################################################

# Container setting
container:
	$(PYTHON_INTERPRETER) src/docker_wrapper.py

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
	
