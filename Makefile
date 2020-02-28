#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = mpower-gait-analysis
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Install Python Dependencies
requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip setuptools wheel
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt

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



