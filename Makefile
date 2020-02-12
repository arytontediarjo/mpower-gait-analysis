#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Install Python Dependencies
requirements: test_environment
	$(PYTHON_INTERPRETER) -m pip install -U pip setuptools wheel
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt

## Make Dataset
data: 
	$(PYTHON_INTERPRETER) src/pipeline/build_gait_features.py 

metadata:
	$(PYTHON_INTERPRETER) src/pipeline/build_metadata.py

group:
	$(PYTHON_INTERPRETER) src/pipeline/build_grouped_features.py

## Update dataset only based on new recordId
update:
	$(PYTHON_INTERPRETER) src/pipeline/build_gait_features.py --update

train:
	$(PYTHON_INTERPRETER) src/models/train_model.py 


## Delete all compiled Python files and files in synapseCache
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
	find ~/.synapseCache -type f -name "featurized_gait_data.csv" -delete

## Lint using flake8
lint:
	flake8 src


## Set up python interpreter environment
create_environment:
	$(PYTHON_INTERPRETER) -m pip install -q virtualenv virtualenvwrapper
	@echo ">>> Installing virtualenvwrapper if not already installed.\nMake sure the following lines are in shell startup file\n\
	export WORKON_HOME=$$HOME/.virtualenvs\nexport PROJECT_HOME=$$HOME/Devel\nsource /usr/local/bin/virtualenvwrapper.sh\n"
	@bash -c "source `which virtualenvwrapper.sh`;mkvirtualenv $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER)"
	@echo ">>> New virtualenv created. Activate with:\nworkon $(PROJECT_NAME)"


## Test python environment is setup correctly
test_environment:
	$(PYTHON_INTERPRETER) test_environment.py

