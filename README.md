# MPower Gait Analysis using PDKit and Rotation Time

This project is a part of Sage Bionetworks mHeatlh Tools, authored by Aryton Tediarjo (mHealth Analytics Co-op) and Larsson Omberg (VP of Systems Biology). This repo will cover steps from data querying processes, data featurization, analysis and classification of Treatment (Multiple Sclerosis and Parkinsons) of Control groups using their gait features. 


## Project Description

This project utilizes the gait signal data that is taken during the mPower walking/balance tests. It uses featurization process based on PDKit module, and automatic rotation-detection in gait. 


## References:
1. [Rotation-Detection Paper](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5811655/)

2. [PDKit Feature Documentations](https://pdkit.readthedocs.io/_/downloads/en/latest/pdf/)

3. [PDKit Gait Repository](https://github.com/pdkit/pdkit/blob/79d6127454f22f7ea352a2540c5b8364b21356e9/pdkit/gait_processor.py)

4. [Freeze of Gait Docs](https://ieeexplore.ieee.org/document/5325884)


## Data Pipeline (HOW-TO):


### Utilizing Makefile:

For ease of use, all pipeline functionalities have been encapsulated into make commands listed in the Makefile, pipeline parameters can also be changed in the Makefile (number of cores, number of partition, featurization parameter). 

Thus, for any future work or changes, you can change the parameter variable through the Makefile.


### 1. Create Docker Environment:

To build a container of the project, go to the project repository and create an image of the required dependencies by building a docker image. 

#### Build the Docker Image
``` 
docker build -t gait-analysis-jupyter-image .
```

To get into the pipeline environment a python wrapper has been created for ease of use. Run the command below to execute the Makefile commands that will run the Docker wrapper script.

#### Run the Docker Container
```
make container 
```

When this command is being ran, it will expose docker container to port 8888 so that we can ssh our browser to the jupyter notebook ran inside the container. 

Several synapse credentials question will be asked to access the container (synapse username, and synapse password). Additionally it will prompt user to input absolute filepaths of .synapsecache for faster feature query and absolute filepaths to github authentication token for updating scripts in Synapse Provenance. 


**Notes on mounting .synapseCache volume to Docker container:**

In mounting volumes of downloaded files from .synapseCache to the docker container, an absolute filepath of file in the synapseCache locally will be mapped entirely into the Docker container. This is done as synapseclient only considers file to have the same md5 if their absolute paths are similar in their logged .cachemap file.


### 2. Generate featurized data:

```bat
make data
```

Running this make command will featurize all gait data in synapse database.


### 3. Update data with new records:
```bat
make update
```

Running this make command will update featurized data by comparing new recordId (unique identifier) with kept records (unique identifier) and append to the feature data with  new recordId.


### 4. Generate cleaned demographics information:
```bat
make demographics
```

### 5. Aggregate featurized data:

```bat
make aggregate
```
Running this command will aggregate all data by its recordId and healthcodes. 
Each Features will be aggregated by interquartiles and median.


## Accessing Jupyter Notebooks
 
#### Jupyter Notebook
```bat
jupyter notebook
```

#### Jupyter Lab
```bat
jupyter lab
```

To access and edit the analysis in jupyter notebook, user can type in jupyter lab/notebook as a shell command. Afterwards, a link will appear in which user will click the link to run jupyter notebook server in browser of choice.


**Notes in running jupyter server in EC2 instance:**

To run jupyter server in EC2 instance, an SSH tunnelling from your local computer to the ec2 is required. Thus, running the command below in local bash terminal will give user tunnel access to the jupyter notebook ran in an ec2 instance, using browser of choice as its interface. 

```bat
ssh -NfL 8888:localhost:8888 ec2
```
