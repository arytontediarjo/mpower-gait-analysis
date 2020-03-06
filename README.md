# MPower Gait Analysis Project

This project is a part of Sage Bionetworks mHeatlh Tools, authored by Aryton Tediarjo (mHealth Analytics Co-op) and Larsson Omberg (VP of Systems Biology). This repo will cover steps from data querying processes, data featurization, analysis and classification of Treatment (Multiple Sclerosis and Parkinsons) of Control groups using their gait features. 

#### -- Project Status: [Work In Progress]


## Project Description

This project utilizes the gait signal data that is taken during the mPower walking/balance tests. It uses featurization process based on PDKit module, and automatic rotation-detection in gait. 



## References:
1. [Rotation-Detection Paper](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5811655/)

2. [PDKit Feature Documentations](https://pdkit.readthedocs.io/_/downloads/en/latest/pdf/)

3. [PDKit Gait Repository](https://github.com/pdkit/pdkit/blob/79d6127454f22f7ea352a2540c5b8364b21356e9/pdkit/gait_processor.py)

4. [Freeze of Gait Docs](https://ieeexplore.ieee.org/document/5325884)


## Data Pipeline (HOW-TO):


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

Running make container command will prompt user to enter credentials to synapseClient so that it will be saved to the container environment. It will also prompt user to insert the absolute filepaths to the .synapseCache.



**Notes on mounting .synapseCache volume to Docker container:**

In mounting volumes of downloaded files from .synapseCache to the docker container, an absolute filepath of file in the synapseCache locally will be mapped entirely into the Docker container. This is done as synapseclient only considers file to have the same md5 if their absolute paths are similar in their logged .cachemap file.



### 2. Generate featurized data:
```
make data
```


### 3. Update data with new records:
```
make update
```


### 4. Generate cleaned demographics information:
```
make demographics
```

### 5. Aggregate featurized data:

```
make aggregate
```


## Accessing Jupyter Notebooks
 
a. Jupyter Notebook
```
jupyter notebook
```

b. Jupyter Lab
```
jupyter lab
```

To access and edit the analysis in jupyter notebook, user can type in jupyter lab/notebook as a shell command. Afterwards, a link will appear in which user will click the link to run jupyter notebook server in browser of choice.


**Notes in running jupyter server in EC2 instance:**

To run jupyter server in EC2 instance, an SSH tunnelling from your local computer to the ec2 is required. Thus, running the command below in local bash terminal will give user tunnel access to the jupyter notebook ran in an ec2 instance, using browser of choice as its interface. 

```
ssh -NfL 8888:localhost:8888 ec2
```
