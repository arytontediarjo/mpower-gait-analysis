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


## How-to-use:

All functionalities have been encapsulated in a makefile, thus some of the process of running this project can be called in a format of "make (some commands)"


### Using pip
```
pip install -r requirements.txt
```

### Using Docker

```
docker build -t <name of docker image> .
docker run -v <path-local-volume>:<path-to-designated-volume-location> -it --rm <name-of-image> /bin/bash
```

#### 1. Create Featurized Data:
```
make data
```

#### 2. Update Data:
```
make update
```

#### 3. Query Cleaned Gait User Demographics:
```
make demographics
```

#### 4. Retrieve Aggregated Features:
```
make aggregate
```
