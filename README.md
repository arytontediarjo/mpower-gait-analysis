# MPower Gait Analysis Project

This project is a part of Sage Bionetworks mHeatlh Tools, authored by Aryton Tediarjo (mHealth Analytics Co-op) and Larsson Omberg (VP of Systems Biology). This repo will cover steps from data querying processes, data featurization, analysis and classification of Treatment (Multiple Sclerosis and Parkinsons) of Control groups using their gait features. 

#### -- Project Status: [Work In Progress]

## Project Description

This project utilizes the gait signal data that is taken during the mPower walking/balance tests, which is stored in Synapse. All the gait data will be queried, and will be processed through a series of QC and featurize using the [pdkit git repository](link: https://github.com/pdkit/pdkit). 

## How-to-use:

All functionalities have been encapsulated in a makefile, thus some of the process of running this project can be called in a format of "make (some commands)"

### Using Docker

```
sudo service docker start
docker build -t <name of docker image> .
docker run -v ~/.synapseCache:/root/.synapseCache -it --rm pipeline /bin/bash
```

Once you are inside the docker container, there will be a Makefile that is already set with few rules on the data pipeline. 

#### 1. Create Featurized Data:
```
make data
```

#### 2. Query Cleaned Gait Demographics:
```
make demographics
```

#### 3. Get Aggregated Features:
```
make aggregate
```
