# base image
FROM jupyter/scipy-notebook:latest

# updating repository
RUN git clone -b to_review_2 https://github.com/arytontediarjo/mpower-gait-analysis.git /home/jovyan/mpower-gait-analysis

# upgrade pip
RUN pip install --upgrade pip

# pip install 
RUN pip install -r /home/jovyan/mpower-gait-analysis/requirements.txt