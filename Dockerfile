# base image
FROM jupyter/scipy-notebook:latest

# updating repository
RUN git clone -b test_dependencies https://github.com/arytontediarjo/mpower-gait-analysis.git /home/jovyan/mpower-gait-analysis

# pip install 
RUN pip install -r /home/jovyan/mpower-gait-analysis/requirements_clean.txt