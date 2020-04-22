# base image
FROM jupyter/scipy-notebook:latest

# updating repository
RUN git clone https://github.com/arytontediarjo/mpower-gait-analysis.git /home/jovyan/mpower-gait-analysis

# upgrade pip
RUN pip install --upgrade pip

# pip install 
RUN pip install -r /home/jovyan/mpower-gait-analysis/requirements.txt --ignore-installed
