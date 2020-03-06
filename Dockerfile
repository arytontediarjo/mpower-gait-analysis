# base image
FROM jupyter/scipy-notebook:latest

# updating repository
RUN git clone https://github.com/arytontediarjo/mpower-gait-analysis.git /home/jovyan/mpower-gait-analysis

COPY requirements.txt /home/jovyan/mpower-gait-analysis/requirements.txt

COPY Makefile /home/jovyan/mpower-gait-analysis/Makefile

# upgrade pip
RUN pip install --upgrade pip

# pip install 
RUN pip install -r /home/jovyan/mpower-gait-analysis/requirements.txt