# base image
FROM jupyter/scipy-notebook:latest

# updating repository
RUN git clone -b reviewed_branch https://github.com/arytontediarjo/mpower-gait-analysis.git /home/jovyan/mpower-gait-analysis

# upgrade pip
RUN pip3 install --upgrade pip

# pip install 
RUN pip3 install -r /home/jovyan/mpower-gait-analysis/requirements.txt --ignore-installed