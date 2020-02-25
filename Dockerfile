# base image
FROM python:3.7

# updating repository
RUN git clone https://github.com/arytontediarjo/mpower-gait-analysis.git /root/mpower-gait-analysis

# upgrade pip
RUN /usr/local/bin/python -m pip install --upgrade pip

RUN pip install -U pip setuptools wheel

# pip install 
RUN pip install -r /root/mpower-gait-analysis/requirements.txt


