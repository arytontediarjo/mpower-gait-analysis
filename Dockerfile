# base image
FROM amancevice/pandas:0.23.4-python3

# updating repository
RUN git clone https://github.com/arytontediarjo/mpower-gait-analysis.git /root/mpower-gait-analysis

# remove later
COPY .synapseConfig /root/.synapseConfig

# upgrade pip
RUN /usr/local/bin/python -m pip install --upgrade pip

# install setups
RUN pip install -U pip setuptools wheel

# pip install 
RUN pip install -r /root/mpower-gait-analysis/requirements.txt


