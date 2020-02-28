import os
from setuptools import find_packages, setup

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='src',
    packages=find_packages(),
    version='0.1.0',
    description='A repository of gait data featurization, exploration and data modelling',
    author='Aryton-SageBionetworks',
    license='MIT',
    install_requrires = required,
)
