"""
Script to create Jupyter Stacks docker container
with mounted volume to synapeCache (absolute path),
will require user input for credentials and
caching instructions
"""

import configparser
import os
import multiprocessing


def build_synapse_config(path, config_dict):
    """
    function to write in .synapseConfig inside the
    project directory, will be used to store
    credentials and caching information

    Args:
        path (type = string) = path to synapseConfig
        config_dict (type = dictionary) = mapped sections in config
    """
    config = configparser.ConfigParser()
    for section in config_dict.keys():
        config.add_section(section)
        for key, values in config_dict[section].items():
            config.set(section, key, values)
    cfgfile = open(path, "w+")
    config.write(cfgfile)
    cfgfile.close()


def main():
    syn_username = input("Enter synapse username: ")
    syn_password = input("Enter synapse password: ")
    cache_path = input("Enter cache path: ")
    config_dict = {
        "cache": {
            "location": cache_path}
    }
    config_path = \
        os.path.join(os.path.dirname(os.getcwd()),
                     "mpower-gait-analysis/.synapseConfig")

    build_synapse_config(config_path, config_dict)

    # create docker
    os.system("docker run --rm -e syn_username={} -e syn_password={} -p 8888:8888 \
        --cpus {} \
        -v {}:/home/jovyan/mpower-gait-analysis/.synapseConfig \
        -v {}:{} \
        -it gait-analysis-jupyter-image /bin/bash"
              .format(syn_username, syn_password,
                      multiprocessing.cpu_count()/2,
                      config_path, cache_path, cache_path))


if __name__ == '__main__':
    main()
