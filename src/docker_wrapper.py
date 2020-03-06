import configparser
import os


def build_synapse_config(path, config_dict):
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
        "authentication": {
            "username": syn_username,
            "password": syn_password},
        "cache": {
            "location": cache_path}
    }
    config_path = \
        os.path.join(os.path.dirname(os.getcwd()),
                     "mpower-gait-analysis/.synapseConfig")

    build_synapse_config(config_path, config_dict)

    # create docker
    os.system("docker run --rm -p 8888:8888 \
        -v {}:/home/jovyan/mpower-gait-analysis/.synapseConfig \
        -v {}:{} \
        -it gait-analysis-jupyter-image /bin/bash"
              .format(config_path, cache_path, cache_path))


if __name__ == '__main__':
    main()
