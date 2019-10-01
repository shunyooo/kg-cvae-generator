import tensorflow as tf

import argparse
import os
import yaml


def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    return sess


def load_config(path):
    with open(path) as f:
        return yaml.load(f)


def load_main_config():
    parser = argparse.ArgumentParser()
    default_config_file_path = os.path.join(os.path.dirname(__file__), 'config_service.yaml')
    parser.add_argument('-c', action="store", default=default_config_file_path)
    main_cofig_path = parser.parse_args().c
    return load_config(main_cofig_path)
