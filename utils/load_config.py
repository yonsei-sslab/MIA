from easydict import EasyDict
import yaml


def load_config(config_name, config_path="./config.yaml"):
    # Read config.yaml file
    with open(config_path) as infile:
        SAVED_CFG = yaml.load(infile, Loader=yaml.FullLoader)
        CFG = EasyDict(SAVED_CFG[config_name])
    return CFG
