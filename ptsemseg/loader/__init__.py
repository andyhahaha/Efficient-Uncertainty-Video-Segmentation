import json

from ptsemseg.loader.camvid_dataset import CamVid
#from ptsemseg.loader.joint_transforms import *
def get_loader(name):
    """get_loader

    :param name:
    """
    return {
        'camvid': CamVid
    }[name]


def get_data_path(name, config_file='config.json'):
    """get_data_path

    :param name:
    :param config_file:
    """
    data = json.load(open(config_file))
    return data[name]['data_path']
