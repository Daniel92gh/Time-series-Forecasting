import os 
import sys 
import torch
import json

from forecasting_library.utils.logger import logger

# set seed
seed = 123
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


def check_path(path):
    
    '''
    Checks if the directory path exists
    :param path to directory

    '''
    if not os.path.exists(path):
        logger.error("Data directory does not exist")
        sys.exit(1)


def check_file(path):
    '''
    Checks if a file exists
    :param path to the file

    '''

    if not os.path.isfile(path):
        logger.error("Data file does not exist")
        sys.exit(1)


def find_csv(path):
    '''
    Finds a csv file for training
    :param path to the directory

    '''

    csv_file = [f for f in os.listdir(path) if f.endswith('.csv')]
    if not csv_file:
        logger.error("Data for Traning does not exist!")
        sys.exit(1)
    else:
        return csv_file[0]    # only one csv file!


def load_config(path):

    '''
    Loads configuration file
    :param path config file path

    '''

    check_file(path)
    with open(path, 'r', encoding='utf-8') as f:
            params = json.load(f) 
    logger.info("Configuration file is loaded !!") 
    return params     


class dict2object(dict):
    def __init__(self, d):
        for k, v in d.items():
            self.__dict__[k] = self[k] = dict2object(v) if isinstance(v, dict) else v   

