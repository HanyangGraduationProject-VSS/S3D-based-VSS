from argparse import ArgumentError
import os
from config import parquets_path_test
from config import parquets_path_train
from os.path import join as path_join
import sys

__parquets_files = None

def checkParquetExist(parquetName, refresh = False):
    print("DeprecatedAPI!! use new Api: checkTrainParquetExist,checkTestParquetExist")
    sys.exit()
    global __parquets_files
    if refresh or __parquets_files is None:
        __parquets_files = set(os.listdir(parquets_path))
    return f'{parquetName}.parquet' in __parquets_files

def checkParquetExist(parquetName, dataset: str,refresh = False):
    global __parquets_files
    if refresh or __parquets_files is None:
        if dataset == "train":
            __parquets_files = set(os.listdir(parquets_path_train))
        elif dataset == "train":
            __parquets_files = set(os.listdir(parquets_path_train))
        raise ArgumentError("dataset should be train or test")

            
    return f'{parquetName}.parquet' in __parquets_files