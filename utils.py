import os
from config import parquets_path
from os.path import join as path_join

__parquets_files = None

def checkParquetExist(parquetName, refresh = False):
    global __parquets_files
    if refresh or __parquets_files is None:
        __parquets_files = set(os.listdir(parquets_path))
    return f'{parquetName}.parquet' in __parquets_files
