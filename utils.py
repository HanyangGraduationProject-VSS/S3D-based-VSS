import os
from config import parquets_path
from os.path import join as path_join


def checkParquetExist(parquetName):
    path = path_join(parquets_path, f'{parquetName}.parquet')
    return os.path.exists(path)
