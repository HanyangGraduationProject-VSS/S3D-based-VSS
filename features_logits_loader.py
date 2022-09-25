import torch
from os.path import join as path_join
import os
import pandas as pd
import numpy as np
from config import parquets_path_train
from config import parquets_path_test


# feature_dirs = (dir for dir in os.listdir(path_feature)
#                   if os.path.isdir(os.path.join(path_feature, dir)))

# for feature_dir in feature_dirs:
#     dir = path_join(path_feature,feature_dir)
#     for feature in os.listdir(dir):
#         print(torch.load(path_join(dir,feature)))
    
def load_feature_map_and_logits(video_key: str, dataset: str):
    # path = path_join('.', 'features_logits', f'{video_key}.parquet')
    path = ""
    if dataset == "train":
        path = path_join(parquets_path_train, f'{video_key}.parquet')
    else:
        path = path_join(parquets_path_test, f'{video_key}.parquet')
    
    parquet_pd = pd.read_parquet(path)
    parquet_pd.feature_map = \
        parquet_pd.feature_map.apply(lambda x : np.frombuffer(x,dtype = np.float32))
    parquet_pd.logits = \
        parquet_pd.logits.apply(lambda x : np.frombuffer(x,dtype = np.float32))
    return parquet_pd