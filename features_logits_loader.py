import torch
from os.path import join as path_join
import os
import pandas as pd
import numpy as np


# feature_dirs = (dir for dir in os.listdir(path_feature)
#                   if os.path.isdir(os.path.join(path_feature, dir)))

# for feature_dir in feature_dirs:
#     dir = path_join(path_feature,feature_dir)
#     for feature in os.listdir(dir):
#         print(torch.load(path_join(dir,feature)))
    
def load_feature_map_and_logits(video_key: str):
    # path = path_join('.', 'features_logits', f'{video_key}.parquet')
    path = path_join('/content/drive/MyDrive/ActivityNet200/parquets', f'{video_key}.parquet')
    parquet_pd = pd.read_parquet(path)
    parquet_pd.feature_map = \
        parquet_pd.feature_map.apply(lambda x : np.frombuffer(x,dtype = np.float32))
    parquet_pd.logits = \
        parquet_pd.logits.apply(lambda x : np.frombuffer(x,dtype = np.float32))
    return parquet_pd