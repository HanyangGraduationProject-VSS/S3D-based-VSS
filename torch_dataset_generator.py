import argparse
import os
from bisect import bisect_right
from enum import Enum

import h5py
import numpy as np
from torch.utils.data import Dataset
from tqdm.notebook import tqdm
from typing_extensions import Literal

from annotation import get_annotations_dataframe
from config import H5DF_DIRPATH, default_window_size


class WindowState(Enum):
    START = 0
    END = 1
    NONE = 2


def window_state(index, start_frames_set, end_frames_set):
    if index in start_frames_set:
        return WindowState.START
    if index in end_frames_set:
        return WindowState.END
    return WindowState.NONE

def get_state(current_frame_idx, start_frames_set, end_frames_set):
    return int(window_state(current_frame_idx, start_frames_set, end_frames_set).value)


class LogitDataset(Dataset):
    def __init__(self, video_keys, num_windows, h5df):
        super().__init__()

        self.num_windows = num_windows
        self.video_ids = video_keys[:]
        self.indices_partial_sum = [0]
        self.h5df = h5df

        for video_key in tqdm(self.video_ids, desc="generating logits dataset"):
            total_frames_in_the_video = self.h5df[video_key]['labels'].shape[0]
            self.indices_partial_sum.append(self.indices_partial_sum[-1] + total_frames_in_the_video)

    def __len__(self) -> int:
        return self.indices_partial_sum[-1] - self.indices_partial_sum[0]

    def __getitem__(self, index):
        video_idx = bisect_right(self.indices_partial_sum, index) - 1
        video_id = self.video_ids[video_idx]

        total_frames = self.indices_partial_sum[video_idx + 1] - self.indices_partial_sum[video_idx]
        assert total_frames > 0, f"the total frame of video {video_id} is 0"
        
        data = np.array(self.h5df[video_id].get('logits'))
        labels = np.array(self.h5df[video_id].get('labels'))
        
        current_frame_idx = index - self.indices_partial_sum[video_idx]
        indices_to_fetch = np.clip(
            np.arange(
                current_frame_idx - self.num_windows // 2,
                current_frame_idx - self.num_windows // 2 + self.num_windows
            ), 0, total_frames - 1
        )
        
        return data[indices_to_fetch], labels[indices_to_fetch]

    def get_video_data(self, video_key):
        if type(video_key) == int:
            video_idx = int(video_key)
            video_key = self.video_ids[video_idx]

        data = np.array(self.h5df[video_key].get('logits'))
        labels = np.array(self.h5df[video_key].get('labels'))
        
        return data, labels, video_key


class FeatureMapDataset(Dataset):
    def __init__(self, video_keys, num_windows, h5df):
        super().__init__()

        self.num_windows = num_windows
        self.video_ids = video_keys[:]
        self.indices_partial_sum = [0]
        self.h5df = h5df

        for video_key in tqdm(self.video_ids, desc="generating feature map dataset"):
            total_frames_in_the_video = self.h5df[video_key]['labels'].shape[0]
            self.indices_partial_sum.append(self.indices_partial_sum[-1] + total_frames_in_the_video)

    def __len__(self) -> int:
        return self.indices_partial_sum[-1] - self.indices_partial_sum[0]

    def __getitem__(self, index):
        video_idx = bisect_right(self.indices_partial_sum, index) - 1
        video_id = self.video_ids[video_idx]

        total_frames = self.indices_partial_sum[video_idx + 1] - self.indices_partial_sum[video_idx]
        assert total_frames > 0, f"the total frame of video {video_id} is 0"
        
        data = np.array(self.h5df[video_id].get('feature_map'))
        labels = np.array(self.h5df[video_id].get('labels'))
        
        current_frame_idx = index - self.indices_partial_sum[video_idx]
        indices_to_fetch = np.clip(
            np.arange(
                current_frame_idx - self.num_windows // 2,
                current_frame_idx - self.num_windows // 2 + self.num_windows
            ), 0, total_frames - 1
        )
        
        return data[indices_to_fetch], labels[indices_to_fetch]

    def get_video_data(self, video_key):
        if type(video_key) == int:
            video_idx = int(video_key)
            video_key = self.video_ids[video_idx]
            
        data = np.array(self.h5df[video_key].get('feature_map'))
        labels = np.array(self.h5df[video_key].get('labels'))
        
        return data, labels, video_key

class DatasetGenerator():
    DATASET_TYPE = Literal['train', 'test']

    def __init__(self, h5df_path: str, dataset_type: DATASET_TYPE = 'train'):
        if not os.path.exists(h5df_path):
            raise FileNotFoundError(f"{h5df_path} does not exist")

        self.h5df_path = h5df_path
        self.h5df = h5py.File(self.h5df_path, 'r')

        self.dataset = get_annotations_dataframe(dataset_type).set_index("video_id")
        self.available_video_keys = []

        video_ids = self.h5df.keys()
        
        not_exist_cnt = 0
        for video_id in tqdm(video_ids, desc="Checking missing data in the dataset"):
            if len(self.h5df[video_id].keys()) == 0: # 없는 경우 제외
                not_exist_cnt += 1
                tqdm.write(f"[{not_exist_cnt:5d}] {video_id} does not exist")
                continue

            self.available_video_keys.append(video_id)
        tqdm.write(f"Total {len(self.available_video_keys)} videos were read")

    def generate_feature_map_dataset(self, dataset_size = None, num_windows = 1):
        if dataset_size is None:
            dataset_size = len(self.available_video_keys)

        assert dataset_size >= 1, "dataset_set_size must be positive"
        assert dataset_size <= len(self.available_video_keys), "dataset_set_size must be smaller than dataset size"
        assert num_windows >= 1, "num_windows must be positive"

        print("## generating feature map dataset ##")

        video_keys = self.available_video_keys[:dataset_size]

        return FeatureMapDataset(video_keys, num_windows, self.h5df)

    def generate_logit_dataset(self, dataset_size = None, num_windows = 1):
        if dataset_size is None:
            dataset_size = len(self.available_video_keys)

        assert dataset_size >= 1, "dataset_set_size must be positive"
        assert dataset_size <= len(self.available_video_keys), "dataset_set_size must be smaller than dataset size"
        assert num_windows >= 1, "num_windows must be positive"

        print("## generating logits dataset ##")

        video_keys = self.available_video_keys[:dataset_size]

        return LogitDataset(video_keys, num_windows, self.h5df)


def isAmbiguous(index, startOrEndFrames):
    if index in startOrEndFrames:
        return False
    width = default_window_size // 2
    for frame in startOrEndFrames:
        if frame - width <= index <= frame + width + 1:
            return True
    return False


if __name__ == '__main__':
    pass
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--num_parquet_to_generate', type=int, default=200,
    #                     help="the number of parquet to make logit and feature_map")

    # args = parser.parse_args()

    # num_parquet_to_use = args.num_parquet_to_generate

    # generator = DatasetGenerator(os.path.join(H5DF_DIRPATH, 'valid.hdf5'), 'valid')

    # logitTrainDataset = generator.generate_logit_dataset(2)
    # featureMapTrainDataset = generator.generate_feature_map_dataset(2)

    # print(f"logit train dataset len: {len(logitTrainDataset)} (includes {len(logitTrainDataset.available_video_keys)} videos)")
    # print(f"featuremap train dataset len: {len(featureMapTrainDataset)} (includes {len(featureMapTrainDataset.available_video_keys)} videos)")
