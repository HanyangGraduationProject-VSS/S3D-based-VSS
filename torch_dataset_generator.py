import argparse
from bisect import bisect_right
from enum import Enum
from functools import lru_cache

import numpy as np
from deprecated import deprecated
from torch.utils.data import Dataset
from tqdm.notebook import tqdm
from typing_extensions import Literal

from annotation import get_annotations_dataframe, convert_seconds_to_frame_indices_in_segments
from config import default_window_size
from features_logits_loader import load_feature_map_and_logits
from utils import checkParquetExist
from torch import Tensor


class WindowState(Enum):
    START = 0
    END = 1
    NONE = 2

def window_state(index, start_frames, end_frames):
    if (start_frames == index).any():
        return WindowState.START
    if (end_frames == index).any():
        return WindowState.END
    return WindowState.NONE


class LogitDataset(Dataset):
    def __init__(self, video_keys, num_windows, get_video_num_of_frames, get_video_logits, get_segment_data):
        super().__init__()

        self.num_windows = num_windows
        self.available_video_keys = video_keys[:]
        self.video_key_to_idx = {key: idx for idx, key in enumerate(self.available_video_keys)}
        self.indices_partial_sum = [0]
        self.get_video_num_of_frames = get_video_num_of_frames
        self.get_video_logits = get_video_logits
        self.get_segment_data = get_segment_data

        for video_key in tqdm(self.available_video_keys, desc="generating feature map train dataset"):
            total_frames_in_the_video = self.get_video_num_of_frames(video_key)
            self.indices_partial_sum.append(self.indices_partial_sum[-1] + total_frames_in_the_video)

    def __len__(self) -> int:
        return self.indices_partial_sum[-1] - self.indices_partial_sum[0]

    def __getitem__(self, index):
        video_idx = bisect_right(self.indices_partial_sum, index) - 1
        video_key = self.available_video_keys[video_idx]
        total_frames = self.indices_partial_sum[video_idx + 1] - self.indices_partial_sum[video_idx]
        assert total_frames > 0, f"the total_frame of {video_key} is 0"
        cached_video = self.get_video_logits(video_key)
        segments = self.get_segment_data(video_key)
        
        frame_idx_in_the_video = index - self.indices_partial_sum[video_idx]
        indices_to_fetch = np.clip(
            np.arange(
                frame_idx_in_the_video - self.num_windows // 2,
                frame_idx_in_the_video - self.num_windows // 2 + self.num_windows
            ), 0, total_frames - 1
        )
        
        states = np.array([int(window_state(idx, segments.start_frame, segments.end_frame).value) for idx in indices_to_fetch])
        return cached_video[indices_to_fetch], states

    def get_video_data(self, video_key):
        if type(video_key) == int:
            video_key = self.available_video_keys[video_key]
        cached_video = self.get_video_logits(video_key)
        segments = self.get_segment_data(video_key)

        total_frames = cached_video.shape[0]
        states = np.array([int(window_state(idx, segments.start_frame, segments.end_frame).value) for idx in range(total_frames)])
        return cached_video, states, video_key


class FeatureMapDataset(Dataset):
    def __init__(self, video_keys, num_windows, get_video_num_of_frames, get_video_feature_map, get_segment_data):
        super().__init__()

        self.num_windows = num_windows
        self.available_video_keys = video_keys[:]
        self.video_key_to_idx = {key: idx for idx, key in enumerate(self.available_video_keys)}
        self.indices_partial_sum = [0]
        self.get_video_num_of_frames = get_video_num_of_frames
        self.get_video_feature_map = get_video_feature_map
        self.get_segment_data = get_segment_data

        for video_key in tqdm(self.available_video_keys, desc="generating feature map train dataset"):
            total_frames_in_the_video = self.get_video_num_of_frames(video_key)
            self.indices_partial_sum.append(self.indices_partial_sum[-1] + total_frames_in_the_video)

    def __len__(self) -> int:
        return self.indices_partial_sum[-1] - self.indices_partial_sum[0]

    def __getitem__(self, index):
        video_idx = bisect_right(self.indices_partial_sum, index) - 1
        video_key = self.available_video_keys[video_idx]
        total_frames = self.indices_partial_sum[video_idx + 1] - self.indices_partial_sum[video_idx]
        assert total_frames > 0, f"the total frame of video {video_key} is 0"
        
        cached_video = self.get_video_feature_map(video_key)
        segments = self.get_segment_data(video_key)
        
        frame_idx_in_the_video = index - self.indices_partial_sum[video_idx]
        indices_to_fetch = np.clip(
            np.arange(
                frame_idx_in_the_video - self.num_windows // 2,
                frame_idx_in_the_video - self.num_windows // 2 + self.num_windows
            ), 0, total_frames - 1
        )
        
        states = np.array([int(window_state(idx, segments.start_frame, segments.end_frame).value) for idx in indices_to_fetch])
        return cached_video[indices_to_fetch], states

    def get_video_data(self, video_key):
        if type(video_key) == int:
            video_key = self.available_video_keys[video_key]
        cached_video = self.get_video_feature_map(video_key)
        segments = self.get_segment_data(video_key)

        total_frames = cached_video.shape[0]
        states = np.array([int(window_state(idx, segments.start_frame, segments.end_frame).value) for idx in range(total_frames)])
        return cached_video, states, video_key

class DatasetGenerator():
    DATASET_TYPE = Literal['train', 'val', 'test']

    def __init__(self, dataset_type: DATASET_TYPE = 'train'):
        self.dataset = get_annotations_dataframe(dataset_type).set_index("video_id")
        self.segment_table = convert_seconds_to_frame_indices_in_segments(df = dataset_type)
        self.video_ids = self.segment_table.index.unique()
        self.available_video_keys = []

        not_exist_idx = 1
        for video_id in tqdm(self.video_ids, desc="generating feature map train dataset"):
            video_key = f"v_{video_id}"
            if not checkParquetExist(video_key): # 없는 파일 제외
                tqdm.write(f"[{not_exist_idx:5d}] {video_key} does not exist")
                not_exist_idx += 1
                continue
            self.available_video_keys.append(video_key)
        tqdm.write(f"Total {len(self.available_video_keys)} videos were read")

    def generate_feature_map_dataset(self, train_set_size, valid_set_size, num_windows = 1):
        assert train_set_size + valid_set_size >= 1, "train_set_size + valid_set_size must be positive"
        assert num_windows >= 1, "num_windows must be positive"

        print("## generating feature map dataset ##")

        train_video_keys = []
        valid_video_keys = []

        for video_key in tqdm(self.available_video_keys[:train_set_size], desc="identifying feature map train dataset") :
            train_video_keys.append(video_key)
        
        for video_key in tqdm(self.available_video_keys[train_set_size:train_set_size+valid_set_size], desc="identifying feature map validation dataset") :
            valid_video_keys.append(video_key)
        
        return FeatureMapDataset(train_video_keys, num_windows, self.get_video_num_of_frames, self.get_video_feature_map, self.get_segment_data), \
               FeatureMapDataset(valid_video_keys, num_windows, self.get_video_num_of_frames, self.get_video_feature_map, self.get_segment_data)

    def generate_logit_dataset(self, train_set_size, valid_set_size, num_windows = 1):
        assert train_set_size + valid_set_size >= 1, "train_set_size + valid_set_size must be positive"
        assert num_windows >= 1, "num_windows must be positive"

        print("## generating logit dataset ##")

        train_video_keys = []
        valid_video_keys = []

        for video_key in tqdm(self.available_video_keys[:train_set_size], desc="generating feature map train dataset") :
            train_video_keys.append(video_key)
        
        for video_key in tqdm(self.available_video_keys[train_set_size:train_set_size+valid_set_size], desc="generating feature map validation dataset") :
            valid_video_keys.append(video_key)
        
        return LogitDataset(train_video_keys, num_windows, self.get_video_num_of_frames, self.get_video_logits, self.get_segment_data), \
               LogitDataset(valid_video_keys, num_windows, self.get_video_num_of_frames, self.get_video_logits, self.get_segment_data)
    
    @lru_cache(maxsize=2 ** 12)
    def get_video_feature_map(self, video_key: str):
        feature_map = load_feature_map_and_logits(video_key).feature_map
        return np.stack(feature_map.values)
    
    @lru_cache(maxsize=2 ** 12)
    def get_video_logits(self, video_key: str):
        logits = load_feature_map_and_logits(video_key).logits
        return np.stack(logits.values)

    @lru_cache(maxsize=2 ** 14)
    def get_segment_data(self, video_key: str):
        return self.get_startframes_endframes(video_key)

    @lru_cache(maxsize=2 ** 14)
    def get_video_num_of_frames(self, video_key: str):
        return load_feature_map_and_logits(video_key).shape[0]

    @lru_cache(maxsize=2 ** 14)
    def get_startframes_endframes(self, key: str):
        return self.segment_table.loc[[key[2:]]][["start_frame", "end_frame"]]

    @deprecated(reason="Memory Inefficient")
    def videos_to_feature_maps_states(self, key: str):
        target_feature_maps = load_feature_map_and_logits(key).feature_map
        feature_maps, states = [], []
        start_frames, end_frames = self.get_startframes_endframes(
            key)

        for i in range(len(target_feature_maps)):
            feature_map = target_feature_maps[i]
            # if isAmbiguous(i, start_frames) or isAmbiguous(i, end_frames):
            #     continue
            feature_maps.append(feature_map)
            states.append(self.window_state(i,start_frames, end_frames))
        return (feature_maps, states)

    @deprecated(reason="Memory Inefficient")
    def videos_to_logits_states(self, key: str):
        target_logits = load_feature_map_and_logits(key).logits
        logits, states = [], []
        start_frames, end_frames = self.get_startframes_endframes(
            key)

        for i in range(len(target_logits)):
            logit = target_logits[i]
            # if isAmbiguous(i, start_frames) or isAmbiguous(i, end_frames):
            #     continue
            logits.append(logit)
            states.append(self.window_state(i,start_frames, end_frames))
        return (logits, states)


def isAmbiguous(index, startOrEndFrames):
    if index in startOrEndFrames:
        return False
    width = default_window_size // 2
    for frame in startOrEndFrames:
        if frame - width <= index <= frame + width + 1:
            return True
    return False


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_parquet_to_generate', type=int, default=200,
                        help="the number of parquet to make logit and feature_map")

    args = parser.parse_args()

    num_parquet_to_use = args.num_parquet_to_generate

    generator = DatasetGenerator()

    logitTrainDataset, logitValidDataset = generator.generate_logit_dataset(int(num_parquet_to_use * 0.5), int(num_parquet_to_use * 0.5))
    featureMapTrainDataset, featureMapValidDataset = generator.generate_feature_map_dataset(int(num_parquet_to_use * 0.5), int(num_parquet_to_use * 0.5))

    print(f"logit train dataset len: {len(logitTrainDataset)} (includes {len(logitTrainDataset.available_video_keys)} videos)")
    print(f"logit valid dataset len: {len(logitValidDataset)} (includes {len(logitValidDataset.available_video_keys)} videos)")
    print(f"featuremap train dataset len: {len(featureMapTrainDataset)} (includes {len(featureMapTrainDataset.available_video_keys)} videos)")
    print(f"featuremap valid dataset len: {len(featureMapValidDataset)} (includes {len(featureMapValidDataset.available_video_keys)} videos)")
