from enum import Enum
from config import default_window_size
from torch.utils.data import Dataset
from annotation import convert_seconds_to_frame_indices_in_segments
from features_logits_loader import load_feature_map_and_logits
from utils import checkParquetExist
import argparse

import numpy as np


class WindowState(Enum):
    START = 0
    END = 1
    NONE = 2


class LogitDataset(Dataset):
    def __init__(self, logits, clip_states) -> None:
        super().__init__()
        self.logits = logits
        self.clip_states = clip_states

    def __len__(self):
        return len(self.logits)

    def __getitem__(self, index):
        return self.logits[index], self.clip_states[index]


class FeatureMapDataset(Dataset):
    def __init__(self, feature_maps, clip_states) -> None:
        super().__init__()
        self.feature_maps = feature_maps
        self.clip_states = clip_states

    def __len__(self):
        return len(self.feature_maps)

    def __getitem__(self, index):
        return self.feature_maps[index].reshape(1,32,32), self.clip_states[index][0]


class DatasetGenerator():
    def __init__(self) -> None:
        self.segment_table = convert_seconds_to_frame_indices_in_segments()
        self.video_ids = self.segment_table.index.unique()

    def generate_feature_map_dataset(self,num_to_gen, num_windows = 1):
        print("##generating feature map dataset##")
        feature_maps, states = [], []
        for idx, video_id in enumerate(self.video_ids[:num_to_gen]):
            print(idx)
            if not checkParquetExist("v_"+video_id):
                continue
            feature_maps_states = self.videos_to_feature_maps_states(
                "v_"+video_id)
            total_frames = len(feature_maps_states[1])
            indices = np.arange(total_frames)
            indices_to_fetch = np.array([np.clip(np.arange(index - num_windows // 2, index + num_windows // 2 + 1), 0, total_frames - 1) for index in indices])
            feature_maps += [*np.array(feature_maps_states[0])[indices_to_fetch]]
            states += [*np.array(list(map(lambda s: s.value, feature_maps_states[1])))[indices_to_fetch]]
        return FeatureMapDataset(feature_maps, states)

    def generate_logit_dataset(self,num_to_gen, num_windows = 1):
        print("##generating logit dataset##")
        logits, states = [], []
        for idx, video_id in enumerate(self.video_ids[:num_to_gen]):
            print(idx)
            if not checkParquetExist("v_"+video_id): 
                continue
            logits_states = self.videos_to_logits_states("v_"+video_id)
            total_frames = len(logits_states[1])
            indices = np.arange(total_frames)
            indices_to_fetch = np.array([np.clip(np.arange(index - num_windows // 2, index + num_windows // 2 + 1), 0, total_frames - 1) for index in indices])
            logits += [*np.array(logits_states[0])[indices_to_fetch]]
            states += [*np.array(list(map(lambda s: s.value, logits_states[1])))[indices_to_fetch]]
        return LogitDataset(logits, states)

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

    def get_startframes_endframes(self, key: str):
        result = self.segment_table .loc[[
            key[2:]]][["start_frame", "end_frame"]]
        start_frames, end_frames = [], []
        for start_frame, end_frame in result.values:
            start_frames.append(start_frame)
            end_frames.append(end_frame)

        return start_frames, end_frames

    def window_state(self, index, start_frames, end_frames):
        if index in start_frames: return WindowState.START
        if index in end_frames: return WindowState.END
        return WindowState.NONE
        


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

    logitDataset = generator.generate_logit_dataset(num_parquet_to_use)
    featureMapDataset = generator.generate_feature_map_dataset(num_parquet_to_use)
    print(f"logit dataset len: {len(logitDataset)}")
    print(f"featuremap dataset len: {len(featureMapDataset)}")
