from enum import Enum
from config import default_window_size
from torch.utils.data import Dataset
from annotation import convert_seconds_to_frame_indices_in_segments
from features_logits_loader import load_feature_map_and_logits
from utils import checkParquetExist
import pandas
import argparse


class WindowState(Enum):
    START = 1
    END = 2
    NONE = 3


class LogitDataset(Dataset):
    def __init__(self, logits, clip_states) -> None:
        super().__init__()
        self.logits = logits
        self.clip_states = clip_states

    def _len(self):
        return len(self.logits)

    def __getitem__(self, index):
        return self.logits[index], self.clip_states[index]


def generateDataset(video_ids, segment_table):
    logits, states = [], []
    for idx, video_id in enumerate(video_ids):
        if not checkParquetExist("v_"+video_id): 
            print(f"passed: index = {idx}, id =  {video_id}")
            continue
        logits_states = videos_to_logits_states("v_"+video_id, segment_table)
        logits += logits_states[0]
        states += logits_states[1]
        print(f"generated index = {idx}, id =  {video_id}")
    return LogitDataset(logits, states)


def videos_to_logits_states(key: str, table):
    target_logits = load_feature_map_and_logits(key).logits
    logits, states = [], []
    # 로짓의 상태 판별하기
    start_frames, end_frames = get_startframes_endframes(key, table)

    for i in range(len(target_logits)):
        logit = target_logits[i]
        # if isAmbiguous(i, start_frames) or isAmbiguous(i, end_frames):
        #     continue
        logits.append(logit)
        if i in start_frames:
            states.append(WindowState.START)
        elif i in end_frames:
            states.append(WindowState.END)
        else:
            states.append(WindowState.NONE)
    return (logits, states)


def isAmbiguous(index, startOrEndFrames):
    if index in startOrEndFrames:
        return False
    width = default_window_size // 2
    for frame in startOrEndFrames:
        if frame - width <= index <= frame + width + 1:
            return True
    return False


def get_startframes_endframes(key: str, table):
    result = table.loc[[key[2:]]][["start_frame", "end_frame"]]
    print(result)
    start_frames, end_frames = [], []
    for start_frame, end_frame in result.values:
        start_frames.append(start_frame)
        end_frames.append(end_frame)

    return start_frames, end_frames


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_parquet_to_generate', type=int, default = 200, help ="the number of parquet to make logit")

    args = parser.parse_args()
    
    num_parquet_to_use = args.num_parquet_to_generate

    segment_table = convert_seconds_to_frame_indices_in_segments()
    video_ids = segment_table.index.unique()[:num_parquet_to_use]
    print(video_ids)
    # video_ids = ["fJ45W32t6h0"]
    logitDataset = generateDataset(video_ids, segment_table)
