from config import default_window_size
import argparse
import os
from os.path import join as path_join
from typing import Generator

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from model import S3D

parser = argparse.ArgumentParser()
parser.add_argument('--frameFolder', type=str, default="", help='Full path of the folder containing per-video frames directories (Default: "./frames")')
parser.add_argument('--outFolder', type=str, default="", help='Output folder to store extracted parquet files (Default: "./features_logits")')
parser.add_argument('--window_size', type=int, default=default_window_size, help=f'The number of frames to feed the model (Default: {default_window_size})')
parser.add_argument('--parallel', type=bool, default=False, help='[Experimental!] Use parallel processing (Default: False)')
parser.add_argument('--parallel_jobs', type=int, default=8, help='# of jobs to do parallel processing (Default: 8)')

class S3DInference:
    def __init__(self, weight_file_path, class_names, device = None, num_class = 400):
        self.weight_file_path = weight_file_path
        self.class_names = class_names
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu") # cpu or cuda
        self.num_class = num_class
        self.model = S3D(self.num_class)

        # load the weight file and copy the parameters
        if os.path.isfile(self.weight_file_path):
            print ('loading weight file')
            weight_dict = torch.load(weight_file_path, map_location=torch.device(self.device))
            model_dict = self.model.state_dict()
            for name, param in weight_dict.items():
                if 'module' in name:
                    name = '.'.join(name.split('.')[1:])
                if name in model_dict:
                    if param.size() == model_dict[name].size():
                        model_dict[name].copy_(param)
                    else:
                        print (' size? ' + name, param.size(), model_dict[name].size())
                else:
                    print (' name? ' + name)

            print (f' loaded powered by {self.device}')
        else:
            print ('weight file?')

        if self.device != 'cpu':
            self.model = self.model.to(self.device)
        torch.backends.cudnn.benchmark = False
        self.model.eval()

    @staticmethod
    def make_snippets(video_frame_dir, list_frames, n_frames):
        # read every `n_frames` the frames of sample clip
        snippet = []
        for frame in list_frames[:n_frames-1]:
            img = cv2.imread(os.path.join(video_frame_dir, frame))
            img = img[...,::-1]
            snippet.append(img)
        for snippet_frame in list_frames[n_frames-1:]:
            img = cv2.imread(os.path.join(video_frame_dir, snippet_frame))
            img = img[...,::-1]
            snippet.append(img)
            yield snippet
            snippet = snippet[1:]

    @staticmethod
    def transform(snippet):
        ''' stack & noralization '''
        snippet = np.concatenate(snippet, axis=-1)
        snippet = torch.from_numpy(snippet).permute(2, 0, 1).contiguous().float()
        snippet = snippet.mul_(2.).sub_(255).div(255)
        return snippet.view(1,-1,3,snippet.size(1),snippet.size(2)).permute(0,2,1,3,4)

    def extract_featuremap_logits(self, video_frame_dir, window_size):
        list_frames = [f for f in os.listdir(video_frame_dir) if os.path.isfile(os.path.join(video_frame_dir, f))]
        list_frames.sort()

        for snippet in S3DInference.make_snippets(video_frame_dir, list_frames, window_size):
            clip = S3DInference.transform(snippet)
            with torch.no_grad():
                if self.device != 'cpu':
                    logits, feature_map = self.model(clip.to(self.device))
                    logits = logits.cpu()
                    feature_map = feature_map.cpu()
                else:
                    logits, feature_map = self.model(clip)
                logits = logits.data[0]
            preds = torch.softmax(logits, 0)

            yield feature_map, preds

def build_extractions_to_dataframe(extractor: Generator):
    df = pd.DataFrame(columns=['frame_idx', 'feature_map', 'logits'])
    for frame_idx, (feature_map, logits) in enumerate(extractor):
        df.loc[frame_idx] = [frame_idx, feature_map.numpy().tobytes(), logits.numpy().tobytes()]
    return df

def save_feature_map_and_logits(dir_path, video_key: str, df: pd.DataFrame):
    df.to_parquet(path_join(dir_path, f'{video_key}.parquet'), compression='gzip')

def save_and_build(video_idx, video_key, frames_dir, out_dir, window_size, model, parallel = True):
    video_frame_dir = path_join(frames_dir, video_key)
    total_frames = len(os.listdir(video_frame_dir))
    if total_frames < window_size:
        print(f'{video_idx:5d}: {video_key} has less than {window_size} frames ({total_frames})')
        return

    extracted_frames = total_frames - window_size + 1
    extractor = model.extract_featuremap_logits(video_frame_dir, window_size)
    if not parallel:
        extractor = tqdm(extractor, total = extracted_frames, desc=f'{video_idx:5d}: {video_key} is being extracted')

    df = build_extractions_to_dataframe(extractor)
    save_feature_map_and_logits(out_dir, video_key, df)
    if not parallel:
        tqdm.write(f'{video_idx:5d}: {video_key} is done')

def main():
    ''' Output the top 5 Kinetics classes predicted by the model '''
    label_path = path_join('.', 'label_map.txt')
    class_names = [c.strip() for c in open(label_path)]
    weight_file_path = path_join('.', 'pretrained_model', 'S3D_kinetics400.pt')

    args = parser.parse_args()

    frames_dir = args.frameFolder or path_join('.', 'frames')
    if args.frameFolder and not os.path.exists(args.frameFolder):
        print('frame folder does not exist')
        print(args.frameFolder)
        return
    
    out_dir = args.outFolder or path_join('.', 'features_logits')
    if args.outFolder and not os.path.exists(args.frameFolder):
        print('out folder does not exist')
        print(args.outFolder)
        return
    
    default_window_size = args.window_size
    if args.window_size < default_window_size:
        print(f'window size must be at least {default_window_size}')
        return

    model = S3DInference(weight_file_path, class_names)

    already_done_videos = set(filename[:-len(".parquet")] for filename in os.listdir(out_dir))
    video_keys = [dir for dir in os.listdir(frames_dir)
                  if os.path.isdir(os.path.join(frames_dir, dir))
                  if dir not in already_done_videos]
    
    if len(already_done_videos) > 0:
        print(f'{len(already_done_videos)} extracted files already exists and skiped')

    tasks = ({
        'video_idx' : video_idx,
        'video_key' : video_key,
        'frames_dir' : frames_dir,
        'out_dir' : out_dir,
        'window_size' : default_window_size,
        'model' : model,
    } for video_idx, video_key in enumerate(video_keys, 1))

    if args.parallel and args.parallel_jobs >= 1:
        from pqdm.processes import pqdm
        pqdm(tasks, save_and_build, n_jobs = args.parallel_jobs, argument_type='kwargs')
    else:
        for task in tasks:
            save_and_build(**task, parallel = False)


if __name__ == '__main__':
    main()
