import os
from os.path import join as path_join
import cv2
import numpy as np
import torch
from tqdm import tqdm
from model import S3D

class S3DInference:
    def __init__(self, weight_file_path, device = None, num_class = 400):
        self.weight_file_path = weight_file_path
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