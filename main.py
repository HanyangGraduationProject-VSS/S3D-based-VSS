import os
import numpy as np
import cv2
import torch
from os.path import join as path_join
from model import S3D

class S3DModel:
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

            print (' loaded')
        else:
            print ('weight file?')

        if self.device != 'cpu':
            self.model = self.model.cuda()
        torch.backends.cudnn.benchmark = False
        self.model.eval()

    def predict(self, video_frame_dir):
        list_frames = [f for f in os.listdir(video_frame_dir) if os.path.isfile(os.path.join(video_frame_dir, f))]
        list_frames.sort()

        # read all the frames of sample clip
        snippet = []
        for frame in list_frames:
            img = cv2.imread(os.path.join(video_frame_dir, frame))
            img = img[...,::-1]
            snippet.append(img)

        clip = transform(snippet)

        with torch.no_grad():
            if self.device != 'cpu':
                logits = self.model(clip.cuda()).cpu().data[0]
            else:
                logits = self.model(clip).data[0]

        preds = torch.softmax(logits, 0).numpy()
        sorted_indices = np.argsort(preds)[::-1][:5]

        print ('\nTop 5 classes ... with probability')
        for idx in sorted_indices:
            print (self.class_names[idx], '...', preds[idx])
        return sorted_indices

def main():
    ''' Output the top 5 Kinetics classes predicted by the model '''
    path_sample = path_join('.', 'frames')
    label_path = path_join('.', 'label_map.txt')
    
    class_names = [c.strip() for c in open(label_path)]
    weight_file_path = path_join('.', 'pretrained_model', 'S3D_kinetics400.pt')

    model = S3DModel(weight_file_path, class_names)

    video_dirs = (dir for dir in os.listdir(path_sample) if os.path.isdir(os.path.join(path_sample, dir)))
    for video_idx, video_dir in enumerate(video_dirs, 1):
        video_frame_dir = path_join(path_sample, video_dir)
        print(f"#{video_idx} {video_dir}")
        model.predict(video_frame_dir)

def transform(snippet):
    ''' stack & noralization '''
    snippet = np.concatenate(snippet, axis=-1)
    snippet = torch.from_numpy(snippet).permute(2, 0, 1).contiguous().float()
    snippet = snippet.mul_(2.).sub_(255).div(255)

    return snippet.view(1,-1,3,snippet.size(1),snippet.size(2)).permute(0,2,1,3,4)

if __name__ == '__main__':
    main()
