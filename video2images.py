# Reference: https://github.com/aliwaqas333/VideoToImages/blob/main/src/videoToImages/videoToImages.py
from mimetypes import init
import os
import cv2
from tqdm import tqdm
from pathlib import Path
import math
import pandas as pd


class Config:
    def __init__(self, video_folder, csv_path, frame_folder) -> None:
        self.video_folder = video_folder
        self.frame_folder = frame_folder
        self.fps = 8
        self.img_size = 224
        self.video_set = set(pd.read_csv(csv_path)['video_id'].to_list())


class VideoToImages:
    def __init__(self, config: Config):
        self.video_folder = Path(f"{config.video_folder}")
        if not os.path.exists(self.video_folder):
            raise RuntimeError('Invalid path: %s' % self.video_folder)

        self.config = config
        self.files = [file for file in os.listdir(
            self.video_folder) if self.is_valid_video_file(file)]
        self.out_folder = config.frame_folder
        self.outfps = config.fps
        self.video_set = config.video_set

        # if len(config.outFolder) == 0:
        #     self.outFolder = os.path.join(".", "frames")
        #     try:
        #         print(f"saving images to : {self.outFolder}")
        #         os.mkdir(self.outFolder)
        #     except OSError:
        #         print("Creation of the directory %s failed, or it already exists." % self.outFolder)
        #     else:
        #         print("Successfully created the directory %s " % self.outFolder)
        if not os.path.exists(self.out_folder):
            try:
                print(f"saving images to : {self.out_folder}")
                os.mkdir(self.out_folder)
            except OSError:
                print(
                    "Creation of the directory %s failed, or it already exists." % self.out_folder)
            else:
                print("Successfully created the directory %s " %
                      self.out_folder)
        else:
            print(f"saving images to : {self.out_folder}")
        self.run()

    def run(self):
        print("Starting conversion")
        for idx, file in enumerate(self.files):
            input_video_path = os.path.join(self.video_folder, file)
            output_frames_dirpath = os.path.join(
                self.out_folder, ''.join(os.path.basename(file).split(".")[:-1]))

            # Create the per video output directory if it does not exist
            if not os.path.exists(output_frames_dirpath):
                os.mkdir(output_frames_dirpath)

            vidcap = cv2.VideoCapture(input_video_path)
            sourcefps = vidcap.get(cv2.CAP_PROP_FPS)
            # If the fps requested is bigger than source video fps, then use source fps
            if self.outfps > sourcefps:
                self.outfps = 0

            print(f"[{idx+1}/{len(self.files)}]converting file: {file}")
            frameCount = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

            skip = 0
            if self.outfps > 0:
                vidcap.set(cv2.CAP_PROP_FPS, self.outfps)

                # Estimating total frames to extract
                frameCount = math.floor((frameCount / sourcefps) * self.outfps)
                skip = int(1000 / self.outfps)

            success, image = vidcap.read()
            for i in tqdm(range(frameCount)):
                if not success:
                    break
                
                if skip > 0:
                    vidcap.set(cv2.CAP_PROP_POS_MSEC, (i * skip))

                if self.config.img_size:
                    image = cv2.resize(
                        image, (self.config.img_size, self.config.img_size))

                cv2.imwrite(os.path.join(
                    output_frames_dirpath, f"{i:05d}.jpg"), image)
                success, image = vidcap.read()

    def is_valid_video_file(self, file):
        '''
        Check if the file is a valid video file.
        @args file: path to the file
        @return: True if the file is a valid video file, False otherwise.
        '''
        name, ext = file.split(' ')
        
        ext: str = ext.lower()
        return ext in {'mp4','mkv', 'webm'} and name[2:] in self.video_set


def convert(video_folder: str, csv_path: str, frame_folder: str):
    config = Config(video_folder, csv_path, frame_folder)
    VideoToImages(config)
