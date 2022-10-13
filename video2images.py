# Reference: https://github.com/aliwaqas333/VideoToImages/blob/main/src/videoToImages/videoToImages.py
import os
import cv2
from tqdm import tqdm
from pathlib import Path
import argparse
import math

parser = argparse.ArgumentParser()
parser.add_argument('--videoFolder', type=str, default="videos",
                    help='full path of path of folder containing videos')
parser.add_argument('--outFolder', type=str, default="", help='output folder to store images')
parser.add_argument('--fps', type=int, default=8, help='frames to output per second.')
parser.add_argument('--img_size', type=int, default=224, help='frames to output per second.')

class VideoToImages:
    def __init__(self, config):
        self.videoFolder = Path(f"{config.videoFolder}")
        if not os.path.exists(self.videoFolder):
            raise RuntimeError('Invalid path: %s' % self.videoFolder)

        self.config = config
        self.files = [file for file in os.listdir(self.videoFolder) if VideoToImages.is_valid_video_file(file)]
        self.outFolder = config.outFolder
        self.outfps = config.fps

        if len(config.outFolder) == 0:
            self.outFolder = os.path.join(".", "frames")
            try:
                print(f"saving images to : {self.outFolder}")
                os.mkdir(self.outFolder)
            except OSError:
                print("Creation of the directory %s failed, or it already exists." % self.outFolder)
            else:
                print("Successfully created the directory %s " % self.outFolder)
        elif not os.path.exists(self.outFolder):
            try:
                print(f"saving images to : {self.outFolder}")
                os.mkdir(self.outFolder)
            except OSError:
                print("Creation of the directory %s failed, or it already exists." % self.outFolder)
            else:
                print("Successfully created the directory %s " % self.outFolder)
        else:
            print(f"saving images to : {self.outFolder}")
        self.run()

    def run(self):
        print("Starting conversion")
        for idx, file in enumerate(self.files):
            input_video_path = os.path.join(self.videoFolder, file)
            output_frames_dirpath = os.path.join(self.outFolder, ''.join(os.path.basename(file).split(".")[:-1]))

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
                    image = cv2.resize(image, (self.config.img_size, self.config.img_size))

                cv2.imwrite(os.path.join(output_frames_dirpath, f"{i:05d}.jpg"), image)
                success, image = vidcap.read()
    
    @staticmethod
    def is_valid_video_file(file):
        '''
        Check if the file is a valid video file.
        @args file: path to the file
        @return: True if the file is a valid video file, False otherwise.
        '''
        filename = file.lower()
        return filename.endswith('.mp4') or filename.endswith('.mkv') or filename.endswith('.webm')

def main():
    VideoToImages(parser.parse_args())

if __name__ == '__main__':
    main()