'''
Using OpenCV takes a mp4 video and produces a number of images.

Requirements
----
You require OpenCV 3.2 to be installed.

Run
----
Open the main.py and edit the path to the video. Then run:
$ python main.py

Which will produce a folder called data with the images. There will be 2000+ images for example.mp4.
'''
import cv2
import numpy as np
import os
import numpy as np
import torch as th
import torch.utils.data as data
from PIL import Image
import os
import pickle
from scipy import signal
from sconv.functional.sconv import spherical_conv
from tqdm import tqdm
import numbers
import cv2
from functools import lru_cache
from random import Random


        vset = set()
        for vid in tqdm(os.listdir(root), desc='scanning dir'):
            if os.path.isdir(os.path.join(root, vid)):
                vset.add(vid)

        print('{} videos found.'.format(len(vset)))
        

        self.data = []
        self.target = []
        for vid in tqdm(vset, desc='video'):
            obj_path = os.path.join(root, vid)
            fcnt = 0
            for frame in tqdm(os.listdir(obj_path), desc='frame({})'.format(vid)):
                if frame.endswith('.jpg'):
                    fid = frame[:-4]
                    if fid not in self.vinfo[vid].keys():
                        print('warn: video {}, frame {} have no gt, abandoned.')
                        continue
                    fcnt += 1
                    if fcnt >= frame_interval:
                        self.data.append(os.path.join(obj_path, frame))
                        self.target.append(self.vinfo[vid][fid])
                        fcnt = 0