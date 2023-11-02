from PIL import Image
import os
from torch.utils.data import Dataset
import numpy as np
import torch
import re
import random

from torch.utils.data import DataLoader

class SmokeDataset(Dataset):
    def __init__(self, root_images, video, transform_hs=None, transform_ns=None, downsampling=True):
        self.video_name = video
        torch.manual_seed(0)
        random.seed(0)
        np.random.seed(0)
        self.root_images = root_images
        self.not_smoked_images = []
        self.heavily_smoked_images = []
        
        # get all paths for hs and ns (every 25th) images respectively
        for filename in os.listdir(root_images):
            if filename != video:
                continue
            video_path = os.path.join(root_images, filename)
            #iterate over smoked categories
            if (filename.startswith("@") or filename.startswith(".")):
                continue
            for smoke_category in os.listdir(video_path):
                if smoke_category == "not_smoked":
                    #iterate over frame segments
                    not_smoked_files_path = os.path.join(video_path, smoke_category)
                    for not_smoked_file in os.listdir(not_smoked_files_path):
                        if(not_smoked_file.startswith("@") or not_smoked_file.startswith(".")):
                            continue
                        #iterate over frames
                        not_smoked_file_path = os.path.join(not_smoked_files_path, not_smoked_file)
                        for frame in os.listdir(not_smoked_file_path):
                            if(frame.startswith("@")):
                                continue
                            frame_n = frame[:len(frame) - 4]
                            frame_n = frame_n[2:]
                            if downsampling and int(frame_n) % 25 != 0:
                                continue
                            self.not_smoked_images.append(os.path.join(not_smoked_file_path, frame))
                if smoke_category == "heavily_smoked":
                    #iterate over frame segments
                    heavily_smoked_files_path = os.path.join(video_path, smoke_category)
                    for heavily_smoked_file in os.listdir(heavily_smoked_files_path):
                        #iterate over frames
                        if(heavily_smoked_file.startswith("@") or heavily_smoked_file.startswith(".")):
                            continue
                        heavily_smoked_file_path = os.path.join(heavily_smoked_files_path, heavily_smoked_file)
                        for frame in os.listdir(heavily_smoked_file_path):
                            self.heavily_smoked_images.append(os.path.join(heavily_smoked_file_path, frame))

        # adjust lists to same size (add more hs images) when less hs than ns
        if len(self.not_smoked_images) > len(self.heavily_smoked_images):
            while(len(self.not_smoked_images) > len(self.heavily_smoked_images)):
                random.shuffle(self.not_smoked_images)
                random.shuffle(self.heavily_smoked_images)
                len_heavily_smoked_imgs = len(self.heavily_smoked_images)
                len_not_smoked_imgs = len(self.not_smoked_images)
                len_diff = len_not_smoked_imgs - len_heavily_smoked_imgs
                self.heavily_smoked_images = self.heavily_smoked_images + (self.heavily_smoked_images[:len_diff])
                #self.not_smoked_images = self.not_smoked_images[:len_heavily_smoked_imgs]
            print(len(self.heavily_smoked_images), len(self.not_smoked_images))

        # adjust lists to same size (remove hs) when less ns than hs (usual case because just every 25th ns img)
        if len(self.not_smoked_images) < len(self.heavily_smoked_images):
            print("Attention more heavy smoked than not smoked")
            random.shuffle(self.heavily_smoked_images)
            len_not_smoked_imgs = len(self.not_smoked_images)
            self.heavily_smoked_images = self.heavily_smoked_images[:len_not_smoked_imgs]
            print(len(self.heavily_smoked_images), len(self.not_smoked_images))


        self.transform_hs = transform_hs
        self.transform_ns = transform_ns
        self.not_smoked_len = len(self.not_smoked_images)
        self.heavily_smoked_len = len(self.heavily_smoked_images)
        self.length_dataset = max(self.not_smoked_len, self.heavily_smoked_len)


    def __len__(self):
        return self.length_dataset

    def getframestring(self, path_string):
        path_list = path_string.split("/")
        video = path_list[-4]
        frame = path_list[-1]
        frame = frame[:len(frame) - 4]
        frame = frame[2:]
        return (video, frame)

    def __getitem__(self, index):
        ##todo maybe other function here because some examples are used more often than others
        not_smoked_img_path = self.not_smoked_images[index % self.not_smoked_len]
        heavily_smoked_img_path = self.heavily_smoked_images[index % self.heavily_smoked_len]

        not_smoked_img = np.array(Image.open(not_smoked_img_path).convert("RGB"))
        heavily_smoked_img = np.array(Image.open(heavily_smoked_img_path).convert("RGB"))

        not_smoked_video_frame = self.getframestring(not_smoked_img_path)
        heavily_smoked_video_frame = self.getframestring(heavily_smoked_img_path)

        if self.transform_hs and self.transform_ns:
            augmentations = self.transform_hs(image=heavily_smoked_img)
            heavily_smoked_img = augmentations["image"]
            
            augmentations = self.transform_ns(image=not_smoked_img)
            not_smoked_img = augmentations["image"]

        return {'ns_img': not_smoked_img,'ns_video':not_smoked_video_frame[0], 'ns_frame':not_smoked_video_frame[1]}, {'hs_img':heavily_smoked_img, 'hs_video':heavily_smoked_video_frame[0], 'hs_frame':heavily_smoked_video_frame[1]}
