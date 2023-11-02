from PIL import Image
import os
from torch.utils.data import Dataset
import numpy as np
import torch
import random
import config_stargan as config

from torch.utils.data import DataLoader

class SmokeDataset(Dataset):
    def __init__(self, root_images, video, transform=None, downsampling=True):
        self.video_name = video
        torch.manual_seed(0)
        random.seed(0)
        np.random.seed(0)
        self.root_images = root_images
        self.not_smoked_images = []
        self.heavily_smoked_images = []
        self.slightly_smoked_images = []
        #iterate over each video
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
                            self.not_smoked_images.append({"path": os.path.join(not_smoked_file_path, frame), "label": "not"})
                if smoke_category == "heavily_smoked":
                    #iterate over frame segments
                    heavily_smoked_files_path = os.path.join(video_path, smoke_category)
                    for heavily_smoked_file in os.listdir(heavily_smoked_files_path):
                        #iterate over frames
                        if(heavily_smoked_file.startswith("@") or heavily_smoked_file.startswith(".")):
                            continue
                        heavily_smoked_file_path = os.path.join(heavily_smoked_files_path, heavily_smoked_file)
                        for frame in os.listdir(heavily_smoked_file_path):
                            self.heavily_smoked_images.append({"path": os.path.join(heavily_smoked_file_path, frame), "label": "heavily"})
                if smoke_category == "slightly_smoked":
                    #iterate over frame segments
                    slightly_smoked_files_path = os.path.join(video_path, smoke_category)
                    for slightly_smoked_file in os.listdir(slightly_smoked_files_path):
                        #iterate over frames
                        if(slightly_smoked_file.startswith("@") or slightly_smoked_file.startswith(".")):
                            continue
                        slightly_smoked_file_path = os.path.join(slightly_smoked_files_path, slightly_smoked_file)
                        for frame in os.listdir(slightly_smoked_file_path):
                            self.slightly_smoked_images.append({"path": os.path.join(slightly_smoked_file_path, frame), "label": "slightly"})


        # adjust lists to same size TODO maybe refactor first part for slightly smoked
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

        if len(self.not_smoked_images) < len(self.heavily_smoked_images):
            print("Attention more heavy smoked than not smoked -> Exited")
            random.shuffle(self.heavily_smoked_images)
            len_not_smoked_imgs = len(self.not_smoked_images)
            self.heavily_smoked_images = self.heavily_smoked_images[:len_not_smoked_imgs]
            print(len(self.heavily_smoked_images), len(self.not_smoked_images))
            
        if len(self.not_smoked_images) < len(self.slightly_smoked_images):
            print("Attention more slightly smoked than not smoked")
            random.shuffle(self.slightly_smoked_images)
            len_not_smoked_imgs = len(self.not_smoked_images)
            self.slightly_smoked_images = self.slightly_smoked_images[:len_not_smoked_imgs]
            print(len(self.slightly_smoked_images), len(self.not_smoked_images))

        # concat and shuffle lists
        self.images_list = self.slightly_smoked_images + self.not_smoked_images + self.heavily_smoked_images
        random.shuffle(self.images_list)

        self.transform = transform
        self.not_smoked_len = len(self.not_smoked_images)
        self.slightly_smoked_len = len(self.slightly_smoked_images)
        self.heavily_smoked_len = len(self.heavily_smoked_images)
        self.length_dataset = len(self.images_list)


    def __len__(self):
        return self.length_dataset

    def getframestring(self, path_string):
        path_list = path_string.split("/")
        video = path_list[8]
        frame = path_list[11]
        frame = frame[:len(frame) - 4]
        frame = frame[2:]
        return (video, frame)

    def __getitem__(self, index):
        image_path_category = self.images_list[index % self.length_dataset]
        image_path = image_path_category["path"]
        image_category = image_path_category["label"]
        
        image = np.array(Image.open(image_path).convert("RGB"))
        video_frame = self.getframestring(image_path)

        if self.transform:
            augmentations = self.transform(image=image)
            image_preprocessed = augmentations["image"]
            
        num_classes = config.NUM_CLASSES
        label = torch.zeros(1, num_classes)
        class_index = 0
        if image_category == "not":
            class_index = 0
        elif image_category == "slightly":
            class_index = 1
        elif image_category == "heavily":
            class_index = 2
        else:
            print("classification error")
        # Create one-hot encoded label tensor
        # 1 is the batch size (change)
        label[0, class_index] = 1
        label = label.squeeze()

        return {'image':image_preprocessed, 'class': label, 'class_name': image_category}, video_frame[0], video_frame[1]

