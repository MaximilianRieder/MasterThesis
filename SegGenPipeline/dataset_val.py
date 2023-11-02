from PIL import Image
import os
from torch.utils.data import Dataset
import numpy as np
import torch
import random
from torchvision import transforms
import config_seg_gan
from torch.utils.data import DataLoader
import re

class DatasetVal(Dataset):
    def __init__(self, root_images, root_masks, videos, transform_image_z_norm=None, tranform_image_bounded=None, transform_mask=None, downsampling=True):
        self.root_images = root_images
        self.org_img_path_domain = []
        self.mask_path_fn = []
        self.complete_data = []
        
        for video in videos:
            # List with all frames and respective category
            for filename in os.listdir(root_images):
                if filename != video:
                    continue
                video_path = os.path.join(root_images, filename)
                #iterate over smoked categories
                if (filename.startswith("@") or filename.startswith(".")):
                    continue
                for smoke_category in os.listdir(video_path):
                    if smoke_category == "not_smoked" or smoke_category == "heavily_smoked" or smoke_category == "slightly_smoked":
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

                                self.org_img_path_domain.append({"path": os.path.join(not_smoked_file_path, frame), "domain": smoke_category, "frame": frame_n, "video":video})
                    
            # List with all Masks
            for filename in os.listdir(root_masks):
                if filename != video:
                    continue
                video_path = os.path.join(root_masks, filename)
                for frame_segment in os.listdir(video_path):
                    if frame_segment.endswith(".mp4") or frame_segment.startswith("@") or frame_segment.startswith("."):
                        continue
                    frame_segment_path = os.path.join(video_path, frame_segment)
                    for mask in os.listdir(frame_segment_path):
                        if mask.endswith("mask.png"):
                            result = re.search('(.*)_mask.png', mask)
                            frame_number = result.group(1)
                            self.mask_path_fn.append({"frame":frame_number, "path":os.path.join(frame_segment_path, mask), "video":video})
            
        # concat the lists with matching frame number     
        for org_data in self.org_img_path_domain:
            img_fn = org_data["frame"]
            img_domain = org_data["domain"]
            img_path_org = org_data["path"]
            img_org_video = org_data["video"]
            mask_match_path = ""

            for mask_data in self.mask_path_fn:
                fn_mask = mask_data["frame"]
                mask_path = mask_data["path"]
                mask_video= mask_data["video"]

                if (img_fn == fn_mask) and (mask_video == img_org_video):
                    mask_match_path = mask_path
                    break
            
            if mask_match_path == "":
                print("error loading mask ")
                
            self.complete_data.append({"frame":img_fn, "video": img_org_video, "domain":img_domain, "path_img":img_path_org, "path_mask":mask_match_path})
           
        self.length_dataset = len(self.complete_data)
        # cut the samples that doesnt fit into one batch anymore -> otherwise error in training        
        number_remaining = self.length_dataset % config_seg_gan.BATCH_SIZE_SEG
        if number_remaining != 0:
            self.complete_data = self.complete_data[:self.length_dataset - number_remaining]
            self.length_dataset = len(self.complete_data)

        self.transform_image_z_norm = transform_image_z_norm
        self.transform_mask = transform_mask
        self.transform_image_bounded = tranform_image_bounded

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, index):
        image_path = self.complete_data[index]["path_img"]
        mask_path = self.complete_data[index]["path_mask"]
        fn = self.complete_data[index]["frame"]
        domain = self.complete_data[index]["domain"]
        video_name = self.complete_data[index]["video"]

        image = np.array(Image.open(image_path))
        mask = np.array(Image.open(mask_path))
        
        if self.transform_image_z_norm:
            augment_image = self.transform_image_z_norm(image=image)
            image_z = augment_image["image"]
        if self.transform_image_bounded:
            augment_image = self.transform_image_bounded(image=image)
            image_bounded = augment_image["image"]
        if self.transform_mask:
            augment_mask = self.transform_mask(image=mask)
            mask = augment_mask["image"]

        return {"data_z":(image_z, mask, video_name, fn, domain), "data_bound":(image_bounded, mask, video_name, fn, domain)}
