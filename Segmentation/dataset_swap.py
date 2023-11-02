from PIL import Image
import os
from torch.utils.data import Dataset
import numpy as np
import torch
import random
from torchvision import transforms
import config_seg
from torch.utils.data import DataLoader
import re

class DatasetSegSmokeSwap(Dataset):
    def __init__(self, root_images, root_masks, videos, root_generated=None, transform_image=None, transform_mask=None, downsampling=True, swappable_domains=["not_smoked"], with_swap=True, ratio=0.5):
        self.root_generated = root_generated
        self.root_images = root_images
        self.org_img_path_domain = []
        self.mask_path_fn = []
        self.gen_img_path_frame = []
        self.complete_data = []
        self.not_smoked_frame_count = 0
        self.slightly_smoked_frame_count = 0
        self.heavily_smoked_frame_count = 0
        self.swap_number = 0
        self.swap_indices = []
        self.with_swap = with_swap
        
        for video in videos:
            # List with all frames and respective category / frame 
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
                                swappable = False
                                if smoke_category in swappable_domains:
                                    swappable = True
                                if smoke_category == "not_smoked":
                                    self.not_smoked_frame_count += 1
                                if smoke_category == "heavily_smoked":
                                    self.heavily_smoked_frame_count += 1
                                if smoke_category == "slightly_smoked":
                                    self.slightly_smoked_frame_count += 1
                                self.org_img_path_domain.append({"path": os.path.join(not_smoked_file_path, frame), "domain": smoke_category, "frame": frame_n, "video":video, "swappable": swappable})
                    
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
                            
            # List with all generated images
            for filename in os.listdir(root_generated):
                if filename != video:
                    continue
                video_path = os.path.join(root_generated, filename)
                for frame_segment in os.listdir(video_path):
                    frame_segment_path = os.path.join(video_path, frame_segment)
                    for frame_file in os.listdir(frame_segment_path):
                        if(frame_file.startswith("@") or frame_file.startswith(".")):
                            continue
                        frame_file_path = os.path.join(frame_segment_path, frame_file)
                        frame_n = frame_file[:len(frame_file) - 4]
                        frame_n = frame_n[2:]
                        if downsampling and int(frame_n) % 25 != 0:
                            continue
                        self.gen_img_path_frame.append({"path": frame_file_path, "frame": frame_n, "video":video})
            
        # concat the lists with matching frame number
        # complete data -> {"frame", "video", "domain", "path_img", "path_mask", "path_gen_img", "swappable"}
        for org_data in self.org_img_path_domain:
            img_fn = org_data["frame"]
            img_domain = org_data["domain"]
            img_path_org = org_data["path"]
            img_org_video = org_data["video"]
            swappable = org_data["swappable"]
            mask_match_path = ""
            gen_match_path = ""

            for mask_data in self.mask_path_fn:
                fn_mask = mask_data["frame"]
                mask_path = mask_data["path"]
                mask_video= mask_data["video"]

                if (img_fn == fn_mask) and (mask_video == img_org_video):
                    mask_match_path = mask_path
                    break
            
            if swappable:
                for generated_data in self.gen_img_path_frame:
                    fn_gen = generated_data["frame"]
                    gen_img_path = generated_data["path"]
                    gen_video = generated_data["video"]
                    if (img_fn == fn_gen) and (gen_video == img_org_video):
                        gen_match_path = gen_img_path
                        break
            
            if (gen_match_path == "" and swappable) and mask_match_path != "" :
                swappable = False
            elif mask_match_path == "":
                print("error loading mask ")
            self.complete_data.append({"frame":img_fn, "video": img_org_video, "domain":img_domain, "path_img":img_path_org, "path_mask":mask_match_path, "path_gen_img":gen_match_path, "swappable": swappable})
                
        if with_swap:
            # calculate ratio (not smoked - heavily smoked) and how many images have to be swapped to reach the defined ratio
            if (self.not_smoked_frame_count / self.heavily_smoked_frame_count) > 1:
                heavily_smoked_target_count = (self.not_smoked_frame_count + self.heavily_smoked_frame_count) * ratio
                self.swap_number = heavily_smoked_target_count - self.heavily_smoked_frame_count
                print(f"ns:{self.not_smoked_frame_count} hs:{self.heavily_smoked_frame_count} sw_n:{self.swap_number}")
            else:
                print("more heavy smoked than not smoked -> check data")
           
        self.length_dataset = len(self.complete_data)
        # cut the samples that doesnt fit into one batch anymore -> otherwise error in training
        number_remaining = self.length_dataset % config_seg.BATCH_SIZE
        if number_remaining != 0:
            self.complete_data = self.complete_data[:self.length_dataset - number_remaining]
            self.length_dataset = len(self.complete_data)

        self.transform_image = transform_image
        self.transform_mask = transform_mask

        if self.with_swap:
            self.set_swap_indices()

    def __len__(self):
        return self.length_dataset
    
    def print_sw_idc(self):
        print(self.swap_indices)
    
    # set this after each epoch to change the swapped indices
    def set_swap_indices(self):
        #sets the indices for swapping out the images during training
        swap_true_indices = [i for i, d in enumerate(self.complete_data) if d["swappable"]]

        swap_indices = random.sample(swap_true_indices, k=int(self.swap_number))
        self.swap_indices = swap_indices

    def __getitem__(self, index):
        image_path = self.complete_data[index]["path_img"]
        mask_path = self.complete_data[index]["path_mask"]
        fn = self.complete_data[index]["frame"]
        domain = self.complete_data[index]["domain"]
        video_name = self.complete_data[index]["video"]
        if self.with_swap:
            if index in self.swap_indices:
                image_path = self.complete_data[index]["path_gen_img"]

        image = np.array(Image.open(image_path))
        mask = np.array(Image.open(mask_path))
        
        if self.transform_image:
            augment_image = self.transform_image(image=image)
            image = augment_image["image"]
        if self.transform_mask:
            augment_mask = self.transform_mask(image=mask)
            mask = augment_mask["image"]
        
        return image, mask, video_name, fn, domain

