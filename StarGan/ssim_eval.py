from PIL import Image
import os
from torch.utils.data import Dataset
import numpy as np
import torch
import random
from torchvision import transforms
#import config_seg
from torch.utils.data import DataLoader
import re
import average_meter
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from average_meter import AverageMeter
from torchmetrics import StructuralSimilarityIndexMeasure
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# This file was just used to evaluate the ssim, while comparing original hs and ss images with generated

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
            # List with all frames and respective category sorted by frame 
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
                                    self.org_img_path_domain.append({"path": os.path.join(not_smoked_file_path, frame), "domain": smoke_category, "frame": frame_n, "video":video, "swappable": swappable})
                                    self.heavily_smoked_frame_count += 1
                                if smoke_category == "slightly_smoked":
                                    self.org_img_path_domain.append({"path": os.path.join(not_smoked_file_path, frame), "domain": smoke_category, "frame": frame_n, "video":video, "swappable": swappable})
                                    self.slightly_smoked_frame_count += 1
            #self.org_img_path_domain.sort(key=lambda x: x["path"])
                    
                            
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
                        # new
                        result = re.search(r'f_(.*?)_input', frame_file)
                        frame_n = result.group(1)
                        # new end
                        """frame_n = frame_file[:len(frame_file) - 4]
                        frame_n = frame_n[2:]"""
                        if downsampling and int(frame_n) % 25 != 0:
                            continue
                        self.gen_img_path_frame.append({"path": frame_file_path, "frame": frame_n, "video":video})

                
        self.org_img_path_domain = self.org_img_path_domain[:len(self.gen_img_path_frame)]
        self.length_dataset = len(self.org_img_path_domain)


    def __len__(self):
        return self.length_dataset


    def __getitem__(self, index):
        image_org_path = self.org_img_path_domain[index]["path"]
        image_gen_path = self.gen_img_path_frame[index]["path"]

        transform = A.Compose(
        [
            A.Resize(width=960, height=540),
            ToTensorV2(),
        ]
        )

        image_org = np.array(Image.open(image_org_path))
        image_gen = np.array(Image.open(image_gen_path))
        

        image_org = transform(image=image_org)
        image_org = image_org["image"]

        image_gen = transform(image=image_gen)
        image_gen = image_gen["image"]
        
        return image_org, image_gen


def test():
    print("test")

    fold1= f"/data/home/rim36739/disk/saved_old_results/StarGan/basline_20_3/star_gan/fold_0001_216/Smoked_frames_lr_disc0.0002_LR_gen_0.0002_epoch"
    fold2= f"/data/home/rim36739/disk/saved_old_results/StarGan/basline_20_3/star_gan/fold_0002_145/Smoked_frames_lr_disc0.0002_LR_gen_0.0002_epoch"
    fold3= f"/data/home/rim36739/disk/saved_old_results/StarGan/basline_20_3/star_gan/fold_0003_148/Smoked_frames_lr_disc0.0002_LR_gen_0.0002_epoch"
    fold4= f"/data/home/rim36739/disk/saved_old_results/StarGan/basline_20_3/star_gan/fold_0004_177/Smoked_frames_lr_disc0.0002_LR_gen_0.0002_epoch"
    fold5= f"/data/home/rim36739/disk/saved_old_results/StarGan/basline_20_3/star_gan/fold_0005_201/Smoked_frames_lr_disc0.0002_LR_gen_0.0002_epoch"
    fold6= f"/data/home/rim36739/disk/saved_old_results/StarGan/basline_20_3/star_gan/fold_0007_217/Smoked_frames_lr_disc0.0002_LR_gen_0.0002_epoch"
    folds = [fold1, fold2, fold3, fold4, fold5, fold6]
    videos_all = ["0001_216","0002_145","0003_148","0004_177","0005_201","0007_217"]
    for idx,fold_name in enumerate(folds):
        writer_train = SummaryWriter(f"/data/home/rim36739/disk/runs/star_ssim/fold_{idx}")
        count = 5
        while count <= 30:
            root_gen = fold_name + f"{count}/Videos_25fps"
            dataset_train = DatasetSegSmokeSwap(
            root_images="/data/home/rim36739/images/Frames/Smoke_Annotations/Videos_25fps",
            root_masks="/data/home/rim36739/images/Masks_with_binaries/classes_17/Videos_25fps",
            root_generated=root_gen,
            videos=videos_all, 
            with_swap=True
            )    
            loader_val = DataLoader(
            dataset_train,
            batch_size=20,
            shuffle=True,
            num_workers=4,
            pin_memory=True
            )
            ssim = StructuralSimilarityIndexMeasure().to("cuda")
            ssim_meter = AverageMeter()
            for image_org, image_gen in tqdm(loader_val):
                ssim_hs = ssim(image_org.to("cuda").float(), image_gen.to("cuda").float())
                ssim_meter.update(ssim_hs)
            writer_train.add_scalar("ssim", ssim_meter.avg, count)
            count += 5

if __name__ == "__main__":
    test()
