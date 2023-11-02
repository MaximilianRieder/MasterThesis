import random
import torch
import config_seg_gan as config_cyclegan
import os
from utils import denormalize_manual
import numpy as np
from PIL import Image

class GenBuffer():
    # Choose buffer size as 
    def __init__(self, cycle_batch_size, for_cyclegan=True, skip_number=4):
        self.cycle_batch_size = cycle_batch_size
        self.for_cyclegan = for_cyclegan
        self.skip_number = skip_number
        self.skip_number_org = skip_number
        if self.cycle_batch_size > 0:
            self.data_ns = []
            self.data_hs = []
 
    # data as batch of (image, mask, video_name, fn, domain) with image in bounded norm
    def push_pop(self, data):

        if self.skip_number > 0:
            self.skip_number -= 1
            return {"returned_cycle":False, "cycle_data": None}
        else:
            self.skip_number = self.skip_number_org

        for idx in range(len(data[0])):
            image = data[0][idx]
            mask = data[1][idx]
            video_name = data[2][idx]
            fn = data[3][idx]
            domain = data[4][idx]

            #TODO test entpacken
            if (domain == "not_smoked") and (len(self.data_ns) < self.cycle_batch_size):
                self.data_ns.append((image, mask, video_name, fn, domain))

            elif (domain == "heavily_smoked") and (len(self.data_hs) < self.cycle_batch_size):
                self.data_hs.append((image, mask, video_name, fn, domain))

        if (len(self.data_ns) == self.cycle_batch_size) and (len(self.data_hs) == self.cycle_batch_size):
            data_ns_img_list = []
            data_ns_video_list = []
            data_ns_frame_list = []
            data_ns_mask_list = []
            data_ns_domain_list = []
            data_hs_img_list = []
            data_hs_video_list = []
            data_hs_frame_list = []
            data_hs_domain_list = []                
            data_hs_mask_list = []  
            
            # iterate over the lists simultaniously
            # each object is (image, mask, video_name, fn, domain)           
            for ns_data_object, hs_data_object in zip(self.data_ns, self.data_hs):
                (image, mask, video_name, fn, domain) = ns_data_object
                # generate data_ns
                data_ns_img_list.append(image.unsqueeze(0))
                data_ns_video_list.append(video_name)
                data_ns_frame_list.append(fn)
                data_ns_mask_list.append(mask.unsqueeze(0))

                (image, mask, video_name, fn, domain) = hs_data_object
                # generate data_ns
                data_hs_img_list.append(image.unsqueeze(0))
                data_hs_video_list.append(video_name)
                data_hs_frame_list.append(fn)            
                data_hs_domain_list.append(domain)                
                data_hs_mask_list.append(mask.unsqueeze(0))        
                # images get translated to heavy smoke
                data_ns_domain_list.append(domain)            

            data_ns_img_list = torch.cat(data_ns_img_list, dim=0)
            data_ns_mask_list = torch.cat(data_ns_mask_list, dim=0)
            data_hs_img_list = torch.cat(data_hs_img_list, dim=0)
            data_hs_mask_list = torch.cat(data_hs_mask_list, dim=0)

            self.data_ns = []
            self.data_hs = []
            
            if self.for_cyclegan:
                return {"returned_cycle":True, "seg_info": {"mask_ns": data_ns_mask_list, "video": data_ns_video_list, "frame":data_ns_frame_list, "domain":data_ns_domain_list},"cycle_data": ({'img':data_ns_img_list, 'ns_video':data_ns_video_list, 'ns_frame':data_ns_frame_list}, {'img':data_hs_img_list,'hs_video':data_hs_video_list, 'hs_frame':data_hs_frame_list})}
            else:
                return {"returned_cycle":True, "batch_seg_hs" :(data_hs_img_list, data_hs_mask_list, data_hs_video_list, data_hs_frame_list, data_hs_domain_list)}

        return {"returned_cycle":False, "cycle_data": None}