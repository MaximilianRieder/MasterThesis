import random
import torch

class ImageBuffer():
    # takes half of the batchsize (images.size()) from images and other half from buffer
    # afterwards swap out random images in buffer with other half of images

    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        if self.buffer_size > 0:
            self.num_imgs = 0
            self.images_stored = []
            self.videos_stored = []
            self.frames_stored = []
 
    def push_pop(self, images, videos_list, frames_list):
        # images: images from batch
        # list of videos and frame numbers respective to images (images -> [img1, img2,...]; videos_list -> [vid1, vid2, ...])
        
        if self.buffer_size == 0:
            return images, videos_list, frames_list
        return_images = []
        return_videos = []
        return_frames = []
        half_idx = int(images.size(0) / 2) 
        for idx, image in enumerate(images):
            image = torch.unsqueeze(image, 0)
            if self.num_imgs < self.buffer_size:
                self.num_imgs = self.num_imgs + 1
                self.images_stored.append(image)
                self.videos_stored.append(videos_list[idx])
                self.frames_stored.append(frames_list[idx])
                return_images.append(image)
                return_videos.append(videos_list[idx])
                return_frames.append(frames_list[idx])
            else:
                if idx < half_idx:
                    return_images.append(image)
                    return_videos.append(videos_list[idx])
                    return_frames.append(frames_list[idx])
                else:
                    random_id = random.randint(0, self.buffer_size - 1)
                    tmp_img = self.images_stored[random_id].clone()
                    tmp_video = self.videos_stored[random_id]
                    tmp_frame = self.frames_stored[random_id]
                    self.images_stored[random_id] = image
                    self.videos_stored[random_id] = videos_list[idx]
                    self.frames_stored[random_id] = frames_list[idx]
                    return_images.append(tmp_img)
                    return_videos.append(tmp_video)
                    return_frames.append(tmp_frame)

        return_images = torch.cat(return_images, 0)
        return return_images, return_videos, return_frames