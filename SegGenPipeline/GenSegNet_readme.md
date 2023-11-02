# Start GenSegNet training:
start segmentation script with (example): 
- CUDA_VISIBLE_DEVICES=card_number python seg_gen_pipe.py fold_index
- CUDA_VISIBLE_DEVICES=1 python seg_gen_pipe.py 1 True
- !! remember to change the pretrained model for different folds

- for training on card 1 and with fold_index 1 (argument)
fold index determines video from list of videos all (config: ["0001_216","0002_145","0003_148","0004_177","0005_201","0007_217"])
(fold_index 1 -> 0002_145) (stating at 0) training with frames from all viedeos except video 0002_145

set for new environment:
- OUTPUT_PATH_IMGS_CYCLE = path where images are saved to (exmpl.:"/data/home/rim36739/disk/saved_imgs/cycle_in_seg_gen") 
- SAVE_MODEL_PATH = path where model parameters are saved to (exmpl.:"/data/home/rim36739/disk/saved_models/segmentation/")
- LOAD_SEG_MODEL_BASE = path to file with model parameters for a pretrained segmentation network (exmpl.:"/data/home/rim36739/g_laufwerk/- Masterarbeit/pretrained_weights/best_deeplabv3plus_resnet101_cityscapes_os16.pth.tar")
- SAVE_SEG_MASK_PATH = path where masks are saved to (exmpl.:"/data/home/rim36739/disk/saved_imgs/saved_imgs_mask/")
- TENSORBOARD_PATH = path where tensorboard files are saved to (exmpl.:"/data/home/rim36739/disk/runs/seg_gan/")
- IMAGES_ROOT_PATH = path to root folder of images (exmpl.:"/data/home/rim36739/images/Frames/Smoke_Annotations/Videos_25fps")
- MASK_ROOT_PATH =  path to root folder of masks (exmpl.:"/data/home/rim36739/images/Masks_with_binaries/classes_17/Videos_25fps")
- LOAD_PATH_SEG = path to pretrained segmentation network like: "/data/home/rim36739/disk/saved_models/segmentation/seg_klein_bl/0002_145/20/with_gen_False/lov_lossTrue/seg_model.pth.tar"
- LOAD_PATH_CYCLE = paths to pretrained cyclegan models in following structure:
    - {"G_NS":"/data/home/rim36739/disk/saved_old_results/CycleGan/Cycle_small_images/0002_145/19/gen_ns.pth.tar", 
    - "G_HS":"/data/home/rim36739/disk/saved_old_results/CycleGan/Cycle_small_images/0002_145/19/gen_hs.pth.tar", 
    - "D_NS":"/data/home/rim36739/disk/saved_old_results/CycleGan/Cycle_small_images/0002_145/19/disc_ns.pth.tar", 
    - "D_HS":"/data/home/rim36739/disk/saved_old_results/CycleGan/Cycle_small_images/0002_145/19/disc_hs.pth.tar"}

Dependencies:
- Install: pip install segmentation-models-pytorch for Lovász Loss

Used folder structure of root images:
* Videos_25fps
    - video_name (0001_216)
        - heavily_smoked
            - frame_segment (1000)
                - frame (f_1000.png)
                ...
            ...
        - not_smoked
        ...
        - slightly_smoked
        ...
    ...

Used folder structure of root masks:
* Videos_25fps
    - video_name (0001_216)
        - frame_segment (1000)
            - mask_color (1000_mask_color.png)
            - mask_color_coded (1000_mask.png)
            - mask_binary (1000_masks_binary.npz)
            ...
        ...
    ...

Used folder structure of root generated images:
* Generated_images
    - fold (fold_0001_216 -> video 0001_216 not included)
        - epoch (epoch5 -> epoch 5)
            - Videos_25fps 
                - video_name (0001_216)
                    - frame_segment (1000)
                        - frame (f_1000)
                        ...
                    ...
                ...