# Start Segmentation training:
start segmentation script with (example): 
- CUDA_VISIBLE_DEVICES=card_number python segmentationtrain.py fold_index with_gen_images
- CUDA_VISIBLE_DEVICES=1 python segmentationtrain.py 1 True

- for training on card 1 and with fold_index 1 (argument)
fold index determines video from list of videos all (config: ["0001_216","0002_145","0003_148","0004_177","0005_201","0007_217"])
(fold_index 1 -> 0002_145) (stating at 0) training with frames from all viedeos except video 0002_145
- Boolean with_gen_images determines if images should be swapped or not

set for new environment:
- SEG_MODEL_PAR_PRETRAIN: path to file with model parameters for a pretrained segmentation network (exmpl.:"/data/home/rim36739/g_laufwerk/Masterarbeit/pretrained_weights/best_deeplabv3plus_resnet101_cityscapes_os16.pth.tar")
- SAVE_IMGS_MASK_PATH: path where masks are saved to (exmpl.:"/data/home/rim36739/disk/saved_imgs/saved_imgs_mask/")
- SAVE_TENSORBAORD_PATH: path where tensorboard files are saved to (exmpl.:"/data/home/rim36739/disk/runs/seg/")
- ROOT_PATH_IMG: path to root folder of images (exmpl.:"/data/home/rim36739/images/Frames/Smoke_Annotations/Videos_25fps")
- ROOT_PATH_MASKS: path to root folder of masks (exmpl.:"/data/home/rim36739/images/Masks_with_binaries/classes_17/Videos_25fps")
- ROOT_PATH_GENERATED_IMGS: path to root folder of generated images (exmpl.:"/data/home/rim36739/disk/saved_old_results/CycleGan/Correct_Baseline_22_2_23/Generated_images/")
- SAVE_MODEL_PATH: path where model parameters are saved to (exmpl.:"/data/home/rim36739/disk/saved_models/segmentation/")

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