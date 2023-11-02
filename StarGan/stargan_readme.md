# Start StarGAN training:
start StarGAN script with (example): 
- CUDA_VISIBLE_DEVICES=gc_number python stargantrain.py fold_index
- CUDA_VISIBLE_DEVICES=1 python stargantrain.py 1

for training on card 1 and with fold_index 1 (argument)
fold index determines video from list of videos all (config: ["0001_216","0002_145","0003_148","0004_177","0005_201","0007_217"])
(fold_index 1 -> 0002_145) (stating at 0) training with frames from all viedeos except video 0002_145

set for new environment:
- ROOT_VIDEO_PATH = path to original frames (exmpl.:"/data/home/rim36739/images/Frames/Smoke_Annotations/Videos_25fps")
- OUTPUT_PATH_IMGS = path where generated frames should be stored (exmpl.:  "/data/home/rim36739/disk/saved_imgs/star_gan")
- CHECKPOINT_GEN_SAVE = path where gen model is saved (exmpl.: "/data/home/rim36739/disk/saved_models/gen.pth.tar")
- CHECKPOINT_DISC_SAVE = path where disc model is saved (exmpl.: "/data/home/rim36739/disk/saved_models/critic.pth.tar")
- TENSORBOARD_PATH = path where tensorboard files are saved (exmpl.: "/data/home/rim36739/disk/runs/run_smoke_stargan_new_run/")

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