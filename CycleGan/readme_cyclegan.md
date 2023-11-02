# Start CycleGAN training:
start CycleGAN script with (example): 
- CUDA_VISIBLE_DEVICES=1 python cyclegantrain.py fold_index
- CUDA_VISIBLE_DEVICES=1 python cyclegantrain.py 1

for training on card 1 and with fold_index 1 (argument)
fold index determines video from list of videos all (config: ["0001_216","0002_145","0003_148","0004_177","0005_201","0007_217"])
(fold_index 1 -> 0002_145) (stating at 0) training with frames from all viedeos except video 0002_145

set for new environment:
- ROOT_VIDEO_PATH: path to original frames (exmpl.:"/data/home/rim36739/images/Frames/Smoke_Annotations/Videos_25fps")
- OUTPUT_PATH_IMGS: path where generated frames should be stored (exmpl.: "/data/home/rim36739/disk/saved_imgs/test")  
- SAVE_MODEL_PATH: path where the models are saved (exmpl.:"/data/home/rim36739/disk/saved_models/cyclegan/")
- SAVE_TENSOR_BOARD_PATH: path where the tensorboard runs are saved (exmpl.:"/data/home/rim36739/disk/runs/run_smoke_cycle_new_run/")
- SAVE_METRICS_EPOCH: denotes when metrics are calculated
- SAVE_MODEL_EPOCH:  denotes when model is saved

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