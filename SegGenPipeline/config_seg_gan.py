import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VIDEOS_ALL = ["0001_216","0002_145","0003_148","0004_177","0005_201","0007_217"]
CUSTOM_LABEL = "SEGGENNET_last_test_vgl2"

OUTPUT_PATH_IMGS_CYCLE = "/data/home/rim36739/disk/saved_imgs/cycle_in_seg_gen" 
SAVE_MODEL_PATH = "/data/home/rim36739/disk/saved_models/segmentation/seg_gan/"
LOAD_SEG_MODEL_BASE = "/data/home/rim36739/g_laufwerk/Masterarbeit/pretrained_weights/best_deeplabv3plus_resnet101_cityscapes_os16.pth.tar"
SAVE_SEG_MASK_PATH = "/data/home/rim36739/disk/saved_imgs/saved_imgs_mask/seg_gan/"
TENSORBOARD_PATH = "/data/home/rim36739/disk/runs/seg_gan/"
IMAGES_ROOT_PATH = "/data/home/rim36739/images/Frames/Smoke_Annotations/Videos_25fps"
MASK_ROOT_PATH = "/data/home/rim36739/images/Masks_with_binaries/classes_17/Videos_25fps"
LOAD_PATH_SEG = "/data/home/rim36739/disk/saved_models/segmentation/seg_klein_bl/0002_145/20/with_gen_False/lov_lossTrue/seg_model.pth.tar"
LOAD_PATH_CYCLE = {"G_NS":"/data/home/rim36739/disk/saved_old_results/CycleGan/Cycle_small_images/0002_145/19/gen_ns.pth.tar", 
                   "G_HS":"/data/home/rim36739/disk/saved_old_results/CycleGan/Cycle_small_images/0002_145/19/gen_hs.pth.tar", 
                   "D_NS":"/data/home/rim36739/disk/saved_old_results/CycleGan/Cycle_small_images/0002_145/19/disc_ns.pth.tar", 
                   "D_HS":"/data/home/rim36739/disk/saved_old_results/CycleGan/Cycle_small_images/0002_145/19/disc_hs.pth.tar"}

# how many ns, hs ... images are loaded => trainingsteps per training
BATCH_SIZE = 3
# how many images are loaded for segmentation -> actual batch size
BATCH_SIZE_SEG = BATCH_SIZE
LEARNING_RATE_SEG = 1e-4
LEARNING_RATE_GEN = 7e-5
LEARNING_RATE_DISC = 7e-5
NUM_WORKERS = 4
NUM_EPOCHS = 30
LAMBDA_IDENTITY = 8.0
LAMBDA_CYCLE = 20.0
# denotes the influence segmentation on image generation
LAMBDA_GEN_PENALTY = 0.08
SAVE_METRICS_EPOCH = 5
LOAD_MODEL = True
LOVASZ_LOSS = True
SAVE_MODEL = True
NUM_CLASSES = 17
MAX_PIXEL_VAL = 255
EVAL_EPOCH = 5
LAMBDA_ADV_SEG_GEN_LOSS = 1

mean_smokeset_hs_z=[0.51948161, 0.38390668, 0.35970329]
std_smokeset_hs_z=[0.18273795, 0.15816768, 0.15964756]
std_bounded_norm=[0.5, 0.5, 0.5]
mean_bounded_norm=[0.5, 0.5, 0.5]

img_height = 288
img_width = 512

transformimage_znorm = A.Compose(
    [
        A.Resize(width=img_width, height=img_height, interpolation=cv2.INTER_NEAREST),
        A.Normalize(mean=[0.51948161, 0.38390668, 0.35970329], std=[0.18273795, 0.15816768, 0.15964756]),
        ToTensorV2(),
     ],
    additional_targets={"image0": "image"},
)
transformimage_nbounded = A.Compose(
    [
        A.Resize(width=img_width, height=img_height, interpolation=cv2.INTER_NEAREST),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
        ToTensorV2(),
     ],
    additional_targets={"image0": "image"},
)
transformmask = A.Compose(
    [
        A.Resize(width=img_width, height=img_height, interpolation=cv2.INTER_NEAREST),
        ToTensorV2(),
     ],
    additional_targets={"image0": "image"},
)

