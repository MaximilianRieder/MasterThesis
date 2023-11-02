import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
#TRAIN_DIR = "data/train"
#VAL_DIR = "data/val"
ROOT_VIDEO_PATH = "/data/home/rim36739/images/Frames/Smoke_Annotations/Videos_25fps"
OUTPUT_PATH_IMGS = "/data/home/rim36739/disk/saved_imgs/star_gan" 
VIDEOS = ["0001_216","0002_145","0003_148","0004_177","0005_201","0007_217"]
CHECKPOINT_GEN_SAVE = "/data/home/rim36739/disk/saved_models/gen.pth.tar"
CHECKPOINT_DISC_SAVE = "/data/home/rim36739/disk/saved_models/critic.pth.tar"
TENSORBOARD_PATH = "/data/home/rim36739/disk/runs/run_smoke_stargan_new_run/"

LOAD_MODEL = False
SAVE_MODEL = False
WITH_METRIC = True
DOWNSAMPLING = True
SAVE_METRICS_EPOCH = 5

BATCH_SIZE = 6
LEARNING_RATE_DISC = 2e-4
LEARNING_RATE_GEN = 2e-4
LAMBDA_IDENTITY = 4.0
LAMBDA_CYCLE = 10.0
NUM_WORKERS = 4
NUM_EPOCHS = 30
BUFFER_SIZE = 20
LAMBDA_CLS = 1
LAMBDA_GP = 10

NUM_CLASSES = 3
MAX_PIXEL_VAL = 255
ZNORM = False
img_width = 960
img_height = 540

if ZNORM:
    mean_smokeset_hs=[0.51948161, 0.38390668, 0.35970329]
    std_smokeset_hs=[0.18273795, 0.15816768, 0.15964756]
    mean_smokeset_ns=[0.51948161, 0.38390668, 0.35970329]
    std_smokeset_ns=[0.18273795, 0.15816768, 0.15964756]
    """mean_smokeset_hs=[0.50199989, 0.43386486, 0.44213915]
    std_smokeset_hs=[0.12747157, 0.1095212,  0.10627534]
    mean_smokeset_ns=[0.54536253, 0.37118833, 0.34167631]
    std_smokeset_ns=[0.18325111, 0.15968467, 0.15475203]"""
else:
    # use bounded norm for everything (normalize values between -1 and 1)
    mean_smokeset_hs=[0.5, 0.5, 0.5]
    std_smokeset_hs=[0.5, 0.5,  0.5]
    mean_smokeset_ns=[0.5, 0.5, 0.5]
    std_smokeset_ns=[0.5, 0.5, 0.5]

std_bounded_norm=[0.5, 0.5, 0.5]
mean_bounded_norm=[0.5, 0.5, 0.5]

transform = A.Compose(
    [
        A.Resize(width=img_width, height=img_height),
        A.Normalize(mean=mean_bounded_norm, std=std_bounded_norm),
        ToTensorV2(),
     ]
)