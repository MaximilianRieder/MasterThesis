import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 2
ROOT_VIDEO_PATH = "/data/home/yis38803/images/DeepMIC/Frames/Smoke_Annotations/Videos_25fps"
OUTPUT_PATH_IMGS = "/data/home/yis38803/disk/saved_imgs/test" 
SAVE_MODEL_PATH = "/data/home/yis38803/disk/saved_models/cyclegan/"
SAVE_TENSOR_BOARD_PATH = "/data/home/yis38803/disk/runs/run_smoke_cycle_new_run/"
SAVE_METRICS_EPOCH = 5
SAVE_MODEL_EPOCH = 5
LOAD_MODEL = False
SAVE_MODEL = True
WITH_METRIC = True

ALL_VIDEO_NAMES = ["0001_216","0002_145"]#,"0003_148","0004_177","0005_201","0007_217"]

LEARNING_RATE_DISC = 1e-4
LEARNING_RATE_GEN = 1e-4
LAMBDA_IDENTITY = 5.0
LAMBDA_CYCLE = 10.0
NUM_WORKERS = 4
NUM_EPOCHS = 30
BUFFER_SIZE = 20

MAX_PIXEL_VAL = 255
image_width = 256
image_height = 144
# use bounded norm for everything (normalize values between -1 and 1)
mean_smokeset_hs=[0.5, 0.5, 0.5]
std_smokeset_hs=[0.5, 0.5,  0.5]
mean_smokeset_ns=[0.5, 0.5, 0.5]
std_smokeset_ns=[0.5, 0.5, 0.5]

transform_hs = A.Compose(
    [
        A.Resize(width=image_width, height=image_height),
        A.Normalize(mean=mean_smokeset_hs, std=std_smokeset_hs),
        ToTensorV2(),
     ]
)

transform_ns = A.Compose(
    [
        A.Resize(width=image_width, height=image_height),
        A.Normalize(mean=mean_smokeset_ns, std=std_smokeset_ns),
        ToTensorV2(),
     ]
)