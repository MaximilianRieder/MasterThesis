import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SEG_MODEL_PAR_PRETRAIN = "/data/home/rim36739/g_laufwerk/Masterarbeit/pretrained_weights/best_deeplabv3plus_resnet101_cityscapes_os16.pth.tar"
SAVE_IMGS_MASK_PATH = "/data/home/rim36739/disk/saved_imgs/saved_imgs_mask/"
SAVE_TENSORBAORD_PATH = "/data/home/rim36739/disk/runs/seg/"
ROOT_PATH_IMG = "/data/home/rim36739/images/Frames/Smoke_Annotations/Videos_25fps"
ROOT_PATH_MASKS = "/data/home/rim36739/images/Masks_with_binaries/classes_17/Videos_25fps"
ROOT_PATH_GENERATED_IMGS = "/data/home/rim36739/disk/saved_old_results/CycleGan/Correct_Baseline_22_2_23/Generated_images/"
SAVE_MODEL_PATH = "/data/home/rim36739/disk/saved_models/segmentation/"
# Denotes the epoch from which the generated images are taken
EPOCH_GENIMG = 20
SAVE_MASK = False
LOAD_MODEL = False
SAVE_MODEL = True

ALL_VIDEO_NAMES = ["0001_216","0002_145","0003_148","0004_177","0005_201","0007_217"]

# If true -> Lovasz loss is used, if false -> Cross entropy
LOVASZ_LOSS = True
# Displays custom label before save location
CUSTOM_LABEL = "segmentation_save"
# Ratio that should be established for heavy_smoked images -> 0.6 = 60% hs images
SWAP_RATIO = 0.5

BATCH_SIZE = 18
LEARNING_RATE = 1e-4
NUM_WORKERS = 4
NUM_EPOCHS = 50
NUM_CLASSES = 17

IMG_WIDTH = 912
IMG_HEIGHT = 513

transformimage = A.Compose(
    [
        A.Resize(width=IMG_WIDTH, height=IMG_HEIGHT, interpolation=cv2.INTER_NEAREST),
        A.Normalize(mean=[0.51948161, 0.38390668, 0.35970329], std=[0.18273795, 0.15816768, 0.15964756]),
        ToTensorV2(),
     ],
    additional_targets={"image0": "image"},
)
transformmask = A.Compose(
    [
        A.Resize(width=IMG_WIDTH, height=IMG_HEIGHT, interpolation=cv2.INTER_NEAREST),
        ToTensorV2(),
     ],
    additional_targets={"image0": "image"},
)