# Desc: Configuration file for the project

import os
from datetime import datetime



# Training hyperparameters
LEARNING_RATE = 1e-3
BATCH_SIZE = 16
EPOCHS = 1
ALPHA = 1
BETA = 25
GAMMA = 1e-5

# Dataset related
DATA_DIR = os.path.join(os.getcwd(), "data", "train2017")
# DATA_DIR = os.path.join(os.getcwd(), "data" ,"img_align_celeba", "img_align_celeba")
STYLE_IMAGE_PATH = os.path.join(os.getcwd(), "style_images", "style5.jpg")
TRAIN_RATIO = 1.0
VAL_RATIO = 0
TEST_RATIO = 0


# Compute related
ACCELERATOR = "cpu" # "gpu" or "cpu

# Path to the folder where the pretrained models are saved / will be saved
CHECKPOINT_PATH = os.path.join(os.getcwd(), "checkpoints")

# other
RUN_NAME = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "vanGogh--up--in--vgg16--1-25-1e-5--allreflect--faces"
