import os
from datetime import datetime

# Model related

# Training hyperparameters
LEARNING_RATE = 1e-3
BATCH_SIZE = 4
EPOCHS = 2
ALPHA = 1e-3
BETA = 1
GAMMA = 0 # 1e-6

# Dataset related
DATA_DIR = os.path.join(os.getcwd(), "data","train2017" ,"train2017")
STYLE_IMAGE_PATH = os.path.join(os.getcwd(), "style_images", "style5.jpg")
TRAIN_RATIO = 1
VAL_RATIO = 0
TEST_RATIO = 0


# Compute related
ACCELERATOR = "cuda" # "gpu" or "cpu

# Path to the folder where the pretrained models are saved / will be saved
CHECKPOINT_PATH = os.path.join(os.getcwd(), "checkpoints")

# other
RUN_NAME = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
