import os
from datetime import datetime

# Model related

# Training hyperparameters
LEARNING_RATE = 3e-5
BATCH_SIZE = 8
EPOCHS = 1
ALPHA = 1e-6
BETA = 0.75
GAMMA = 0.1

# Dataset related
DATA_DIR = os.path.join(os.getcwd(), "data", "train2017")
STYLE_IMAGE_PATH = os.path.join(os.getcwd(), "style_images", "style1.jpeg")
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1


# Compute related
ACCELERATOR = "cpu" # "gpu" or "cpu

# Path to the folder where the pretrained models are saved / will be saved
CHECKPOINT_PATH = os.path.join(os.getcwd(), "checkpoints")

# other
RUN_NAME = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
