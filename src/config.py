import os
from datetime import datetime

# Model related

# Training hyperparameters
LEARNING_RATE = 3e-5
BATCH_SIZE = 32
EPOCHS = 1

# Dataset related
DATA_DIR = os.path.join(os.getcwd(), "data")
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1


# Compute related
ACCELERATOR = "cuda" # "gpu" or "cpu

# Path to the folder where the pretrained models are saved / will be saved
CHECKPOINT_PATH = os.path.join(os.getcwd(), "checkpoints")

# other
RUN_NAME = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
