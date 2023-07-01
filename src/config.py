import os

# Model related

# Training hyperparameters
LEARNING_RATE = 3e-5
BATCH_SIZE = 16
EPOCHS = 10

# Dataset related
DATA_DIR = os.path.join(os.getcwd(), "data")
TRAIN_RATIO = 0.6
VAL_RATIO = 0.1
TEST_RATIO = 0.3


# Compute related
ACCELERATOR = "cpu" # "gpu" or "cpu

# Path to the folder where the pretrained models are saved / will be saved
CHECKPOINT_PATH = os.path.join(os.getcwd(), "checkpoints")
