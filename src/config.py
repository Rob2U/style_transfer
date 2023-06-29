import os

# Model related

# Training hyperparameters
LEARNING_RATE = 3e-5
BATCH_SIZE = 128
EPOCHS = 10

# Dataset related
DATA_DIR = os.path.join(os.getcwd(), "data")
TRAIN_RATIO = 0.6
VAL_RATIO = 0.1
TEST_RATIO = 0.3


# Compute related
ACCELERATOR = "gpu"

# Path to the folder where the pretrained models are saved / will be saved
CHECKPOINT_PATH = os.path.join(os.getcwd(), "checkpoints")
