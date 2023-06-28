import os

# Training hyperparameters
NUM_CLASSES = 10
NUM_PATCHES = 64
PATCH_SIZE = 4
NUM_CHANNELS = 3

EMBED_DIM = 256 # the dimensionality of the embedded tokens / patches
HIDDEN_DIM = 512
DROPOUT = 0.2
NUM_HEADS = 8
NUM_LAYERS = 6

LEARNING_RATE = 3e-5
BATCH_SIZE = 128
MIN_EPOCHS = 1
MAX_EPOCHS = 15

# Dataset
DATA_DIR = "data/"
NUM_WORKERS = 8

# Compute related
ACCELERATOR = "gpu"
DEVICES = 1

# Path to the folder where the datasets are/should be downloaded (e.g. CIFAR10)
DATASET_PATH = os.environ.get("PATH_DATASETS", "data/")
# Path to the folder where the pretrained models are saved
CHECKPOINT_PATH = os.environ.get("PATH_CHECKPOINT", "saved_models/VisionTransformers/")
