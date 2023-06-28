import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torchvision
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from torchvision import transforms

from dataset import LightningDataModule
from models.model import TorchLightningModel
from config import *

import wandb
from pytorch_lightning.loggers import WandbLogger

def train_model(dm, trainer, logger, model_class):
    model = model_class(
            **{
                "learning_rate": LEARNING_RATE,
                "embed_dim": EMBED_DIM,
                "hidden_dim": HIDDEN_DIM,
                "num_heads": NUM_HEADS,
                "num_layers": NUM_LAYERS,
                "patch_size": PATCH_SIZE,
                "num_channels": NUM_CHANNELS,
                "num_patches": NUM_PATCHES,
                "num_classes": NUM_CLASSES,
                "dropout": DROPOUT,
            }
     )
    #logger.watch(model, log='gradients', log_freq=100)
    trainer.fit(model, dm)
    #load best checkpoint after training
    model = model_class.load_from_checkpoint(
        trainer.checkpoint_callback.best_model_path
    )
    return model

# loads a pretrained model
def run_pretrained_model(pretrained_filename, model_class):
    print("Found pretrained model at %s, loading..." % pretrained_filename)
    # Automatically loads the model with the saved hyperparameters
    model = model_class.load_from_checkpoint(pretrained_filename)
    
    return model

def run_model():
    wandb_logger = WandbLogger(project='ViT-CIFAR10')
    
    wandb_logger.log_hyperparams({
        "batch_size": BATCH_SIZE,  
        "learning_rate": LEARNING_RATE,
        "num_epochs": MAX_EPOCHS,
        "num_layers": NUM_LAYERS,
        "num_heads": NUM_HEADS,
        "embed_dim": EMBED_DIM,
        "hidden_dim": HIDDEN_DIM,
        "dropout": DROPOUT,
    })
    
    dm = LightningDataModule(
        data_dir=DATA_DIR,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        dataset_path=DATASET_PATH,
    )
    trainer = L.Trainer(
        default_root_dir=os.path.join(CHECKPOINT_PATH, "ViT"),
        accelerator=ACCELERATOR,
        devices=DEVICES,
        min_epochs=MIN_EPOCHS,
        max_epochs=MAX_EPOCHS,
        enable_checkpointing=True,
        logger=wandb_logger,
    )
    
    pretrained_filename = os.path.join(CHECKPOINT_PATH, "ViT.ckpt")

    model_class = TorchLightningModel
    
    if os.path.isfile(pretrained_filename):
        model = run_pretrained_model(pretrained_filename, model_class)
    else:
        model = train_model(dm, trainer, wandb_logger, model_class)
    
    results = trainer.test(model, datamodule=dm)
    
    return model, results

if __name__ == "__main__":
    # Setting the seed
    L.seed_everything(42)

    # Ensure that all operations are deterministic on GPU (if used) for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print("Device:", ACCELERATOR)
    model, results = run_model()
