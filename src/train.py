import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torch.utils.data as data
import torchvision
from torchvision import transforms

#from torchvision.datasets import MNIST # test template

import os
import datetime

from .trainer import Trainer, configure_optimizer
from .dataset import perform_train_val_test_split, COCOImageDatset
from .models import VGG16Wrapper
from .config import (
    LEARNING_RATE,
    DATA_DIR,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
    BATCH_SIZE,
    EPOCHS,
    CHECKPOINT_PATH,
    ACCELERATOR,
    RUN_NAME,
)
import wandb

def train_model(model_class, train_dl, val_dl):
    # configure model
    model = model_class(
            **{ # add all model related hyperparameters here
            }
     )
    model = model.to(ACCELERATOR)
        
    # configure the trainer
    optimizer = configure_optimizer(model=model, learning_rate=LEARNING_RATE)
    trainer = Trainer(model, optimizer, wandb.log, accelerator=ACCELERATOR) 
    trainer.train(train_loader=train_dl, val_loader=val_dl, epochs=EPOCHS)
    
    # save the best model to checkpoint directory
    torch.save(trainer.best_model["model_state"], os.path.join(CHECKPOINT_PATH, f"{model_class}--{RUN_NAME}.pth"))
    
    return trainer.model

# loads a pretrained model
def run_pretrained_model(pretrained_filename, model_class):
    print("Loading pretrained model from %s..." % pretrained_filename)
    # Automatically loads the model with the saved hyperparameters
    model = model_class.load_from_checkpoint(pretrained_filename)
    
    return model

def init_logger():
    wandb.init(
        # set the wandb project where this run will be logged
        entity="robert-weeke",
        project="style-transfer",
        name=RUN_NAME,
        # track hyperparameters and run metadata
        config={
        "learning_rate": LEARNING_RATE,
        "architecture": "CNN",
        "dataset": "MNIST-TEST",
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,  
        "learning_rate": LEARNING_RATE,
        }
    )

if __name__ == "__main__":
    # Initialize the logger
    init_logger()

    # Ensure that all operations are deterministic on GPU (if used) for reproducibility
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    print("Device:", ACCELERATOR)
    
    # load dataset and create dataloaders
    train_ds, val_ds, test_ds = perform_train_val_test_split(
        COCOImageDatset,
        DATA_DIR,
        TRAIN_RATIO,
        VAL_RATIO, 
        TEST_RATIO
    )
    
    # create dataloaders
    train_dl = data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_dl = data.DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_dl = data.DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    
    # train the model
    model = train_model(VGG16Wrapper, train_dl, val_dl)
    
    # test the model TODO: implement test function
