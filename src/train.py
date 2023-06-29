import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torch.utils.data as data
import torchvision
from torchvision import transforms

from trainer import Trainer
import os

from dataset import perform_train_val_test_split
from models.model import TorchModel
from config import (
    LEARNING_RATE,
    DATA_DIR,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
    BATCH_SIZE,
    EPOCHS,
    CHECKPOINT_PATH,
    ACCELERATOR,
)
import wandb

def train_model(model_class, train_dl, val_dl):
    # configure model
    model = model_class(
            **{ # add all model related hyperparameters here
            }
     )
        
    # configure the trainer
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    trainer = Trainer(model, optimizer, wandb.log, accelerator=ACCELERATOR) 
    trainer.train(model, train_loader=train_dl, val_loader=val_dl, epochs=EPOCHS)
    
    # save the model TODO: structure the model / checkpoint saving better
    torch.save(trainer.model.state_dict(), os.path.join(CHECKPOINT_PATH, "model.pth"))
    
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
        project="style-transfer",
        
        # track hyperparameters and run metadata
        config={
        "learning_rate": 0.02,
        "architecture": "CNN",
        "dataset": "CIFAR-100",
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,  
        "learning_rate": LEARNING_RATE,
        }
    )

if __name__ == "__main__":
    # Initialize the logger
    init_logger()

    # Ensure that all operations are deterministic on GPU (if used) for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print("Device:", ACCELERATOR)
    
    # load dataset and create dataloaders
    train_ds, val_ds, test_ds = perform_train_val_test_split(
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
    model = train_model(TorchModel, train_dl, val_dl)
    
    # test the model TODO: implement test function
