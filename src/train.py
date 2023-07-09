import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torch.utils.data as data
import torchvision
from torchvision import transforms
import PIL.Image as Image

#from torchvision.datasets import MNIST # test template

import os
import datetime

from .trainer import Trainer, configure_optimizer
from .dataset import perform_train_val_test_split, COCOImageDatset, test_transform
from .models import JohnsonsImageTransformNet
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
    STYLE_IMAGE_PATH,
)

import wandb

def train_model(model_class, train_dl, val_dl):
    # configure model
    model = model_class()
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
    model = model_class()
    model.load_state_dict(torch.load(pretrained_filename))
    
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
        STYLE_IMAGE_PATH,
        TRAIN_RATIO,
        VAL_RATIO, 
        TEST_RATIO
    )
    
    # # create dataloaders
    train_dl = data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_dl = data.DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_dl = data.DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    # torch.autograd.set_detect_anomaly(True)
    # train the model
    model = train_model(JohnsonsImageTransformNet, train_dl, val_dl)
    
    model.eval()
    
    # model = run_pretrained_model(
    #     "checkpoints/<class 'src.models.johnson_model.JohnsonsImageTransformNet'>--2023-07-09_12-25-48.pth",
    #     JohnsonsImageTransformNet
    # )
    model = model.to(ACCELERATOR)
    image_to_style = Image.open("test_images/test1.jpg")
    image_to_style = Image.open("test_images/000000436138.jpg")
    image_transformed = test_transform()(image_to_style)
    img_crop = transforms.ToPILImage()(image_transformed.squeeze(0).cpu())
    img_crop \
        .save("test_images/test1_crop.jpg")
    
    result = model(image_transformed.unsqueeze(0).to(ACCELERATOR))

    result_image = transforms.ToPILImage()(result.squeeze(0).cpu())
    # save the result
    result_image \
        .save("test_images/test1_result.jpg")
    
    # test the model TODO: implement test function
