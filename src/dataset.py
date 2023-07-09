import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms

import pandas as pd
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

from src.config import ACCELERATOR   
        
    
class COCOImageDatset(Dataset): # might consider loading multiple images of the same style for better generalization
    def __init__(self, root, style_image_path, transform):
        self.root = root
        self.transform = train_transform()
        
        # load images from root directory in dataframe 
        imgs = list(sorted(os.listdir(self.root)))
        
        np.random.seed(42)
        self.images = np.asarray(imgs)
        np.random.shuffle(self.images)
        
        # load style image
        self.style_image = Image.open(style_image_path).convert("RGB")
        # self.style_image = self.transform(self.style_image)
        # print(self.style_image.shape)
        
    def __len__(self):
        #return len(self.images)
        return 1000
    
    def __getitem__(self, index):
        image_path = os.path.join(self.root, self.images[index])
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        image = image.to(ACCELERATOR)
        
        if image.shape[1] != 224 and image.shape[2] != 224:
            print("Image shape: ", image.shape)
        
        style_image = self.transform(self.style_image).to(ACCELERATOR)
        
        return image, style_image
        


def train_transform():
    return transforms.Compose(
        [
            #crop 224x224
            transforms.Resize(224, antialias=True),
            transforms.RandomCrop((224, 224)),
            transforms.ToTensor(),
        ]
    )


def test_transform():
    return transforms.Compose(
        {
            transforms.Resize(224, antialias=True),
            transforms.RandomCrop((224, 224)),
            transforms.ToTensor(),
        }
    )
    


def perform_train_val_test_split(dataset, data_dir, style_image_path, train_size, val_size, test_size):
        if train_size + val_size + test_size != 1:
            raise ValueError("train_size + val_size + test_size must equal 1")
        
        dataset_train = dataset(
            root=data_dir,
            style_image_path=style_image_path,
            transform=train_transform(),
        )
        
        dataset_test = dataset(
            root=data_dir,
            style_image_path=style_image_path,
            transform=test_transform(),
        )
        
        dataset_size = len(dataset_train) # replace with actual dataset size
        train_set_size = int(dataset_size * train_size)
        val_set_size = int(dataset_size * val_size)
        test_set_size = dataset_size - train_set_size - val_set_size 
        
        # perform train/val/test split 
        # (a little complex because we need to make sure the same random seed is used for each split)
        rng = torch.manual_seed(42)
        torch.set_rng_state(rng.get_state())
        train_ds, _ = torch.utils.data.random_split(dataset_train, [train_set_size, test_set_size + val_set_size])
        
        torch.set_rng_state(rng.get_state())
        train_set, test_ds = torch.utils.data.random_split(dataset_test, [train_set_size + val_set_size, test_set_size])
        
        torch.set_rng_state(rng.get_state())
        _, val_ds = torch.utils.data.random_split(train_set, [train_set_size, val_set_size])

        return train_ds, val_ds, test_ds


if __name__ == "__main__":
    # test dataset
    dataset = COCOImageDatset(
        root="/Users/robert/Desktop/style_transfer/style_transfer/data/train2017",
        style_image_path="/Users/robert/Desktop/style_transfer/style_transfer/style_images/style1.jpeg",
        transform=train_transform(),
    )
    ax, fig = plt.subplots(2)
    fig[0].imshow(dataset[0][0].permute(1, 2, 0))
    fig[1].imshow(dataset[0][1].permute(1, 2, 0))
    plt.show()
    
        