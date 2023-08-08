# Desc: Defines the dataset for the style transfer model

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms

import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

from .config import ACCELERATOR   
        
    
class ImageDatset(Dataset): # might consider loading multiple images of the same style for better generalization
    """
    A simple dataset class for the style transfer model, that tries to load all files in the given root directory as images.
    """
    
    def __init__(self, root, style_image_path):
        self.root = root
        self.transform = train_transform()
        
        # load images from root directory in dataframe 
        imgs = list(sorted(os.listdir(self.root)))
        
        np.random.seed(42)
        self.images = np.asarray(imgs)
        np.random.shuffle(self.images)
        
        # load style image
        self.style_image = Image.open(style_image_path).convert("RGB")
        
    def __len__(self):
        return len(self.images)
        

    def __getitem__(self, index):
        image_path = os.path.join(self.root, self.images[index])
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        image = image.to(ACCELERATOR)
        
        if image.shape[1] != 256 and image.shape[2] != 256:
            print("Image shape: ", image.shape)
        
        style_image = self.transform(self.style_image).to(ACCELERATOR)
        
        return image, style_image
        

def train_transform():
    return transforms.Compose(
        [
            transforms.Resize(256, antialias=True),
            transforms.CenterCrop((256, 256)),
            transforms.ToTensor(),
        ]
    )

# in the end we did not use this transform as there was no point in switching transforms between train and test, because the dataset was already big enough
def test_transform():
    return transforms.Compose(
        {
            transforms.Resize(256, antialias=True),
            transforms.CenterCrop((256, 256)),
            transforms.ToTensor(),
        }
    )
    

# same as above, we did not need to use this function in the end
def perform_train_val_test_split(dataset, data_dir, style_image_path, train_size, val_size, test_size):
        if train_size + val_size + test_size != 1:
            raise ValueError("train_size + val_size + test_size must equal 1")
        
        dataset_train = dataset(
            root=data_dir,
            style_image_path=style_image_path,
        )
        
        dataset_test = dataset(
            root=data_dir,
            style_image_path=style_image_path,
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
    dataset = ImageDatset(
        root="/Users/robert/Desktop/style_transfer/style_transfer/data/train2017",
        style_image_path="/Users/robert/Desktop/style_transfer/style_transfer/style_images/style1.jpeg",
        transform=train_transform(),
    )
    ax, fig = plt.subplots(2)
    fig[0].imshow(dataset[0][0].permute(1, 2, 0))
    fig[1].imshow(dataset[0][1].permute(1, 2, 0))
    plt.show()
    
        