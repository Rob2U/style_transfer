import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import MNIST

from src.config import ACCELERATOR

class TorchDataset(Dataset):
    def __init__(self, train, transform, download):
        super().__init__()
        self.train = train
        self.transform = transform
        self.download = download    
        
    def __len__(self):
        pass
    
    def __getitem__(self, index):
        pass
    
    def __download__(self):
        pass
    
    
class MNISTWrapper(MNIST):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def __getitem__(self, index):
        img, lbl = super().__getitem__(index)
        #print(img)
        img = img.float()
        img = torch.cat((img, img, img), dim=0)
        img = img.reshape(3, 224, 224)
        img = img.to(ACCELERATOR)
        
        return img, lbl

def train_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(224, antialias=True),
    ])

def test_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(224, antialias=True),
        
    ])


def perform_train_val_test_split(dataset, data_dir, train_size, val_size, test_size):
        if train_size + val_size + test_size != 1:
            raise ValueError("train_size + val_size + test_size must equal 1")
        
        dataset_train = dataset(
            root=data_dir,
            train=True,
            transform=train_transform(),
            download=True,
        )
        
        dataset_test = dataset(
            root=data_dir,
            train=True,
            transform=test_transform(),
            download=True,
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
