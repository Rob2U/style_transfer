import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms


class TorchDataset(Dataset):
    def __init__(self, root, train, transform, download):
        super().__init__()
        self.root = root
        self.train = train
        self.transform = transform
        self.download = download    
        
    def __len__(self):
        pass
    
    def __getitem__(self, index):
        pass
    
    def __download__(self):
        pass

def train_transform():
    return None

def test_transform():
    return None

def perform_train_val_test_split(dataset_path, train_size, val_size, test_size):
        if train_size + val_size + test_size != 1:
            raise ValueError("train_size + val_size + test_size must equal 1")
        
        dataset_size = 69420 # replace with actual dataset size
        train_set_size = int(dataset_size * train_size)
        val_set_size = int(dataset_size * val_size)
        test_set_size = int(dataset_size * test_size)
        
        dataset_train = TorchDataset(
            root=dataset_path,
            train=True,
            transform=train_transform(),
            download=True,
        )
        
        dataset_test = TorchDataset(
            root=dataset_path,
            train=True,
            transform=test_transform(),
            download=True,
        )
        
        # perform train/val/test split 
        # (a little complex because we need to make sure the same random seed is used for each split)
        torch.seed(42)
        train_ds, _ = torch.utils.data.random_split(dataset_train, [train_set_size, test_set_size + val_set_size])
        
        torch.seed(42)
        train_set, test_ds = torch.utils.data.random_split(dataset_test, [train_set_size + val_set_size, test_set_size])
        
        torch.seed(42)
        _, val_ds = torch.utils.data.random_split(train_set, [train_set_size, val_set_size])

        return train_ds, val_ds, test_ds
