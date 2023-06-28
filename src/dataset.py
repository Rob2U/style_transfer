import lightning as L
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
        


class LightningDataModule(L.LightningDataModule):
    def __init__(self, data_dir, batch_size, num_workers, dataset_path):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset_path = dataset_path

    #execute only on 1 GPU
    def prepare_data(self):
        self.fetch_dataset(checkpoint_path=self.data_dir)
        
    #execute on every GPU
    def setup(self, stage):
        test_transform = transforms.Compose(
            [
                # some transforms here
            ]
        )
        # For training, we add some augmentation
        train_transform = transforms.Compose(
            [
                # some transforms here
            ]
        )
        # Loading the training dataset. We need to split it into a training and validation part
        # We need to do a little trick because the validation set should not use the augmentation.
        train_dataset = TorchDataset(
            root=self.dataset_path,
            train=True,
            transform=train_transform,
            download=True,
        )
        val_dataset = TorchDataset(
            root=self.dataset_path,
            train=True,
            transform=test_transform,
            download=True,
        )
        L.seed_everything(42)
        self.train_ds, _ = torch.utils.data.random_split(train_dataset, [45000, 5000])
        L.seed_everything(42)
        _, self.val_ds = torch.utils.data.random_split(val_dataset, [45000, 5000])

        self.test_ds = TorchDataset(
            root=self.data_dir, train=False, transform=test_transform, download=True
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def fetch_dataset(self, checkpoint_path):
        # download dataset
        pass
