import lightning as L
import torch.nn as nn
import torch.optim as optim



class TorchModel(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs

    def forward(self, x):
        pass


class TorchLightningModel(L.LightningModule):
    def __init__(self, **model_kwargs):
        super().__init__()
        self.save_hyperparameters() # saves all passed arguments as hyperparameters
        self.model = TorchModel(**model_kwargs)
        
    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[100, 150], gamma=0.1
        )
        return [optimizer], [lr_scheduler]

    def _calculate_loss(self, batch, mode="train"):
        return 0 # calculate desired loss here

    def training_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch, mode="train")
        return loss

    def validation_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="val")

    def test_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="test")
