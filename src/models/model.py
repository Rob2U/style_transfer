import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F



class TorchModel(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.layer1 = nn.Linear(1, 1)

    def forward(self, x):
        return self.layer1(x)
        


def configure_optimizers(self):
    optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[100, 150], gamma=0.1
    )
    return [optimizer], [lr_scheduler]
