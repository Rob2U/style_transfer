import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch

from torchvision.models import vgg16, VGG16_Weights

from src.config import LEARNING_RATE, ACCELERATOR

import src.models.summary as summary


class TorchModel(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs

    def forward(self, x):
        pass
    
class VGG16Wrapper(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.vgg16 = vgg16(weights=VGG16_Weights.DEFAULT, progress=True).features[:]
        self.linear = nn.Linear(512*7*7, 10)
        self.flatten = nn.Flatten()
        

    def forward(self, x):
        x = x.to(ACCELERATOR)
        return self.linear(self.flatten(self.vgg16(x)))
    
    
class JohnsonsImageTransformNet(nn.Module):
    def __init__(self, filters_res_block=128, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        
        
        # 1. Conv Layer
        conv_1_filters = 32
        conv_1_kernel_size = 9
        conv_1_stride = 1
        self.conv2d_1 = nn.Conv2d(3, conv_1_filters, 
                                  kernel_size=conv_1_kernel_size, 
                                  stride=conv_1_stride, 
                                  padding=conv_1_kernel_size//2
                                  )
        
        # 2. Conv Layer
        conv_2_filters = 64
        conv_2_kernel_size = 3
        conv_2_stride = 2
        self.conv2d_2 = nn.Conv2d(conv_1_filters, conv_2_filters, 
                                  kernel_size=conv_2_kernel_size, 
                                  stride=conv_2_stride, 
                                  padding=conv_2_kernel_size//2
                                  )

        # 3. Conv Layer
        conv_3_filters = 128
        conv_3_kernel_size = 3
        conv_3_stride = 2
        self.conv2d_3 = nn.Conv2d(conv_2_filters, conv_3_filters, 
                                  kernel_size=conv_3_kernel_size, 
                                  stride=conv_3_stride, 
                                  padding=conv_3_kernel_size//2
                                  )
        self.res_blocks = nn.Sequential(
            *[JohnsonsImageTransformNetResidualBlock(filters=filters_res_block) for _ in range(5)]
        )
        # ------------ RECONSTRUCTION ------------
        conv_4_filters = 64
        conv_4_kernel_size = 3
        conv_4_stride = 2 # 1/2
        self.conv2d_4 = nn.ConvTranspose2d(conv_3_filters, conv_4_filters,
                                           kernel_size=conv_4_kernel_size,
                                           stride=conv_4_stride,
                                           padding=conv_4_kernel_size//2,
                                           output_padding=1
                                           )
        
        conv_5_filters = 32
        conv_5_kernel_size = 3
        conv_5_stride = 2 # 1/2
        
        self.conv2d_5 = nn.ConvTranspose2d(conv_4_filters, conv_5_filters,
                                           kernel_size=conv_5_kernel_size,
                                           stride=conv_5_stride,
                                           padding=conv_5_kernel_size//2,
                                           output_padding=1
                                           )
        
        conv_6_filters = 3
        conv_6_kernel_size = 9
        conv_6_stride = 1 # 1/2
        self.conv2d_6 = nn.ConvTranspose2d(conv_5_filters, conv_6_filters,
                                           kernel_size=conv_6_kernel_size,
                                           stride=conv_6_stride,
                                           padding=conv_6_kernel_size//2,
                                           output_padding=0
                                           )
        

    def forward(self, x):
        x = x.to(ACCELERATOR)
        # -- deconstruction --
        dec_x = self.conv2d_3(self.conv2d_2(self.conv2d_1(x)))
        res_x = self.res_blocks(dec_x)
        rec_x = self.conv2d_6(self.conv2d_5(self.conv2d_4(res_x)))
        
        return rec_x
        
    
class JohnsonsImageTransformNetResidualBlock(nn.Module):
    def __init__(self, filters=128, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.conv2d_1 = nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1)
        self.batch_norm_1 = nn.BatchNorm2d(filters)
        self.relu = nn.ReLU()
        self.conv2d_2 = nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1)
        self.batch_norm_2 = nn.BatchNorm2d(filters)
        
    def forward(self, x):
        x = x.to(ACCELERATOR)
        y = self.conv2d_1(x)
        y = self.batch_norm_1(y)
        y = self.relu(y)
        y = self.conv2d_2(y)
        y = self.batch_norm_2(y)
        y = y + x 
        # alternative to residual connection:
        # y = self.relu(y)
        
        return y
    
if __name__ == '__main__':
    model = JohnsonsImageTransformNet()
    sample = torch.randn(1, 3, 256, 256)
    print(model(sample).shape)
    print(model)
    summary.summary(
        model=model,
        input_size=(3, 256, 256),
        batch_size=1,
        device='cpu',
    )