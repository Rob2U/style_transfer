import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch

from torchvision.models import vgg16, VGG16_Weights

from src.config import LEARNING_RATE, ACCELERATOR

import src.models.summary as summary


# class TorchModel(nn.Module):
#     def __init__(self, **kwargs):
#         super().__init__()
#         self.kwargs = kwargs

#     def forward(self, x):
#         pass
    
    
class JohnsonsImageTransformNet(nn.Module):
    def __init__(self, filters_res_block=128, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        
        
        # 1. Conv Layer
        conv_1_filters = 32
        conv_1_kernel_size = 9 # originally 9
        conv_1_stride = 1 # originally 1
        self.conv2d_1 = nn.Conv2d(3, conv_1_filters, 
                                  kernel_size=conv_1_kernel_size, 
                                  stride=conv_1_stride, 
                                  padding_mode='reflect', #'zeros'
                                  padding=4 #conv_1_kernel_size//2 # originally NOT -1
                                  )
        # self.batch_norm_1 = nn.BatchNorm2d(conv_1_filters)
        self.in1 = nn.InstanceNorm2d(conv_1_filters, affine=True)
        self.relu_1 = nn.ReLU()
        
        # 2. Conv Layer
        conv_2_filters = 64
        conv_2_kernel_size = 3 # originally 3
        conv_2_kernel_size = 3 # originally 3
        conv_2_stride = 2
        self.conv2d_2 = nn.Conv2d(conv_1_filters, conv_2_filters, 
                                  kernel_size=conv_2_kernel_size, 
                                  stride=conv_2_stride, 
                                  padding_mode='reflect', #'zeros'
                                  padding=1 #conv_2_kernel_size//2 # originally NOT -1
                                  )
        # self.batch_norm_2 = nn.BatchNorm2d(conv_2_filters)
        self.in2 = nn.InstanceNorm2d(conv_2_filters, affine=True)
        self.relu_2 = nn.ReLU()

        # 3. Conv Layer
        conv_3_filters = 128
        conv_3_kernel_size = 3 # originally 3
        conv_3_kernel_size = 3 # originally 3
        conv_3_stride = 2
        self.conv2d_3 = nn.Conv2d(conv_2_filters, conv_3_filters, 
                                  kernel_size=conv_3_kernel_size, 
                                  stride=conv_3_stride, 
                                  padding_mode='reflect', #'zeros'
                                  padding=1 #conv_3_kernel_size//2
                                  )
        # self.batch_norm_3 = nn.BatchNorm2d(conv_3_filters)
        self.in3 = nn.InstanceNorm2d(conv_3_filters, affine=True)
        self.relu_3 = nn.ReLU()
        
        self.res_blocks = nn.Sequential(
            *[JohnsonsImageTransformNetResidualBlock(filters=filters_res_block) for _ in range(5)]
        )
        # ------------ RECONSTRUCTION ------------
        conv_4_filters = 64
        conv_4_kernel_size = 3 
        conv_4_stride = 2
        self.up1 = UpsampleConvLayer(in_channels=conv_3_filters, 
                                    out_channels=conv_4_filters,
                                    kernel_size=conv_4_kernel_size,
                                    stride=conv_4_stride,
        )
        # self.conv2d_4 = nn.ConvTranspose2d(conv_3_filters, conv_4_filters,
        #                                    kernel_size=conv_4_kernel_size,
        #                                    stride=conv_4_stride,
        #                                    padding_mode='zeros', #'zeros'
        #                                    padding=1, #conv_4_kernel_size//2, # originally NOT -1
        #                                    output_padding=1 #1
        #                                    )
        # self.batch_norm_4 = nn.BatchNorm2d(conv_4_filters)
        self.in4 = torch.nn.InstanceNorm2d(conv_4_filters, affine=True)
        self.relu_4 = nn.ReLU()
        
        
        conv_5_filters = 32
        conv_5_kernel_size = 3
        conv_5_stride = 2
        self.up2 = UpsampleConvLayer(conv_4_filters, 
                                    conv_5_filters, 
                                    kernel_size=conv_5_kernel_size, 
                                    stride=conv_5_stride
        )
        
        # self.conv2d_5 = nn.ConvTranspose2d(conv_4_filters, conv_5_filters,
        #                                    kernel_size=conv_5_kernel_size,
        #                                    stride=conv_5_stride,
        #                                    padding_mode='zeros', #'zeros'
        #                                    padding=1, #conv_5_kernel_size//2, 
        #                                    output_padding=1 #1
        #                                    )
        # self.batch_norm_5 = nn.BatchNorm2d(conv_5_filters)
        self.in5 = torch.nn.InstanceNorm2d(conv_5_filters, affine=True)
        self.relu_5 = nn.ReLU()
        
        
        conv_6_filters = 3
        conv_6_kernel_size = 9
        conv_6_stride = 1 # 1/2
        self.conv2d_6 = nn.Conv2d(conv_5_filters, conv_6_filters,
                                           kernel_size=conv_6_kernel_size,
                                           stride=conv_6_stride,
                                           padding_mode='reflect', #'zeros'
                                           padding=4, #conv_6_kernel_size//2, # originally NOT -1
                                           )
        self.in6 = torch.nn.InstanceNorm2d(conv_6_filters, affine=True)
        self.relu_6 = nn.ReLU()
        # self.batch_norm_6 = nn.BatchNorm2d(conv_6_filters)        

    def forward(self, x):
        x = x.to(ACCELERATOR)
        # -- deconstruction --
        conv1 = self.relu_1(self.in1(self.conv2d_1(x)))
        conv2 = self.relu_2(self.in2(self.conv2d_2(conv1)))
        conv3 = self.relu_3(self.in3(self.conv2d_3(conv2)))
        res_x = self.res_blocks(conv3)
        
        conv4 = self.relu_4(self.in4(self.up1(res_x)))
        conv5 = self.relu_5(self.in5(self.up2(conv4)))
        # conv4 = self.relu_4(self.batch_norm_4(self.up1(res_x)))
        # conv5 = self.relu_5(self.batch_norm_5(self.up2(conv4)))
        # conv6 = self.conv2d_6(conv5) 
        conv6 = self.relu_6(self.in6(self.conv2d_6(conv5)))
        return conv6
    
class JohnsonsImageTransformNetResidualBlock(nn.Module):
    def __init__(self, filters=128, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.conv2d_1 = nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding_mode='reflect', padding=1)
        self.in1 = nn.InstanceNorm2d(filters, affine=True)
        # self.batch_norm_1 = nn.BatchNorm2d(filters)
        self.relu = nn.ReLU()
        self.conv2d_2 = nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding_mode='reflect', padding=1)
        self.in2 = nn.InstanceNorm2d(filters, affine=True)
        # self.batch_norm_2 = nn.BatchNorm2d(filters)
        
    def forward(self, x):
        x = x.to(ACCELERATOR)
        y = self.conv2d_1(x)
        # y = self.in1(y)
        y = self.in1(y)
        y = self.relu(y)
        y = self.conv2d_2(y)
        # y = self.in2(y)
        y = self.in2(y)
        y = self.relu(y)
        y = y + x 
        # alternative to residual connection:
        #y = self.relu(y)
        
        return y
    
class UpsampleConvLayer(torch.nn.Module):
    """
        http://distill.pub/2016/deconv-checkerboard/
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.upsampling_factor = stride
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=kernel_size//2, padding_mode='reflect')

    def forward(self, x):
        if self.upsampling_factor > 1:
            x = torch.nn.functional.interpolate(x, scale_factor=self.upsampling_factor, mode='nearest')
        return self.conv2d(x)    

    
if __name__ == '__main__':
    model = JohnsonsImageTransformNet().to(ACCELERATOR)
    sample = torch.randn(1, 3, 256, 256).to(ACCELERATOR)
    print(model(sample).shape)
    print(model)
    summary.summary(
        model=model,
        input_size=(3, 256, 256),
        batch_size=1,
        device=ACCELERATOR,
    )