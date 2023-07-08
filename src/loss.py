import torch
from torchvision.models import vgg16, VGG16_Weights
import torch.nn as nn
from torchmetrics import TotalVariation

from src.config import ACCELERATOR
from src.models.summary import summary

class VGG16Pretrained(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.vgg16 = vgg16(weights=VGG16_Weights.DEFAULT, progress=True).features[:]
        self.to(ACCELERATOR)
        self.vgg16.eval()
        self.vgg16.requires_grad_(False)
        self.vgg16.to(ACCELERATOR)
    
    def forward(self, x):
        relu1_2_out = self.vgg16[0:3](x)
        relu2_2_out = self.vgg16[3:8](relu1_2_out)
        relu3_3_out = self.vgg16[8:15](relu2_2_out)
        relu4_3_out = self.vgg16[15:22](relu3_3_out)
        return relu1_2_out, relu2_2_out, relu3_3_out, relu4_3_out
        
        
def _total_variation_reg(x):
    tv = TotalVariation()
    tv.to(ACCELERATOR)
    return tv(x)

def _gram_matrix(x):
    b, c, h, w = x.shape
    x = x.view(b, c, h*w)
    x_t = x.transpose(-1, -2)
    
    return torch.bmm(x, x_t) / (c * h * w)

def _style_loss(x,y):
    gram_x = _gram_matrix(x) # b, c, c
    gram_y = _gram_matrix(y) # b, c, c
    diff = gram_x - gram_y # b, c, c
    # now we need to calculate the euclidean norm over the c X c dimensions
    diff_norm = torch.linalg.norm(diff, ord=2, dim=(1,2))
    # print("Style_loss shape: ", diff_norm.shape)
    return diff_norm

def _feature_loss(x,y):
    b, c, h, w = x.shape
    diff = x - y
    # calculate the euclidean norm over the c X h X w dimensions
    x_norm_w = torch.linalg.norm(diff, ord=2, dim=(3))
    x_norm_h = torch.linalg.norm(x_norm_w, ord=2, dim=(2))
    x_norm_c = torch.linalg.norm(x_norm_h, ord=2, dim=(1))
    
    x_norm = x_norm_c**2 / (c * h * w)
    # print("Feature_loss shape: ", x_norm.shape)
    return x_norm

def calculate_loss(alpha, beta, gamma, generated_image, style_image, original_image):
    style_image = style_image.to(ACCELERATOR)
    original_image = original_image.to(ACCELERATOR)
    
    generated_image_relu1_2, generated_image_relu2_2, generated_image_relu3_3, generated_image_relu4_3 = VGG16Pretrained()(generated_image)
    style_image_relu1_2, style_image_relu2_2, style_image_relu3_3, style_image_relu4_3 = VGG16Pretrained()(style_image)
    _, _, original_image_relu3_3, _ = VGG16Pretrained()(original_image)
    
    # Feature Loss
    feature_loss = _feature_loss(generated_image_relu3_3, original_image_relu3_3)
    
    # Style Loss
    style_loss1 = _style_loss(generated_image_relu1_2, style_image_relu1_2)
    style_loss2 = _style_loss(generated_image_relu2_2, style_image_relu2_2)
    style_loss3 = _style_loss(generated_image_relu3_3, style_image_relu3_3)
    style_loss4 = _style_loss(generated_image_relu4_3, style_image_relu4_3)
    
    # Total Variation Regularization
    total_var_reg = _total_variation_reg(generated_image)
    
    loss = alpha * feature_loss + beta * (style_loss1 + style_loss2 + style_loss3 + style_loss4) + gamma * total_var_reg
    #print("Loss: ", loss)
    loss = torch.mean(loss)
    
    return loss
    
    
    
if __name__ == "__main__":
    # vgg16 = VGG16Pretrained()
    # x = torch.randn(1, 3, 256, 256)
    # print(vgg16(x).shape)
    
    # print(total_variation_reg(x))
    
    #summary(vgg16, (3, 256, 256), device="cpu")
    
    # x = torch.randn(10, 3, 256, 256)
    # x2 = torch.randn(10, 3, 256, 256)
    # x3 = torch.randn(10, 3, 256, 256)
    
    # print(calculate_loss(1, 1, 1, x, x2, x3))
    
    import src.dataset
    import matplotlib.pyplot as plt
    
    dataset = src.dataset.COCOImageDatset(
        root="data/train2017/train2017/",
        style_image_path="style_images/style3.jpg",
        transform=src.dataset.train_transform(),
    )
    img, style = dataset[0]
    
    # ax, fig = plt.subplots(2)
    # fig[0].imshow(dataset[0][0].permute(1, 2, 0))
    # fig[1].imshow(dataset[0][1].permute(1, 2, 0))
    # plt.show()
    
    print(calculate_loss(1, 1, 1, img.unsqueeze(0), style.unsqueeze(0), img.unsqueeze(0)))
    
    