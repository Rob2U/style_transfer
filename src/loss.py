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
    
    def get_relu_1_2(self, x):
        return self.vgg16[0:3](x)
    
    def get_relu_2_2(self, x):
        return self.vgg16[0:8](x)
    
    def get_relu_3_3(self, x):
        return self.vgg16[0:15](x)
    
    def get_relu_4_3(self, x):
        return self.vgg16[0:22](x)
        
        
def _total_variation_reg(x):
    if _total_variation_reg.tv is None:
        _total_variation_reg.tv = TotalVariation()
        _total_variation_reg.tv.to(ACCELERATOR)
    return torch.mean(_total_variation_reg.tv(x)) # b, c, h, w -> b, 1 -> 1
_total_variation_reg.tv = None

def _gram_matrix1(x):
    b, c, h, w = x.shape
    x = x.view(b, c, h*w) 
    return torch.bmm(x, x.transpose(-1, -2)).div(c * h * w) # b,c, h*w -> b, c, c

def _gram_matrix(x):
    b ,c, h, w = x.size()
    g = torch.bmm(x.view(b, c, h*w), x.view(b, c, h*w).transpose(1,2))
    return g.div(h*w)

mse = torch.nn.MSELoss(reduction='mean')
def _style_loss(x,y):
    gram_x = _gram_matrix(x) # b, c, c
    gram_y = _gram_matrix(y) # b, c, c
    return mse(gram_y, gram_x) # b, c, c -> 1

def _feature_loss(x,y):
    x_norm = torch.mean((x - y)**2) # b, c, h, w -> 1
    return x_norm



def calculate_loss(alpha, beta, gamma, generated_image, style_image, original_image):
    style_image = style_image.to(ACCELERATOR)
    original_image = original_image.to(ACCELERATOR)
    
    generated_image_relu1_2, generated_image_relu2_2, generated_image_relu3_3, generated_image_relu4_3 = VGG16Pretrained()(generated_image)
    
    if calculate_loss.style_image_relu1_2 is None or generated_image_relu1_2.shape != calculate_loss.style_image_relu1_2.shape:
        calculate_loss.style_image_relu1_2, calculate_loss.style_image_relu2_2, calculate_loss.style_image_relu3_3, calculate_loss.style_image_relu4_3 = VGG16Pretrained()(style_image)

    original_image_relu3_3 = VGG16Pretrained().get_relu_3_3(original_image)
    
    # Feature Loss
    feature_loss = _feature_loss(generated_image_relu3_3, original_image_relu3_3)
    # print("Feature Loss: ", feature_loss)
    
    # Style Loss
    style_loss1 = _style_loss(generated_image_relu1_2, torch.zeros(generated_image_relu1_2.shape, device=ACCELERATOR) + calculate_loss.style_image_relu1_2)
    style_loss2 = _style_loss(generated_image_relu2_2, torch.zeros(generated_image_relu2_2.shape, device=ACCELERATOR) + calculate_loss.style_image_relu2_2)
    style_loss3 = _style_loss(generated_image_relu3_3, torch.zeros(generated_image_relu3_3.shape, device=ACCELERATOR) + calculate_loss.style_image_relu3_3)
    style_loss4 = _style_loss(generated_image_relu4_3, torch.zeros(generated_image_relu4_3.shape, device=ACCELERATOR) + calculate_loss.style_image_relu4_3)
    
    # print("Style Loss: ", style_loss1, style_loss2, style_loss3, style_loss4)
    
    # Total Variation Regularization
    total_var_reg = _total_variation_reg(generated_image)
    # print("Total Variation Regularization: ", total_var_reg)
    
    loss = alpha * feature_loss + beta * (style_loss1 + style_loss2 + style_loss3 + style_loss4) + gamma * total_var_reg
    #print("Loss: ", loss)
    # loss = torch.mean(loss) -> unnecessary because loss is already a scalar
    
    return loss
calculate_loss.style_image_relu1_2 = None
calculate_loss.style_image_relu2_2 = None
calculate_loss.style_image_relu3_3 = None
calculate_loss.style_image_relu4_3 = None
    
    
if __name__ == "__main__":
    vgg16model = VGG16Pretrained()
    x = torch.randn(1, 3, 256, 256)
    print(vgg16model(x))
    
    print(_total_variation_reg(x))
    
    summary(vgg16model, (3, 256, 256), device="cpu")
    
    x = torch.randn(10, 3, 256, 256)
    x2 = torch.randn(10, 3, 256, 256)
    x3 = torch.randn(10, 3, 256, 256)
    
    print(calculate_loss(1, 1, 1, x, x2, x3))
    
    import src.dataset
    import matplotlib.pyplot as plt
    
    dataset = src.dataset.COCOImageDatset(
        root="data/train2017/",
        style_image_path="style_images/style5.jpg",
        transform=src.dataset.train_transform(),
    )
    img, style = dataset[0]
    
    ax, fig = plt.subplots(2)
    fig[0].imshow(dataset[0][0].permute(1, 2, 0))
    fig[1].imshow(dataset[0][1].permute(1, 2, 0))
    plt.show()
    
    print(calculate_loss(1, 1, 1, img.unsqueeze(0), style.unsqueeze(0), img.unsqueeze(0)))
    
    