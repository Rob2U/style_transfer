import torch
import torchvision.models 
import torch.nn as nn
from torchmetrics import TotalVariation

from src.config import ACCELERATOR
from src.models.summary import summary

cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406])
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225])

class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = mean.clone().detach().view(-1, 1, 1).to(ACCELERATOR)
        self.std = std.clone().detach().view(-1, 1, 1).to(ACCELERATOR)

    def forward(self, img):
        # normalize ``img``
        return (img - self.mean) / self.std

class LossCNN(nn.Module):
    def __init__(self, content_layers, style_layers, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.content_layers = content_layers
        self.style_layers = style_layers
        self.vgg16 = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.DEFAULT, progress=True).features.to(ACCELERATOR).eval()
        self.normalization = Normalization(cnn_normalization_mean, cnn_normalization_std).to(ACCELERATOR)
        # may be worth to check if trimming the vgg16 model to only the layers we need is worth it
    
    def forward(self, x):
        content_activations = []
        style_activations = []
        x = x.to(ACCELERATOR)
        
        x = self.normalization(x)
        j = 0
        for layer in self.vgg16.children():
            
            if isinstance(layer, nn.Conv2d):
                j += 1
                name = 'conv_{}'.format(j)
            elif isinstance(layer, nn.ReLU):
                name = 'relu_{}'.format(j)
                # The in-place version doesn't play very nicely with the ``ContentLoss``
                # and ``StyleLoss`` we insert below. So we replace with out-of-place
                # ones here.
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = 'pool_{}'.format(j)
            elif isinstance(layer, nn.BatchNorm2d):
                name = 'bn_{}'.format(j)
            else:
                raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))
            
            x = layer(x)
            
            if name in self.content_layers:
                # print(name)
                content_activations.append(x)

            if name in self.style_layers:
                # print(name)
                style_activations.append(x)
                
        return content_activations, style_activations
    
    
    
class LossCalculator():
    def __init__(self, **kwargs):
        self.mse = torch.nn.MSELoss(reduction='mean').to(ACCELERATOR)
        self.content_layers = ['relu_3'] # may try conv_3
        self.style_layers = ['relu_1', 'relu_2', 'relu_3', 'relu_4'] # prev. relu_1, relu_2, relu_3, relu_4
        self.loss_net = LossCNN(self.content_layers, self.style_layers)
        self.tv = TotalVariation().to(ACCELERATOR)
        
        self.style_image_style_activations = None
        
        
    def _total_variation_reg(self, x):
        return torch.mean(self.tv(x)) # b, c, h, w -> b, 1 -> 1

    def _gram_matrix(self, x):
        a, b, c, d = x.size()

        features = x.view(a * b, c * d) 

        G = torch.mm(features, features.t())  # compute the gram product
        return G.div(a * b * c * d)
    
    # def _gram_matrix(self, x):
    #     b, c, h, w = x.shape
    #     x = x.view(b, c, h*w) 
    #     return torch.bmm(x, x.transpose(-1, -2)).div(c * h * w) # b,c, h*w -> b, c, c

    def _style_loss(self, x, y):
        gram_x = self._gram_matrix(x) # b, c, c
        gram_y = self._gram_matrix(y) # b, c, c
        return self.mse(gram_y, gram_x) # b, c, c -> 1

    def _feature_loss(self, x, y):
        return self.mse(x, y) # b, c, h, w -> 1


    def calculate_loss(self, alpha, beta, gamma, generated_image, style_image, original_image):
        style_image = style_image.to(ACCELERATOR)
        original_image = original_image.to(ACCELERATOR)
        
        
        generated_content_activations, generated_style_activations = self.loss_net(generated_image)
        original_content_activations, _ = self.loss_net(original_image)
        
        if self.style_image_style_activations is None or generated_content_activations[0].shape != self.style_image_style_activations[0].shape:
            _, self.style_image_style_activations = self.loss_net(style_image)
            
            self.style_image_relu1_2 = self.style_image_style_activations[0]
            self.style_image_relu2_2 = self.style_image_style_activations[1]
            self.style_image_relu3_3 = self.style_image_style_activations[2]
            self.style_image_relu4_3 = self.style_image_style_activations[3]
        
        # Feature Loss
        content_losses = torch.zeros(len(self.content_layers), device=ACCELERATOR)
        for i in range(len(self.content_layers)):
            content_losses[i] = self._feature_loss(
                generated_content_activations[i], 
                original_content_activations[i]
            )
        content_loss = torch.sum(content_losses)
        
        # Style Loss
        style_losses = torch.zeros(len(self.style_layers), device=ACCELERATOR)
        for i in range(len(self.style_layers)):
            style_losses[i] = self._style_loss(
                generated_style_activations[i], 
                torch.zeros(generated_style_activations[i].shape, device=ACCELERATOR) + self.style_image_style_activations[i]
            )
        style_loss = torch.sum(style_losses)
        
        # Total Variation Regularization
        total_var_reg = self.tv(generated_image)
        
        loss = alpha * content_loss + beta * style_loss + gamma * total_var_reg

        return loss
    
if __name__ == "__main__":
    
    x = torch.randn(10, 3, 256, 256)
    x2 = torch.randn(10, 3, 256, 256)
    x3 = torch.randn(10, 3, 256, 256)
    
    lc = LossCalculator()
    
    print(lc.calculate_loss(1, 1, 1, x, x2, x3))
    
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
    
    print(lc.calculate_loss(1, 1, 1, img.unsqueeze(0), style.unsqueeze(0), img.unsqueeze(0)))
    
    