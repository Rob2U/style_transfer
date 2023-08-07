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
        self.vgg16 = torchvision.models.vgg16(pretrained=True, progress=True).features.to(ACCELERATOR)
        for param in self.vgg16.parameters(): # freeze the vgg16 model
            param.requires_grad = False
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
                
                
                
        # print("style act shape: ", len(style_activations))
                
        return content_activations, style_activations
    

class LossCalculator():
    def __init__(self, **kwargs):
        self.mse = torch.nn.MSELoss(reduction='mean').to(ACCELERATOR)
        self.content_layers = ['relu_7'] # use conv4_2
        self.style_layers = ['relu_2', 'relu_4', 'relu_7', 'relu_11'] # conv1_1, conv2_1, conv3_1, conv4_1, conv5_1
        self.loss_net = LossCNN(self.content_layers, self.style_layers)
        self.tv = TotalVariation().to(ACCELERATOR)
        
        self.style_image_style_activations = None
        
        
    def _total_variation_reg(self, x):
        return torch.mean(self.tv(x)) # b, c, h, w -> b, 1 -> 1

    def _style_loss(self, x, y): # loss implementation from ulyanov et al.
        b, c, h, w = x.shape
        F_x = x.view(b, c, h * w)
        F_y = y.view(b, c, h * w)
        
        G_x = torch.bmm(F_x, F_x.transpose(-1, -2))
        G_y = torch.bmm(F_y, F_y.transpose(-1, -2))
        
        G_diff = G_x - G_y # b, c, c
        G_diff = torch.div(G_diff, c * h * w * 2) # b, c, c
        # now pointwise square
        G_diff_square = torch.mul(G_diff, G_diff) # b, c, c
        
        return torch.sum(G_diff_square).div(b) # b, c, c -> 1 (div by b to get mean)

    def _feature_loss(self, x, y):
        b, c, h, w = x.shape
        F_x = x.view(b, c, h * w)
        F_y = y.view(b, c, h * w)
        # pointwise subtract and square
        F_diff = F_x - F_y # b, c, h * w
        F_diff_square = torch.mul(F_diff, F_diff) # b, c, h * w
        F_diff_square_norm = torch.div(F_diff_square, 2) # * c * h * w) # b, c, h * w
        return torch.sum(F_diff_square_norm).div(b * c * h * w) # b, c, h * w -> 1 (div by b to get mean

    def calculate_loss(self, alpha, beta, gamma, generated_image, style_image, original_image):
        style_image = style_image.to(ACCELERATOR)
        original_image = original_image.to(ACCELERATOR)
        generated_content_activations, generated_style_activations = self.loss_net(generated_image)
        
        style_loss_weights = [1.0, 1.0, 1.0, 1.0, 1.0]  # [1/5, 1/5, 1/5, 1/5, 1/5]
        # style_loss_weights = [0.7, 0.3, 0.03, 0.03, 0.03]
        # style_loss_weights = [0.7, 0.3, 0.03, 0.03, 0.03]
        
        original_content_activations = None
        with torch.no_grad(): # the style image and content activations are not generated by the model so we don't need to track gradients
            original_content_activations, _ = self.loss_net(original_image)
            if self.style_image_style_activations is None or \
                generated_style_activations[0].shape != self.style_image_style_activations[0].shape:
                _, self.style_image_style_activations = self.loss_net(style_image)

        # Feature Loss
        content_losses = torch.zeros(len(self.content_layers), device=ACCELERATOR)
        for i in range(len(self.content_layers)):
            content_losses[i] = self._feature_loss(generated_content_activations[i], original_content_activations[i])
        content_loss = torch.sum(content_losses)
        
        # Style Loss
        style_losses = torch.zeros(len(self.style_layers), device=ACCELERATOR)
        for i in range(len(self.style_layers)):
            style_losses[i] = style_loss_weights[i] * self._style_loss(
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
    
    