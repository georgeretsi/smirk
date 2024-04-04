import os

import torch
import torch.nn.functional as F
import torchvision


class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self):
        super(VGGPerceptualLoss, self).__init__()
        blocks = [torchvision.models.vgg16(weights='DEFAULT').features[:4].eval(),
                  torchvision.models.vgg16(weights='DEFAULT').features[4:9].eval(),
                  torchvision.models.vgg16(weights='DEFAULT').features[9:16].eval(),
                  torchvision.models.vgg16(weights='DEFAULT').features[16:23].eval()]
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x, y):
        x = x * 0.5 + 0.5
        y = y * 0.5 + 0.5
        x = (x - self.mean) / self.std
        y = (y - self.mean) / self.std

        x = F.interpolate(x, mode='bilinear', size=(224, 224), align_corners=False)
        y = F.interpolate(y, mode='bilinear', size=(224, 224), align_corners=False)
        perceptual_loss = 0.0
        style_loss = 0.0

        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)

            perceptual_loss += torch.nn.functional.l1_loss(x, y)

            # b, ch, h, w = x.shape
            # act_x = x.reshape(x.shape[0], x.shape[1], -1)
            # act_y = y.reshape(y.shape[0], y.shape[1], -1)
            # gram_x = act_x @ act_x.permute(0, 2, 1) / (ch * h * w)
            # gram_y = act_y @ act_y.permute(0, 2, 1) / (ch * h * w)
            # style_loss += torch.nn.functional.l1_loss(gram_x, gram_y)

        return perceptual_loss#, style_loss
