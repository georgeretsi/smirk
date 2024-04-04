# -*- coding: utf-8 -*-
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# Using this computer program means that you agree to the terms 
# in the LICENSE file included with this software distribution. 
# Any use not explicitly granted by the LICENSE is prohibited.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# For comments or questions, please email us at deca@tue.mpg.de
# For commercial licensing contact, please contact ps-license@tuebingen.mpg.de

import torch
import torch.nn as nn
from torchvision import models
from .resnet import resnet50, load_state_dict
import torch.nn.functional as F

import torchvision


class ExpressionLoss(nn.Module):
    """ Code borrowed from EMOCA https://github.com/radekd91/emoca """
    def __init__(self):
        super(ExpressionLoss, self).__init__()

        self.backbone = resnet50(num_classes=100, include_top=False, emoca_specific=True).eval().cuda()

        emotion_checkpoint = torch.load('assets/ResNet50/checkpoints/deca-epoch=01-val_loss_total/dataloader_idx_0=1.27607644.ckpt')['state_dict']
        
        del emotion_checkpoint['backbone.fc.weight']
        del emotion_checkpoint['backbone.fc.bias']
        #del emotion_checkpoint['linear.weight']
        #del emotion_checkpoint['linear.bias']
        #emotion_checkpoint['backbone.fc.weight'] = self.backbone.state_dict()['fc.weight']
        #emotion_checkpoint['backbone.fc.bias'] = self.backbone.state_dict()['fc.bias']
        del emotion_checkpoint['linear.weight']
        del emotion_checkpoint['linear.bias']
         
        self.load_state_dict(emotion_checkpoint, strict=False)
        
    def _cos_metric(self, x1, x2):
        return 1.0 - F.cosine_similarity(x1, x2, dim=1)

    def forward(self, gen, tar, use_mean=True, metric='l2'):
        
        gen_out = self.backbone(gen).view(gen.shape[0], -1)
        tar_out = self.backbone(tar).view(tar.shape[0], -1)

        if metric == 'l2':
            loss = ((gen_out - tar_out)**2).mean(dim=1)
        elif metric == 'l1':
            loss = (torch.abs(gen_out - tar_out)).mean(dim=1)
        elif metric == 'cos':
            loss = self._cos_metric(gen_out, tar_out)
        else:
            raise ValueError('Unknown metric {}'.format(metric))
        
        if use_mean:
            loss = loss.mean()
            
        return loss
    

