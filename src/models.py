import pretrainedmodels
import torch.nn as nn
from torch.nn import functional as F
import torch

class SeResnext50_32x4D(nn.Module):
    def __init__(self, pretrained):
        super(SeResnext50_32x4D, self).__init__()
        self.base_model = pretrainedmodels.__dict__['se_resnext50_32x4d'](pretrained=pretrained)
        self.out   = nn.Linear(2048, 2)

    def forward(self, images, targets, weights=None):
        bs, _, _, _ = images.shape
        out = self.base_model.features(images)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.reshape(bs, -1)
        out = self.out(out)
        loss = nn.CrossEntropyLoss(weight=weights)(out, targets)
        return out, loss
