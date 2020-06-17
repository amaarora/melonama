import pretrainedmodels
import torch.nn as nn
from torch.nn import functional as F
import torch

class SeResnext50_32x4D(nn.Module):
    def __init__(self, pretrained):
        super(SeResnext50_32x4D, self).__init__()
        self.base_model = pretrainedmodels.__dict__['se_resnext50_32x4d'](pretrained=pretrained)
        self.out   = nn.Linear(2048, 1)

    def forward(self, images, targets, weights=None):
        bs, _, _, _ = images.shape
        out = self.base_model.features(images)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.reshape(bs, -1)
        out = self.out(out)

        # add weighted BCE loss due to class imbalance
        if weights is not None:
            weights_ = weights[targets.data.view(-1).long()].view_as(targets)
            loss_func = nn.BCEWithLogitsLoss(reduction='none')
            loss = loss_func(out, targets.view(-1,1).type_as(out))
            loss_class_weighted = loss * weights_
            loss = loss_class_weighted.mean()
        else:
            loss = nn.BCEWithLogitsLoss()(out, targets.view(-1,1).type_as(out))
        
        return out, loss
