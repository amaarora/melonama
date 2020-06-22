import pretrainedmodels
import torch.nn as nn
from torch.nn import functional as F
import torch
from efficientnet_pytorch import EfficientNet

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        return F_loss.mean()

loss_fn = FocalLoss(0.75, 2)

class SeResnext50_32x4D(nn.Module):
    def __init__(self, pretrained):
        super(SeResnext50_32x4D, self).__init__()
        self.base_model = pretrainedmodels.__dict__['se_resnext50_32x4d'](pretrained=pretrained)
        self.out   = nn.Linear(2048, 1)

    def forward(self, images, targets, weights=None, args=None):
        bs, _, _, _ = images.shape
        out = self.base_model.features(images)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.reshape(bs, -1)
        out = self.out(out)

        # Choose loss function based on args
        if not args.focal_loss and weights is not None:
            weights_ = weights[targets.data.view(-1).long()].view_as(targets)
            loss_func = nn.BCEWithLogitsLoss(reduction='none')
            loss = loss_func(out, targets.view(-1,1).type_as(out))
            loss_class_weighted = loss * weights_
            loss = loss_class_weighted.mean()
        elif not args.focal_loss:
            loss = nn.BCEWithLogitsLoss()(out, targets.view(-1,1).type_as(out))
        elif args.focal_loss:
            loss = loss_fn(out, targets.view(-1,1).type_as(out))

        return out, loss


class EfficientNetBx(nn.Module):
    def __init__(self, pretrained=True, arch_name='efficientnet-b0'):
        super(EfficientNetBx, self).__init__()
        self.pretrained = pretrained
        self.base_model = EfficientNet.from_pretrained(arch_name) if pretrained else EfficientNet.from_name(arch_name)
        nftrs = self.base_model._fc.in_features
        self.base_model._fc = nn.Linear(nftrs, 1)

    def forward(self, images, targets, weights=None, args=None):
        out = self.base_model(images)  

#         Choose loss function based on args
        if not args.focal_loss and weights is not None:
            weights_ = weights[targets.data.view(-1).long()].view_as(targets)
            loss_func = nn.BCEWithLogitsLoss(reduction='none')
            loss = loss_func(out, targets.view(-1,1).type_as(out))
            loss_class_weighted = loss * weights_
            loss = loss_class_weighted.mean()
        elif not args.focal_loss:
            loss = nn.BCEWithLogitsLoss()(out, targets.view(-1,1).type_as(out))
        elif args.focal_loss:
            loss = loss_fn(out, targets.view(-1,1).type_as(out))

        return out, loss

