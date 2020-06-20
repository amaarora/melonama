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


class ClassificationHeadB0(nn.Module):
    def __init__(self, n_out):
        super(ClassificationHeadB0, self).__init__()
        self.n_out = n_out
        self.bn1 = torch.nn.BatchNorm1d(1280, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        self.l1 = nn.Linear(1280, 512)
        self.bn2 = torch.nn.BatchNorm1d(512, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        self.l2 = nn.Linear(512, n_out)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        bs, _, _, _ = x.shape
        x = self.pool(x)
        x = x.reshape(bs, -1)
        x = self.bn1(x)
        x = self.dropout1(x)
        x = self.l1(x)
        x = self.relu(x)
        x = self.bn2(x)
        x = self.dropout2(x)
        return self.l2(x)    


class EfficientNetBx(nn.Module):
    def __init__(self, pretrained, arch_name='efficientnet-b0'):
        super(EfficientNetBx, self).__init__()
        self.pretrained = pretrained
        self.base_model = EfficientNet.from_pretrained(arch_name) if pretrained else EfficientNet.from_name(arch_name)
        self.head = ClassificationHeadB0(1)

    def forward(self, images, targets, weights=None, args=None):
        out = self.base_model.extract_features(images)  
        out = self.head(out)
        loss = nn.BCEWithLogitsLoss()(out, targets.view(-1,1).type_as(out))
        return out, loss


