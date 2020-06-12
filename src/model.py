import pretrainedmodels
import torch.nn as nn

MODEL_DISPATCHER = {
    'serexnext_50': pretrainedmodels.se_resnext50_32x4d
}

class SeResnext50_32x4D(nn.Module):
    def __init__(self, pretrained):
        super(SeResnext50_32x4D, self).__init__()
        self.model = pretrainedmodels.se_resnext50_32x4d(pretrained=pretrained)
        self.out   = nn.Linear(2048, 1)

    def forward(self, images, targets):
        out = self.model(images)
        out = self.out(out)
        loss = nn.BCEWithLogitsLoss()(out, targets)
        return out, loss
