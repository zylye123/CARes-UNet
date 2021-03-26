
import torch.nn.functional as F
from utils import one_hot_segmentation
import torch.nn as nn
import torch






class CrossEntropy(nn.Module):
    def __init__(self):
        super().__init__()
        self.Loss = nn.CrossEntropyLoss()

    def forward(self, pred, target):
        if len(target.size()) == 4:
            target = torch.squeeze(target, 1).long()
        return self.Loss(pred, target)

class DiceLossSoftmax(nn.Module):
    def __init__(self, smooth=1e-4):
        super().__init__()
        self.smooth = smooth

    def flatten(self, tensor):
        tensor = tensor.transpose(0, 1).contiguous()
        return tensor.view(tensor.size(0), -1)

    def forward(self, pred, target):
        return 1.0 - self.dice_coef(pred, target)

    def dice_coef(self, pred, target):
        n, c = pred.shape[:2]

        pred = F.softmax(pred, dim=1)
        target = one_hot_segmentation(target, c).float()

        pred = pred.view(n, c, -1)
        target = target.view(n, c, -1)

        intersect = torch.sum(target * pred, -1)
        dice = (2 * intersect + self.smooth) / (torch.sum(target, -1) + torch.sum(pred, -1) + self.smooth)
        # in class axis
        dice = torch.mean(dice, dim=-1)
        # in batch axis
        return torch.mean(dice)
    


class DiceWithBceLoss(nn.Module):
    def __init__(self, weights=[1., 1.]):
        super().__init__()
        self.dice_loss = DiceLossSoftmax()
        self.bce_loss = CrossEntropy()
        self.weights = weights

    def forward(self, pred, target):
        dl = self.dice_loss(pred, target)
        bl = self.bce_loss(pred, target)
        loss = self.weights[0] * dl + self.weights[1] * bl

        return loss
