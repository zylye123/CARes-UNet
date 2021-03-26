import torch


def one_hot_segmentation(label, num_cls):
    batch_size = label.size(0)
    label = label.long()
    out_tensor = torch.zeros(batch_size, num_cls, *label.size()[2:]).to(label.device)
    out_tensor.scatter_(1, label, 1)

    return out_tensor




def dice_coef_2d(pred, target):
    pred = torch.argmax(pred, dim=1, keepdim=True).float()
    target = torch.gt(target, 0.5).float()
    n = target.size(0)
    smooth = 1e-4
    
    target = target.view(n, -1)
    pred = pred.view(n, -1)
    intersect = torch.sum(target * pred, dim=-1)
    dice = (2 * intersect + smooth) / (torch.sum(target, dim=-1) + torch.sum(pred, dim=-1) + smooth)
    dice = torch.mean(dice)

    return dice