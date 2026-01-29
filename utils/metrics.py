import torch

def dice_coeff(pred, target, smooth=1e-5):
    """
    计算 Dice Coefficient (F1 Score)
    pred: 经过 sigmoid 的预测概率图 (B, 1, H, W)
    target: 真实标签 (B, 1, H, W)
    """
    # 将预测值二值化（为了严格的指标计算），或者直接使用软概率（为了梯度）
    # 在验证阶段通常使用二值化
    pred = (pred > 0.5).float()
    
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice.mean().item()

def iou_score(pred, target, smooth=1e-5):
    """
    计算 IoU (Intersection over Union)
    """
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) - intersection
    
    iou = (intersection + smooth) / (union + smooth)
    return iou.mean().item()