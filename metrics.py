import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
# Metrics
from torchmetrics.classification import BinaryF1Score, BinaryJaccardIndex
import segmentation_models_pytorch as smp

def cal_metrics(preds, target, threshold=0.5):
    """
        Calculate DL metrics

        Parameters:
            preds (numpy array): model prediction results
            target (numpy array): ground truth
            threshold (float): threhold 
        Returns:
            F1 score, Iou score
    """
    preds_binary = (preds > threshold).astype(np.float32)
    preds_binary, target = torch.tensor(preds_binary), torch.tensor(target)

    # F1
    f1 = BinaryF1Score()
    f1_score = f1(preds_binary, target)
    # IoU
    IoU = BinaryJaccardIndex()
    iou_score = IoU(preds_binary, target)    

    return f1_score, iou_score



# PyTroch version

SMOOTH = 1e-6

def iou_pytorch(outputs: torch.Tensor, labels: torch.Tensor):
    # You can comment out this line if you are passing tensors of equal shape
    # But if you are passing output from UNet or something it will most probably
    # be with the BATCH x 1 x H x W shape
    outputs = outputs.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W
    
    intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels).float().sum((1, 2))         # Will be zzero if both are 0
    
    iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0
    
    thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresolds
    
    return thresholded  # Or thresholded.mean() if you are interested in average across the batch
    
    
# Numpy version
# Well, it's the same function, so I'm going to omit the comments

def iou_numpy(outputs: np.array, labels: np.array):
    outputs = outputs.squeeze(1)
    
    intersection = (outputs & labels).sum((1, 2))
    union = (outputs | labels).sum((1, 2))
    
    iou = (intersection + SMOOTH) / (union + SMOOTH)
    
    thresholded = np.ceil(np.clip(20 * (iou - 0.5), 0, 10)) / 10
    
    return thresholded  # Or thresholded.mean()





# Criterion : https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch#Dice-Loss

# class DiceLoss(nn.Module):
#     def __init__(self, weight=None, size_average=True):
#         super(DiceLoss, self).__init__()

#     def forward(self, inputs, targets, smooth=1):
        
#         #comment out if your model contains a sigmoid or equivalent activation layer
#         # inputs = F.sigmoid(inputs)       
        
#         #flatten label and prediction tensors
#         inputs = inputs.view(-1)
#         targets = targets.view(-1)
        
#         intersection = (inputs * targets).sum()                            
#         dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
#         return 1 - dice



class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def __call__(self, logits, true):
        intersection = torch.sum(true * logits, dim=(1,2,3))
        sum_of_squares_pred = torch.sum(torch.square(logits), dim=(1,2,3))
        sum_of_squares_true = torch.sum(torch.square(true), dim=(1,2,3))
        dice = 1 - (2 * intersection + self.smooth) / (sum_of_squares_pred + sum_of_squares_true + self.smooth)
        loss = torch.mean(dice)
        return loss
    


class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        # comment out if your model contains a sigmoid or equivalent activation layer
        # inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss
        print(f"BCE: {BCE}, DICE: {dice_loss}")
        return Dice_BCE
    



#PyTorch
class IoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoULoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        # inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #intersection is equivalent to True Positive count
        #union is the mutually inclusive area of all labels & predictions 
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection 
        
        IoU = (intersection + smooth)/(union + smooth)
                
        return 1 - IoU

# class BCEDiceLoss(nn.Module):
#     def __init__(self, bce_weight=1.0, weight=None, size_average=True):
#         super(BCEDiceLoss, self).__init__()
#         self.bce_weight = bce_weight
#         self.bce_loss = nn.BCEWithLogitsLoss(weight=weight, size_average=size_average)

#     def forward(self, logits, targets):
#         # 이진 교차 엔트로피 손실 계산
#         bce_loss = self.bce_loss(logits, targets)

#         # 소프트맥스 함수를 사용하여 확률을 얻음
#         probs = F.softmax(logits, dim=1)

#         # 각 클래스에 대한 Dice Loss 계산
#         intersection = torch.sum(probs * targets, dim=(2, 3))
#         union = torch.sum(probs + targets, dim=(2, 3))
#         dice_loss = 1.0 - (2.0 * intersection + 1.0) / (union + 1.0)
        
#         # BCELoss - log(Dice Loss)로 전체 손실 계산
#         # bce_dice_loss = bce_loss - torch.log(dice_loss + 1e-5)
#         bce_dice_loss = self.bce_weight * bce_loss + (1 - self.bce_weight) * dice_loss
 
#         return bce_dice_loss, dice_loss
