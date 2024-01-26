import os, random
from glob import glob
import nibabel as nib
import pandas as pd
import numpy as np
from copy import deepcopy

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torchmetrics.classification import BinaryF1Score, BinaryJaccardIndex
from sklearn.metrics import precision_score, recall_score, confusion_matrix

def get_file_row_nii(file_paths):
    """Produces ID of a patient, image and mask filenames from a particular path"""
    file_paths = sorted(list(set(glob(f'{file_paths}/*.nii.gz')) - set(glob(f'{file_paths}/*_mask.nii.gz'))))
    temp = []
    for file in file_paths:
        for i in range(0, len(nib.load(file).get_fdata().transpose())):
            path = os.path.abspath(file)
            path_no_ext, ext = "".join(path.split('.')[0]), ".".join(path.split('.')[1:])   # . 을 기준으로 파일명/확장자 분리
            filename = os.path.basename(path)
            patient_id = filename.split('.')[0]
            temp.append([patient_id, path, f'{path_no_ext}_mask.{ext}', i])

    filenames_df = pd.DataFrame(temp, columns=['Patient', 'image_filename', 'mask_filename','index'])
    return filenames_df

def to_numpy(x):
    """
    Converts a PyTorch tensor to a NumPy array and rearranges dimensions.

    Args:
        x (torch.Tensor): Input PyTorch tensor.

    Returns:
        np.ndarray: NumPy array with dimensions rearranged.
    """
    return x.to('cpu').detach().numpy().transpose(0, 2, 3, 1)

def denorm(x, mean, std):
    """
    Converts a PyTorch tensor to a NumPy array and rearranges dimensions.

    Args:
        x (torch.Tensor): Input PyTorch tensor.

    Returns:
        np.ndarray: NumPy array with dimensions rearranged.
    """

    return (x * std) + mean

def save(ckpt_dir, model, optim, epoch, iou, f1):
    """
    Save torch model in check point directory.
    Args: 
        ckpt_dir (str): checkpoint directory path
        model (torch model) : Pytorch model weights
        optim (torch.optimizer): pytorch optimizer
        epoch (int): train epoch        
    """
    os.makedirs(ckpt_dir, exist_ok=True)

    torch.save({'model': deepcopy(model.state_dict()),
                'optim': deepcopy(optim.state_dict())},
                f"{ckpt_dir}/epoch_{epoch}_iou{round(iou,4)}_f1{round(f1,4)}.pth")

def load(ckpt_dir, model, optim):
    """
        Load torch model in check point directory.
    """
    if os.path.exists(ckpt_dir):
        epoch = 0
        return model, optim, epoch
    
    ckpt_lst = os.listdir(ckpt_dir)
    ckpt_lst.sort(key=lambda f: int("".join(filter(str.isdigit, f))))

    dict_model = torch.load(f"{ckpt_dir}/{ckpt_lst[-1]}")   # load laetset model
    model.load_state_dict(dict_model['model'])
    optim.load_state_dict(dict_model['optim'])
    epoch = int(ckpt_lst[-1].split('epoch')[1].split('.pth')[0])

    return model, optim, epoch


def seed_fix(SEED):
    # Seed 
    os.environ['SEED'] = str(SEED)
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    np.random.seed(SEED)
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    np.random.seed(SEED)
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(SEED)


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
    target = (target >= threshold).float()

    # F1
    f1 = BinaryF1Score()
    f1_score = f1(preds_binary, target)
    # IoU
    IoU = BinaryJaccardIndex()
    iou_score = IoU(preds_binary, target)    

    return f1_score, iou_score



def cal_metrics_test(preds, target, threshold=0.5):
    """
        Calculate DL metrics

        Parameters:
            preds (numpy array): model prediction results
            target (numpy array): ground truth
            threshold (float): threhold 
        Returns:
            F1 score, Iou score, Precision, Recall, Sensitivity, Specificity
    """
    preds_binary = (preds > threshold).astype(np.float32)
    preds_binary, target = torch.tensor(preds_binary), torch.tensor(target)
    preds_flatten, target_flatten = preds_binary.flatten(), target.flatten()

    # F1
    f1 = BinaryF1Score()
    f1_score = f1(preds_binary, target)
    # IoU
    IoU = BinaryJaccardIndex()
    iou_score = IoU(preds_binary, target)    

    # Precision
    prec = precision_score(target_flatten, preds_flatten)
    # Recall
    rec = recall_score(target_flatten, preds_flatten)

    return f1_score, iou_score, prec, rec
