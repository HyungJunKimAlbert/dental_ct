# Base Package
import os, warnings, math
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
# Torch
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from sklearn.model_selection import train_test_split
import segmentation_models_pytorch as smp
import torch.optim as optim
import albumentations as A

# parser
import argparse
# Customized function
from dataset.dataset import NiiDataset
from utils.util import save, load, seed_fix, get_file_row_nii, to_numpy, cal_metrics
from models.model import AttentionUNet

def set_args():
    parser = argparse.ArgumentParser(description="Test Segmentation Model", 
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--batch_size", default=4, type=int, dest="batch_size")
    parser.add_argument('--classes', default=32, type=int,  dest='classes')
    parser.add_argument("--data_dir", default="/home/hjkim/projects/Res_UNET/notebook/Data/nii", type=str, dest='data_dir')
    parser.add_argument("--model_dir", default="/home/hjkim/projects/Res_UNET/checkpoint/Attn_UNET_bce/epoch_29_iou0.78_f10.87.pth", type=str, dest='model_dir')

    return parser.parse_args()

def valid_one_epoch(model, data_loader, criterion, device):
    model.eval()
    loss_arr, iou_arr, f1_arr, prec_arr, rec_arr = [], [], [], [], []

    with torch.no_grad():
        for batch_idx, data in tqdm(enumerate(data_loader)):
            input, label = data
            input, label = input.to(device), label.to(device)
            output = model(input)
            loss = criterion(output, label)

            loss_arr += [loss.item()] # Mean value (total 32 channels)

            # For Tensorboard
            input = to_numpy(input)         # Batch, Height, Weight, Channels
            label = to_numpy(label)   
            output = to_numpy(output)            

            # F1-score , IoU score
            f1_score, iou_score, prec, rec, sensitivity, specificity = cal_metrics(output, label)
            iou_arr += [iou_score.item()]     
            f1_arr += [f1_score.item()]
            prec_arr += [prec.item()]
            rec_arr += [rec.item()]

        avg_loss = np.mean([x for x in loss_arr if math.isnan(x)!=True])
        avg_iou = np.mean([x for x in iou_arr if math.isnan(x)!=True])
        avg_f1 = np.mean([x for x in f1_arr if math.isnan(x)!=True])

    return avg_loss, avg_iou, avg_f1, np.mean(prec_arr), np.mean(rec_arr)


def main():

# 0. Fix seed and Set args
    SEED=42
    IMAGE_SIZE = 256

    seed_fix(SEED)
    args = set_args()

# 1. Define option
    CLASSES = [f'Segment_{i}' for i in range(1, args.classes + 1)]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # transforms method
    transform = {
        'valid': transforms.Compose([
        transforms.Lambda(lambda x: (x - x.min()) / (x.max() - x.min()) if x.min() != x.max() else x),
        transforms.ToTensor(),
        transforms.Resize([IMAGE_SIZE, IMAGE_SIZE], interpolation=0),
        ])
    }

    # Path define
    model_dir = args.model_dir
    # model parametersßß
    BATCH_SIZE = args.batch_size

    # model = smp.UnetPlusPlus(     # UnetPlusPlus
    # encoder_name="resnet34",  
    # encoder_weights="imagenet",
    # in_channels=1,
    # activation='softmax2d',
    # classes=32).to(device)
    model = AttentionUNet().to(device)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters())

    saved_checkpoint = torch.load(args.model_dir, map_location=device)
    model.load_state_dict(saved_checkpoint['model'])
    optimizer.load_state_dict(saved_checkpoint['optim'])

    criterion = nn.BCELoss()
    print(f"DEVICE: {device}")
    print(f"model_dir: { model_dir }")

# 3. Import dataset
    files_df = get_file_row_nii(args.data_dir)
    train_df, test_df = train_test_split(files_df, test_size=0.2, random_state=SEED)    # 8:2
    valid_df, test_df = train_test_split(test_df, test_size=0.1, random_state=SEED)     # 5:5
    
    # test_df = pd.read_csv("/home/hjkim/projects/Res_UNET/notebook/testset.csv", index_col=0)
    test_dataset = NiiDataset(test_df, transform=transform['valid'], classes=CLASSES)
    
    # Loader
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 4. Start Evaluation

    print(f"Test Total Batches: { len(test_loader) }")
    print('Testing Processing....')
    avg_loss_test, avg_iou_test, avg_f1_test, avg_prec, avg_rec = valid_one_epoch(model, test_loader, criterion, device)
    print(f"Test Loss: {avg_loss_test}, IoU: {avg_iou_test}, F1-score: {avg_f1_test}, Precision: {avg_prec}, Recall: {avg_rec}")

if __name__ == '__main__':

    main()

