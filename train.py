# Base Package
import os, warnings, math
import numpy as np
warnings.filterwarnings('ignore')
# Torch
import torch
import torch.nn as nn
import torchvision
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import segmentation_models_pytorch as smp
# parser
import argparse
# etc
from torch.utils.tensorboard import SummaryWriter       # tensorboard --logdir="/home/hjkim/projects/dental_ct/log" --port=6009
from sklearn.model_selection import train_test_split
# Customized function
from dataset.dataset import NiiDataset
from utils.util import save, load, seed_fix, get_file_row_nii, to_numpy, cal_metrics
from models.model import UNet, AttentionUNet,DuckNet
# from losses.losses import DiceLoss, DiceBCELoss, IoULoss, FocalLoss


def set_args():
    parser = argparse.ArgumentParser(description="Train Segmentation Model", 
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--lr", default=1e-3, type=float, dest="lr")
    parser.add_argument("--batch_size", default=4, type=int, dest="batch_size")
    parser.add_argument("--num_epoch", default=50, type=int, dest='num_epoch')
    parser.add_argument('--classes', default=32, type=int,  dest='classes')

    parser.add_argument("--data_dir", default="/home/hjkim/projects/local_dev/dental_ai/nii_origin", type=str, dest='data_dir')
    parser.add_argument("--ckpt_dir", default="./checkpoint", type=str, dest="ckpt_dir")
    parser.add_argument("--log_dir", default="./log", type=str, dest='log_dir')
    parser.add_argument("--result_dir", default="./result", type=str, dest="result_dir")

    parser.add_argument("--mode", default="train", type=str, dest="mode")
    parser.add_argument("--train_continue", default="off", type=str, dest="train_continue")
    
    return parser.parse_args()


def train_one_epoch(model, dataloader, criterion, optimizer, device, writer_train):
    model.train()
    loss_arr = []
    iou_arr = []
    f1_arr = []

    for batch_idx, data in tqdm(enumerate(dataloader)):
        input, label = data # Input: torch.Size([4, 1, 512, 512]), Label: torch.Size([4, 32, 512, 512])
        input, label = input.to(device), label.to(device)

        output = model(input)   # Output :torch.Size([4, 32, 512, 512]), Loss: torch.Size([4, 32])
        optimizer.zero_grad()
        
        loss = criterion(output, label) 
        
        loss_arr += [loss.item()] # Mean value (total 32 channels)
        loss.backward()
        optimizer.step()
        # For Tensorboard
        input = to_numpy(input)         # Batch, Height, Weight, Channels
        label = to_numpy(label)   
        output = to_numpy(output)  

        # F1-score , IoU score
        try:
            f1_score, iou_score = cal_metrics(output, label)
            iou_arr += [iou_score.item()]     
            f1_arr += [f1_score.item()]
        except Exception as e:
            print(e, type(e))
            pass

        # writer_train.add_image('input', input, batch_idx, dataformats='NHWC')
        # writer_train.add_image('label', label_fn, batch_idx, dataformats='NHWC')
        # writer_train.add_image('output', output_fn, batch_idx, dataformats='NHWC')

    avg_loss = np.mean(loss_arr)
    avg_iou = np.mean([x for x in iou_arr if math.isnan(x)!=True])
    avg_f1 = np.mean([x for x in f1_arr if math.isnan(x)!=True])
        
    return avg_loss, avg_iou, avg_f1

def valid_one_epoch(model, data_loader, criterion, device, writer_valid, epoch, result_dst_path):
    model.eval()
    loss_arr = []
    iou_arr = []
    f1_arr = []
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

            # concat 32 channels data (label, output) ==> for tensorboard image (concat all images vertically)
            # output
            output_fn = output.copy()
            output_fn[output_fn<0.1] = 0
            output_fn = output_fn.sum(axis=-1, keepdims=True)  # 32ch --> 3ch 
            output_fn = np.concatenate([output_fn] * 3, axis=-1)
            # label
            label_fn = label.copy()
            label_fn = label_fn.sum(axis=-1, keepdims=True)   # 32ch --> 3ch 
            label_fn = np.concatenate([label_fn] * 3, axis=-1)

            # F1-score , IoU score
            try:
                f1_score, iou_score = cal_metrics(output, label)
                iou_arr += [iou_score.item()]     
                f1_arr += [f1_score.item()]
            except Exception as e:
                print(e, type(e))
                pass

            writer_valid.add_image('input', input, batch_idx, dataformats='NHWC')
            writer_valid.add_image('label', label_fn, batch_idx, dataformats='NHWC')
            writer_valid.add_image('output', output_fn, batch_idx, dataformats='NHWC')
        # np.save(f"{result_dst_path}/input_{epoch}ep.npy" ,input)
        # np.save(f"{result_dst_path}/label_{epoch}ep.npy" ,label)
        # np.save(f"{result_dst_path}/output_{epoch}ep.npy", output)

        avg_loss = np.mean(loss_arr)
        avg_iou = np.mean([x for x in iou_arr if math.isnan(x)!=True])
        avg_f1 = np.mean([x for x in f1_arr if math.isnan(x)!=True])
        
    return avg_loss, avg_iou, avg_f1


def main():

# 0. Fix seed and Set args
    SEED=42
    IMAGE_SIZE = 256
    early_stop_counter = 0 
    early_stopping_epochs = 10

    seed_fix(SEED)
    args = set_args()

# 1. Define option
    CLASSES = [f'Segment_{i}' for i in range(1, args.classes + 1)]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # transforms method
    transform = {
        'train': transforms.Compose([
        transforms.Lambda(lambda x: (x - x.min()) / (x.max() - x.min()) if x.min() != x.max() else x),
        transforms.ToTensor(),
        transforms.Resize([IMAGE_SIZE, IMAGE_SIZE], interpolation=0),
    ]),

        'valid': transforms.Compose([
        transforms.Lambda(lambda x: (x - x.min()) / (x.max() - x.min()) if x.min() != x.max() else x),
        transforms.ToTensor(),
        transforms.Resize([IMAGE_SIZE, IMAGE_SIZE], interpolation=0),
        ])
    }

    # Path define
    data_dir = args.data_dir
    ckpt_dir = args.ckpt_dir
    log_dir = args.log_dir
    result_dir = args.result_dir
    # model parameters
    lr = args.lr
    BATCH_SIZE = args.batch_size
    NUM_EPOCHS = args.num_epoch
    FILTERS = 16

    model = DuckNet(FILTERS).to(device) # Customized UNET with attention (activation : Softmax 2d)

    # model = smp.Unet(     # UnetPlusPlus
    # # encoder_name="resnet34",
    # encoder_name="resnext50_32x4d",  
    # encoder_weights="imagenet",
    # in_channels=1,
    # activation='softmax2d',
    # classes=32).to(device)

    criterion = nn.BCELoss()
    # criterion = FocalLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
    mode = args.mode
    train_continue = args.train_continue

    print(f"DEVICE: {device}")
    print(f"Mode: { mode }, Train_continue: { train_continue }, Learning Rate: { lr }, Batch size: { BATCH_SIZE }, NUM_EPOCHS: { NUM_EPOCHS }")
    print(f"data_dir: { data_dir }")
    print(f"ckpt_dir: { ckpt_dir }")
    print(f"log_dir: { log_dir }")
    print(f"result_dir: { result_dir }")

# 3. Import dataset
    files_df = get_file_row_nii(args.data_dir)

    train_df, test_df = train_test_split(files_df, test_size=0.2, random_state=SEED)    # 8:2
    valid_df, test_df = train_test_split(test_df, test_size=0.5, random_state=SEED)     # 5:5

    train_dataset = NiiDataset(train_df, transform=transform['train'], classes=CLASSES)
    valid_dataset = NiiDataset(valid_df, transform=transform['valid'], classes=CLASSES)
    # test_dataset = NiiDataset(test_df, transform=transform['valid'], classes=CLASSES)
    
    # Loader
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
    # test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    if mode == 'train': # TRAIN MODE
        ST_EPOCH = 0
        if train_continue == "on": # load weights if you want to continue training
            model, optim, ST_EPOCH = load(ckpt_dir=ckpt_dir, model=model, optim=optim)

# 4. Start Training
                                      
        # Tensorboard
        writer_train = SummaryWriter(log_dir = os.path.join(log_dir, 'train'))
        writer_valid = SummaryWriter(log_dir = os.path.join(log_dir, 'valid'))

        global_f1 = 0.0

        print(f"Train Total Batches: { len(train_loader)  }, Valid Total Batches: { len(valid_loader) }")

        for epoch in range(ST_EPOCH+1, NUM_EPOCHS+1):
            
            print('Training Processing....')
            avg_loss, avg_iou, avg_f1 = train_one_epoch(model, train_loader, criterion, optimizer, device, writer_train)
            print('Validation Processing...')
            avg_loss_valid, avg_iou_valid, avg_f1_valid = valid_one_epoch(model, valid_loader, criterion, device, writer_valid, epoch, result_dir)
            # Scheduler
            scheduler.step(avg_iou_valid)       # IoU 향상 없을 시,
            
            # train
            writer_train.add_scalar('loss', avg_loss, epoch)               # Loss (DICE)
            writer_train.add_scalar('IoU', avg_iou, epoch)                 # IoU
            writer_train.add_scalar('F1_score', avg_f1, epoch)             # F1 score
            
            # Valid
            writer_valid.add_scalar('loss', avg_loss_valid, epoch)
            writer_valid.add_scalar('IoU', avg_iou_valid, epoch)           # IoU
            writer_valid.add_scalar('F1_score', avg_f1_valid, epoch)       # F1 score
            print(f"EPOCH: {epoch}/{NUM_EPOCHS}, Train Loss: {avg_loss}, IoU: {avg_iou}, F1-score: {avg_f1} \n\tValid Loss: {avg_loss_valid}, IoU: {avg_iou_valid}, F1-score: {avg_f1_valid}")
        
            # Early stopping
            # if avg_f1_valid > global_f1:
            #     early_stop_counter += 1
            # else:
            #     global_f1 = avg_f1_valid
            #     early_stop_counter = 0

            # if early_stop_counter >= early_stopping_epochs:
            #     print("Early Stopping!")
            #     save(ckpt_dir=ckpt_dir, model=model, optim=optimizer, epoch=epoch, iou=avg_iou_valid, f1=avg_f1_valid)
            #     break

            # Save Model
            if global_f1 < avg_f1_valid:
                global_f1 = avg_f1_valid
                save(ckpt_dir=ckpt_dir, model=model, optim=optimizer, epoch=epoch, iou=avg_iou_valid, f1=avg_f1_valid)

        # Tensorboard close
        writer_train.close()
        writer_valid.close()

if __name__ == '__main__':

    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    main()

