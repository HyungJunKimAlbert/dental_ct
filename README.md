# dental_ct
CBCT Multi-class Segmentation


# ResNet34 + U-Net Segmentation model

## Outline
    - nii나 tiff 파일이 저장된 경로를 인자로 전달(-f)
    - input_channel 및 class 또한 인자로 전달(-c, -i)
    - wandb 로그인 필요
    - 추가적인 부분은 train.py 참고

## Dataset (*.nii)
    - Preprocessing Methods : RescaleSlope, RescaleInterupt
    - Data Split : Train:Validation:Test= 8:1:1
    - Create DataFrame for reading Nii filepaths
    - Image dimension : (Batch, Channel, Height, Width)
    - input(Batch, 3, 512,512), mask(Batch, 32, 512, 512)
        - Masking : 0 or 1 (Binary masking)

## Model Architecture
    - ResNet32 + UNet

## Train
    - Dataset
        - Train 2180,  Valid 273,  Test 273
        - Image Preprocessing : Resize Image(256,256), Min-Max scaling(Each image)
    - Model Parameters
        Epochs: 50
        - learning_rate: 1e-3
        - Optimizer: Adam
        - Loss: BCE Loss
        - Metric: IOU, F1-score
    
## Results
    - Best F1-score: 0.90 
    - Bset IOU Score: 0.82

    
## Futher  study
    - apply class weight
    - change Loss function
    - change Model architecture
