## ResNet34 + U-Net Segmentation model

### Project Outline
    - Task : dental CBCT Multi-class Segmentation (32 channels)
    - Logging : tensorboard (2.14.0 version)

### Dataset (*.nii)
    - Preprocessing Methods
        - RescaleSlope, RescaleInterupt
    - Data Split
        - Train:Validation:Test = 8:1:1
    - Create DataFrame for reading Nii filepaths
    - Image dimension shape : (Batch, Channel, Height, Width)
        - input(Batch, 1, 256, 256), mask(Batch, 32, 256, 256)
        - Masking : 0 or 1 (Binary masking)

### Model Architecture
    - ResNet34 + UNET
    - ResNet34 + UNET PlusPlus
    - ResNet34 + Attention + UNET

### Train
    - Dataset
        - Train 2180,  Valid 273,  Test 273
        - Image Preprocessing : Resize Image(256,256), Min-Max scaling(Each image)
    - Model Parameters
        - Epochs: 50
        - learning_rate: 1e-3
        - Optimizer: Adam
        - Loss: BCE Loss
        - Metric: IOU, F1-score
    
### Results

Method                   | Best Epoch | Loss | Jaccard(IoU) | F1-score 
------------------------ | ---------- | ---- | ------------ | -------- 
ResNet34 + U-Net         | 47         | BCE  | 0.8159       | 0.8971 
ResNet34 + UNET PlusPlus | 45         | BCE  | **0.8203**   | **0.8999** 
Attention + UNET         | 49         | BCE  | 0.7827       | 0.8689 
Attention + UNET         | 49         | DICE | 0.6896       | 0.8030 
VGG19 + UNET             | 33         | BCE  | 0.7487       | 0.8498 
Densenet121 + UNET       | 00         | BCE  | 0.0          | 0.0 
DUCK-NET(2024 SOTA)      | 00         | BCE  | 0.0          | 0.0 


* DUCK-NET PARAMETERS

    <small>------------------------------------------</small>  
    <small>Total params: 22,911,216</small>  
    <small>Trainable params: 22,911,216</small>  
    <small>Non-trainable params: 0</small>  
    <small>------------------------------------------</small>  
    <small>Input size (MB): 1.00</small>  
    <small>Forward/backward pass size (MB): 10187.50</small>  
    <small>Params size (MB): 87.40</small>  
    <small>Estimated Total Size (MB): 10275.90</small>  
    <small>------------------------------------------</small>  

### Futher study
    - Apply class weight (completed)
    - Change Loss function (completed)
    - Change Model architecture
        - Encoder : densenet121, mit_b0 (Mixed ViT)
        - Decoder : Unet++ (completed)
    - Apply Attention layer in Decoder blocks (completed)
    - DUCK-NET (SOTA model) Experiment
    - Change split method

### Post-hoc Analysis
    - label comparison
    - prediction fail analysis



