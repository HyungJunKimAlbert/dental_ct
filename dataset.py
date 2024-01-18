import numpy as np
import cv2
import nibabel as nib
from torch.utils.data import Dataset 
from PIL import Image as Image
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

class NiiDataset(Dataset):
    '''
    dataset class overloads the __init__, __len__, __getitem__ methods of the Dataset class. 
    
    Parameters :
        df:  DataFrame object.
        data_path: Location of the dataset.
        transform: Transformations to apply to the image.
        train: A boolean indicating whether it is a training_set or not.
    '''

    CLASSES = [f'segment_{i}' for i in range(0, 33)]

    def __init__(self,df, classes=None, transform=None):
        super(Dataset, self).__init__()

        self.df = df
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
        self.transform = transform
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image = nib.load(row['image_filename']).get_fdata()[:, :, row['index']]
        image = image[..., np.newaxis].astype('float32')

        mask = nib.load(row['mask_filename']).get_fdata()[:, :, row['index']]
        
        # mask 데이터로부터 특정 클래스 추출
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float32')
        
        if self.transform:
            image = self.transform(image)   # Applies transformation to the image.
            mask = self.transform(mask)     # Applies transformation to the mask.

        return image, mask
