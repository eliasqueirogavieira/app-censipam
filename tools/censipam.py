import torch 
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import io
from pathlib import Path
from typing import Tuple

import numpy  as  np

from skimage.io import imread
import torchvision.transforms.functional as TF 

from typing import Tuple, List, Union, Tuple, Optional
from semseg.augmentations import Compose, RandomHorizontalFlip, RandomResizedCrop, RandomVerticalFlip


def get_train_augmentation(size: Union[int, Tuple[int], List[int]], seg_fill: int = 0):
    return Compose([
        RandomHorizontalFlip(p=0.5),
        #RandomResizedCrop(size, scale=(0.5, 2.0), seg_fill=seg_fill),
        RandomVerticalFlip(p=0.5),
        #Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        Normalize(scaler=255)
    ])


def get_val_augmentation(size: Union[int, Tuple[int], List[int]]):
    return Compose([
        #Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        Normalize(scaler=255)
    ])


class Normalize:

    def __init__(self, scaler = 255):
        self.scaler = scaler
        

    def __call__(self, img: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        img = img.float()
        img /= self.scaler
        return img, mask



class Censipam(Dataset):

    CLASSES = [
        'background', 'deforestation' ]

    # same number as classes
    PALETTE = torch.tensor([
        [120, 120, 120], [180, 120, 120]
    ])

    def __init__(self, root: str, split: str = 'train', ignore_lbl: int = 0, transform = None) -> None:
        super().__init__()
        assert split in ['train', 'val']
        split = 'training' if split == 'train' else 'validation'
        self.transform = transform
        self.n_classes = len(self.CLASSES)
        self.ignore_label = ignore_lbl

        img_path = Path(root) / 'imgs'
        self.files = list(img_path.glob('*.tif'))
    
        if not self.files:
            raise Exception(f"No images found in {img_path}")
        print(f"Found {len(self.files)} {split} images.")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        img_path = str(self.files[index])
        lbl_path = str(self.files[index]).replace('imgs', 'labels').replace('.tif', '.png')

        image = np.transpose( imread(img_path), (2, 0, 1) )
        label = np.expand_dims( imread(lbl_path), 0 )
        label[label == 0] = 1
        label[label == 255] = 2

        image = torch.tensor(image)
        label = torch.tensor(label).type(torch.int16)
        
        if self.transform:
            image, label = self.transform(image, label)
        
        return image, label.squeeze().long() - 1
        #return image, label.long() - 1



if __name__ == '__main__':
    from semseg.utils.visualize import visualize_dataset_sample
    visualize_dataset_sample(ADE20K, '/home/sithu/datasets/ADEChallenge/ADEChallengeData2016')