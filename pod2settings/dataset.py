import numpy as np
from numpy.lib import recfunctions as rfn
import scipy.ndimage
from pathlib import Path
import imageio.v3 as imageio
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset as BaseDataset
import albumentations as albu

def get_training_augmentation():
    train_transform = [
        albu.HorizontalFlip(p=0.5),
        albu.VerticalFlip(p=0.5),
        albu.ShiftScaleRotate(scale_limit=0.1, rotate_limit=10., shift_limit=0.1, p=1., border_mode=1),
        albu.PadIfNeeded(min_height=768, min_width=960, always_apply=True, border_mode=1)
    ]
    return albu.Compose(train_transform)

def get_validation_augmentation():
    train_transform = [
        albu.PadIfNeeded(min_height=768, min_width=960, always_apply=True, border_mode=1)
    ]
    return albu.Compose(train_transform)

def get_obj_classes():
    classes = {
        'NoBone' : 0,
        'RibBone' : 1
    }
    return classes
    
class BoneSegmentationDataset(BaseDataset):
    def __init__(self, data_folders, subset):
        self.fnames = []
        self.labels = []
        self.stats = np.array([])
        self.num_channels = len(data_folders)
        obj_classes = get_obj_classes()
        print(data_folders)
        
        for class_folder in obj_classes.keys():
            class_fnames = sorted((data_folders[0] / subset / class_folder).glob('*.tiff'))
            num_objs = len(class_fnames)
            class_labels = [obj_classes[class_folder]] * num_objs
            
            class_stats = np.genfromtxt(data_folders[0] / subset / '{}.csv'.format(class_folder), delimiter=',', names=True)
            print(class_stats.dtype)
            if self.stats.size == 0:
                self.stats = class_stats
            else:
                self.stats = rfn.stack_arrays([self.stats, class_stats])
            
            self.fnames.extend(class_fnames)
            self.labels.extend(class_labels)
            
        print('labels')
        print(self.labels)
            
        self.subset = subset
        self.data_folders = data_folders
        # automated chicken thresholding
        self.thr = 0.05
                                
    def __getitem__(self, i):
        inp = []
        fname = self.fnames[i].name
        class_folder = self.fnames[i].parts[-2]
        for data_folder in self.data_folders:
            img = imageio.imread(data_folder / self.subset / class_folder / fname)
            inp.append(img)
        
        inp = np.array(inp)
        inp = np.moveaxis(inp, 0, 2)
        
        segm_path = self.fnames[i].parents[1] / '{}_segm'.format(self.fnames[i].parts[-2]) / self.fnames[i].name
        mask = imageio.imread(segm_path)
        mask = np.moveaxis(mask, 0, 2)
                                    
        if self.subset == 'train':
            augmentation = get_training_augmentation()
            transformed = augmentation(image=inp, mask=mask)
            inp = transformed['image']
            # albumentation is HxWxC while the model expects CxHxW
            inp = np.moveaxis(inp, 2, 0)
            mask = transformed['mask']
            mask = np.moveaxis(mask, 2, 0)
        elif self.subset == 'val' or self.subset == 'test':
            augmentation = get_validation_augmentation()
            transformed = augmentation(image=inp, mask=mask)
            inp = transformed['image']
            inp = np.moveaxis(inp, 2, 0)
            mask = transformed['mask']
            mask = np.moveaxis(mask, 2, 0)
            
        if inp.shape[0] == 2:
            q = np.divide(inp[0], inp[1], out=np.zeros_like(inp[0]), where=inp[1]!=0)
        else:
            q = None
            
        for k in range(inp.shape[0]):
            inp[k,:] -= inp[k,:].min()
            inp[k,:] /= inp[k,:].max()
                                
        return {
            'input' : inp,
            'label' : self.labels[i],
            'FO_size' : self.stats['FO_size'][i],
            'Attenuation' : self.stats['Attenuation'][i],
            'Contrast' : self.stats['Contrast'][i],
            'mask' : mask,
            'quotient' : q,
            'img_id' : i
        }
            
    def __len__(self):
        return len(self.fnames)
