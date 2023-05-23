import numpy as np
import scipy.ndimage
from pathlib import Path
import imageio.v3 as imageio
import tifffile

import torch
from torch.utils.data import Dataset as BaseDataset
import albumentations as albu
from pod2settings import measure

def get_training_augmentation():
    train_transform = [
        albu.HorizontalFlip(p=0.5),
        albu.PadIfNeeded(min_height=640, min_width=320, always_apply=True, border_mode=1),
        albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0.2, shift_limit=0.1, p=1, border_mode=1),
        albu.PadIfNeeded(min_height=640, min_width=320, always_apply=True, border_mode=1),
        albu.RandomCrop(height=640, width=320, always_apply=True),
    ]
    return albu.Compose(train_transform)

def get_validation_augmentation():
    train_transform = [
        albu.PadIfNeeded(min_height=640, min_width=320, always_apply=True, border_mode=1),
        albu.RandomCrop(height=640, width=320, always_apply=True)
    ]
    return albu.Compose(train_transform)

def get_obj_classes():
    classes = {
        'NoBone' : 0,
        'FanBone' : 1,
        'RibBone' : 2,
        'WishBone' : 3
    }
    return classes

def log_cor(img):
    bg_mask = img > 58000
    img = img.astype(np.float32)
    img /= 65535
    img = -np.log(img)
    img[bg_mask] = 0.
    return img

class IndustrialDatasetGaussian(BaseDataset):
    def __init__(self, data_folder, subset, inp_mode, add_noise=0, augmentation=None):
        self.fnames_chA = []
        self.fnames_chB = []
        self.labels = []
        obj_classes = get_obj_classes()
        
        for class_folder in obj_classes.keys():
            class_fnames_chA = sorted((data_folder / subset / class_folder / 'chA').glob('*.tiff'))
            class_fnames_chB = sorted((data_folder / subset / class_folder / 'chB').glob('*.tiff'))
            num_objs = len(class_fnames_chA)
            class_labels = [obj_classes[class_folder]] * num_objs
            
            self.fnames_chA.extend(class_fnames_chA)
            self.fnames_chB.extend(class_fnames_chB)
            self.labels.extend(class_labels)
            
        assert inp_mode in ['A', 'B']
        self.add_noise = add_noise
        self.inp_mode = inp_mode
        self.subset = subset
        self.data_folder = data_folder
        
    def fname2id(self, fname, label):
        converted_name = '{:02d}{:04d}'.format(label+1, int(fname.stem))
        img_id = int(converted_name)
        return img_id
        
    def __getitem__(self, i):
        if self.inp_mode == 'A':
            inp = imageio.imread(self.fnames_chA[i])
            inp = log_cor(inp)
            inp = np.expand_dims(inp, 2)
        elif self.inp_mode == 'B':
            inp = imageio.imread(self.fnames_chB[i])
            inp = log_cor(inp)
            inp = np.expand_dims(inp, 2)
            
        if self.subset == 'train':
            augmentation = get_training_augmentation()
            transformed = augmentation(image=inp)
            inp = transformed['image']
            # albumentation is HxWxC while the model expects CxHxW
            inp = np.moveaxis(inp, 2, 0)
        elif self.subset == 'val' or self.subset == 'test':
            augmentation = get_validation_augmentation()
            transformed = augmentation(image=inp)
            inp = transformed['image']
            inp = np.moveaxis(inp, 2, 0)
            
        if self.add_noise != 0:
            noisy_pattern = np.random.normal(loc = 0., scale = self.add_noise, size=inp.shape)
            inp += noisy_pattern
            
        if self.subset == 'test' and self.labels[i] == 1:
            obj_classes = get_obj_classes()
            fname_segm = self.data_folder / self.subset / 'FanBone' / 'segm' / '{}.tiff'.format(self.fnames_chA[i].stem)
            segm = imageio.imread(fname_segm)
            fo_size = measure.get_size(segm, self.fname2id(self.fnames_chA[i], self.labels[i]))
        else:
            fo_size = 0
        
        return {
            'img_id' : self.fname2id(self.fnames_chA[i], self.labels[i]),
            'input' : inp,
            'label' : self.labels[i],
            'fo_size' : fo_size
        }
            
        
    def __len__(self):
        return len(self.fnames_chA)
    
class OldSimpleDataset(BaseDataset):
    def __init__(self, subset, add_noise=0):
        self.subset = subset
        self.add_noise = add_noise
        self.h = 640
        self.w = 320
        self.fo_impact = 0.02
        self.rect_impact = 0.4
        
        self.samples = 900
        if subset == 'train':
            np.random.seed(seed = 9)
        elif subset == 'val':
            np.random.seed(seed = 15)
        elif subset == 'test':
            np.random.seed(seed = 22)
        
        self.c = np.zeros((self.samples, 2))
        self.c[:,0] = np.random.uniform(200, 400, size=(self.samples,))
        self.c[:,1] = np.random.uniform(120, 240, size=(self.samples,))
        self.ax = np.zeros((self.samples, 2))
        self.ax[:,0] = np.random.uniform(70, 180, size=(self.samples,))
        self.ax[:,1] = np.random.uniform(40, 100, size=(self.samples,))
        self.add_c = np.zeros((self.samples, 2))
        self.add_c[:,0] = np.random.uniform(-0.7, 0.7, size=(self.samples,))
        self.add_c[:,1] = np.random.uniform(-0.7, 0.7, size=(self.samples,))
        self.add_ax = np.zeros((self.samples, 2))
        self.add_ax[:,0] = np.random.uniform(0.5, 1.5, size=(self.samples,))
        self.add_ax[:,1] = np.random.uniform(1.5, 1.5, size=(self.samples,))
        self.fo_c = np.zeros((self.samples, 2))
        self.fo_c[:,0] = np.random.uniform(-0.7, 0.7, size=(self.samples,))
        self.fo_c[:,1] = np.random.uniform(-0.7, 0.7, size=(self.samples,))
        self.fo_ax = np.zeros((self.samples, 2))
        sphere_r = np.random.uniform(5., 30., size=(300,))
        sphere_dif = np.random.uniform(-3., 3., size=(300,))
        self.fo_ax[300:600,0] = sphere_r
        self.fo_ax[300:600,1] = sphere_r + sphere_dif
        rect_l = np.random.uniform(10., 40., size=(300,))
        rect_w = np.random.uniform(3., 15., size=(300,))
        self.fo_ax[600:900,0] = rect_l
        self.fo_ax[600:900,1] = rect_w
        self.angle = np.random.uniform(0., 90., size=(self.samples,))
        self.label = np.zeros((self.samples,), dtype=np.uint8)
        self.label[300:600] = 1
        self.label[600:900] = 2
        
    def gen_rotated_rect(self, ax, angle):
        max_dim = int(max(ax))
        ax = [int(ax[0]), int(ax[1])]
        half_dim = max_dim//2
        im = np.zeros((max_dim, max_dim))
        im[half_dim-ax[0]//2 : half_dim+(ax[0]-ax[0]//2), half_dim-ax[1]//2 : half_dim+(ax[1]-ax[1]//2)] = 1.
        im = scipy.ndimage.rotate(im, angle, reshape=False)
        return im
        
    def gen_image(self, c, ax, add_c, add_ax, fo_c, fo_ax, angle, label):
        im = np.zeros((self.h, self.w), dtype=np.float32)
        
        X, Y = np.ogrid[:self.h,:self.w]
        dist = np.sqrt((X - c[0])**2 / ax[0]**2 + (Y - c[1])**2 / ax[1]**2)
        th = 1 - dist
        th[th < 0] = 0
        
        dist2 = np.sqrt((X - (c[0] + add_c[0]*ax[0]) )**2 / (add_ax[0]*ax[0])**2 + (Y - (c[1] + add_c[1]*ax[1]) )**2 / (add_ax[1]*ax[1])**2)
        th2 = 1 - dist2
        th2[th == 0] = 0
        th2[th2 < 0] = 0
        
        if label == 1: 
            fo_dist = np.sqrt((X - (c[0] + fo_c[0]*ax[0]) )**2 / fo_ax[0]**2 + (Y - (c[1] + fo_c[1]*ax[1]) )**2 / fo_ax[1]**2)
            fo_th = 1 - fo_dist
            fo_th[fo_th < 0] = 0
        elif label == 2:
            max_dim = int(max(fo_ax))
            rect = self.gen_rotated_rect(fo_ax, angle)
            fo_th = np.zeros_like(im)
            c_fo = [int(c[0] + fo_c[0]*ax[0]), int(c[1] + fo_c[1]*ax[1])]
            fo_th[c_fo[0]:c_fo[0]+max_dim, c_fo[1]:c_fo[1]+max_dim] = self.rect_impact * rect
        else:
            fo_th = 0
        
        im = th + th2 + self.fo_impact * np.sqrt(fo_ax[0]**2 + fo_ax[1]**2)*fo_th
        im = im.astype(np.float32)
        
        return im
        
    def __getitem__(self, i):
        img = self.gen_image(self.c[i,:], self.ax[i,:], self.add_c[i,:], self.add_ax[i,:], self.fo_c[i,:], self.fo_ax[i,:], self.angle[i], self.label[i])
        img = np.expand_dims(img, 0)
        #print(img.dtype)
        if self.label[i] == 1:
            fo_size = np.sqrt(self.fo_ax[i,0]**2 + self.fo_ax[i,1]**2)
        elif self.label[i] == 2:
            fo_size = max(self.fo_ax[i,:])
        else:
            fo_size = 0.
            
        if self.add_noise != 0:
            noisy_pattern = np.random.normal(loc = 0., scale = self.add_noise, size=img.shape)
            img += noisy_pattern
            
        return {
            'img_id' : i,
            'input' : img,
            'label' : self.label[i],
            'fo_size' : fo_size
        }
        
    def __len__(self):
        return self.samples
    
class SimpleGenerator():
    def __init__(self, subset, add_noise=0):
        self.subset = subset
        self.add_noise = add_noise
        self.h = 640
        self.w = 320
        self.fo_impact = 0.02
        self.rect_impact = 0.4
        
        self.samples = 900
        if subset == 'train':
            np.random.seed(seed = 9)
        elif subset == 'val':
            np.random.seed(seed = 15)
        elif subset == 'test':
            np.random.seed(seed = 22)
        
        self.c = np.zeros((self.samples, 2))
        self.c[:,0] = np.random.uniform(200, 400, size=(self.samples,))
        self.c[:,1] = np.random.uniform(120, 240, size=(self.samples,))
        self.ax = np.zeros((self.samples, 2))
        self.ax[:,0] = np.random.uniform(70, 180, size=(self.samples,))
        self.ax[:,1] = np.random.uniform(40, 100, size=(self.samples,))
        self.add_c = np.zeros((self.samples, 2))
        self.add_c[:,0] = np.random.uniform(-0.7, 0.7, size=(self.samples,))
        self.add_c[:,1] = np.random.uniform(-0.7, 0.7, size=(self.samples,))
        self.add_ax = np.zeros((self.samples, 2))
        self.add_ax[:,0] = np.random.uniform(0.5, 1.5, size=(self.samples,))
        self.add_ax[:,1] = np.random.uniform(1.5, 1.5, size=(self.samples,))
        self.fo_c = np.zeros((self.samples, 2))
        self.fo_c[:,0] = np.random.uniform(-0.7, 0.7, size=(self.samples,))
        self.fo_c[:,1] = np.random.uniform(-0.7, 0.7, size=(self.samples,))
        self.fo_ax = np.zeros((self.samples, 2))
        sphere_r = np.random.uniform(5., 30., size=(300,))
        sphere_dif = np.random.uniform(-3., 3., size=(300,))
        self.fo_ax[300:600,0] = sphere_r
        self.fo_ax[300:600,1] = sphere_r + sphere_dif
        rect_l = np.random.uniform(10., 40., size=(300,))
        rect_w = np.random.uniform(3., 15., size=(300,))
        self.fo_ax[600:900,0] = rect_l
        self.fo_ax[600:900,1] = rect_w
        self.angle = np.random.uniform(0., 90., size=(self.samples,))
        self.label = np.zeros((self.samples,), dtype=np.uint8)
        self.label[300:600] = 1
        self.label[600:900] = 2
        
    def gen_rotated_rect(self, ax, angle):
        max_dim = int(max(ax))
        ax = [int(ax[0]), int(ax[1])]
        half_dim = max_dim//2
        im = np.zeros((max_dim, max_dim))
        im[half_dim-ax[0]//2 : half_dim+(ax[0]-ax[0]//2), half_dim-ax[1]//2 : half_dim+(ax[1]-ax[1]//2)] = 1.
        im = scipy.ndimage.rotate(im, angle, reshape=False)
        return im
        
    def gen_image(self, c, ax, add_c, add_ax, fo_c, fo_ax, angle, label):
        im = np.zeros((self.h, self.w), dtype=np.float32)
        
        X, Y = np.ogrid[:self.h,:self.w]
        dist = np.sqrt((X - c[0])**2 / ax[0]**2 + (Y - c[1])**2 / ax[1]**2)
        th = 1 - dist
        th[th < 0] = 0
        
        dist2 = np.sqrt((X - (c[0] + add_c[0]*ax[0]) )**2 / (add_ax[0]*ax[0])**2 + (Y - (c[1] + add_c[1]*ax[1]) )**2 / (add_ax[1]*ax[1])**2)
        th2 = 1 - dist2
        th2[th == 0] = 0
        th2[th2 < 0] = 0
        
        if label == 1: 
            fo_dist = np.sqrt((X - (c[0] + fo_c[0]*ax[0]) )**2 / fo_ax[0]**2 + (Y - (c[1] + fo_c[1]*ax[1]) )**2 / fo_ax[1]**2)
            fo_th = 1 - fo_dist
            fo_th[fo_th < 0] = 0
        elif label == 2:
            max_dim = int(max(fo_ax))
            rect = self.gen_rotated_rect(fo_ax, angle)
            fo_th = np.zeros_like(im)
            c_fo = [int(c[0] + fo_c[0]*ax[0]), int(c[1] + fo_c[1]*ax[1])]
            fo_th[c_fo[0]:c_fo[0]+max_dim, c_fo[1]:c_fo[1]+max_dim] = self.rect_impact * rect
        else:
            fo_th = 0
        
        im = th + th2 + self.fo_impact * np.sqrt(fo_ax[0]**2 + fo_ax[1]**2)*fo_th
        im = im.astype(np.float32)
        
        return im
    
    def generate_all(self):
        root_folder = Path('../gen_data')
        dataset_folder = root_folder / 'simple_data_n{:0.2f}'.format(self.add_noise)
        dataset_folder.mkdir(exist_ok=True)
        subset_foldet = root_folder / 'simple_data_n{:0.2f}'.format(self.add_noise)
        subset_foldet.mkdir(exist_ok=True)
        
        label_to_folder = {
            0 : '0',
            1 : '1',
            2 : '2'
        }
        
        for label in label_to_folder.values():
            (subset_foldet / label).mkdir(exist_ok=True)
        label_counter = np.zeros((len(label_to_folder.keys())), dtype=np.int32)
        
        for i in range(self.samples):
            img = self.gen_image(self.c[i,:], self.ax[i,:], self.add_c[i,:], self.add_ax[i,:], self.fo_c[i,:], self.fo_ax[i,:], self.angle[i], self.label[i])
            tifffile.imwrite(subset_foldet / label_to_folder[self.label[i]] / '{:04d}.tiff'.format(label_counter[self.label[i]]), img)
            label_counter[self.label[i]] += 1