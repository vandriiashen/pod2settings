import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tifffile
import imageio.v3 as imageio
import shutil
import pickle
import statsmodels.api as sm

import cv2
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision import models
import pytorch_lightning as L
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import pod2settings as pts

def dsegm_train(data, iteration):
    log_folder = Path('../log')
    ckpt_folder = Path('../ckpt')
    
    data_folders = data['train']
    
    train_name = '{}_{}'.format(data['arch'], data['train'][0].parts[-1])
    '''
    data_folders = []
    for subfolder in subfolders:
        data_folders.append(root_folder / subfolder)
    '''
    train_ds = pts.dataset.BoneSegmentationDataset(data_folders, 'train')
    train_dataloader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=8)
    val_ds = pts.dataset.BoneSegmentationDataset(data_folders, 'val')
    val_dataloader = DataLoader(val_ds, batch_size=2, shuffle=False, num_workers=8)
    
    tb_logger = TensorBoardLogger(save_dir=log_folder, name='{}_dsegm_{:02d}'.format(train_name, iteration))
    checkpoint_callback = ModelCheckpoint(dirpath=ckpt_folder / '{}_dsegm_{:02d}'.format(train_name, iteration), save_top_k=1, monitor="val_loss")
    trainer = L.Trainer(max_epochs=300, callbacks=[checkpoint_callback], logger=[tb_logger])
    model = pts.SegmentationModel(arch = 'DeepLabV3Plus', encoder = 'tu-efficientnet_b0', num_channels = 2, num_classes = 2)
    #model = pts.SegmentationModel(arch = 'DeepLabV3Plus', encoder = 'timm-mobilenetv3_small_minimal_100', num_channels = 2, num_classes = 2, encoder_depth=5, encoder_weights=None)
    trainer.fit(model, train_dataloader, val_dataloader)
    
def make_comp(i, inp, tg, pred):
    im_tg = np.zeros_like(tg[0,:])
    im_tg[tg[0,:] > 0] = 1
    im_tg[tg[1,:] > 0] = 2
    
    im_pred = np.zeros_like(pred[0,:])
    im_pred[pred[0,:] > 0] = 1
    im_pred[pred[1,:] > 0] = 2
    
    fig, ax = plt.subplots(1, 3, figsize=(16,5))
    ax[0].imshow(inp[0,:])
    ax[0].set_title('Input', fontsize=20)
    ax[1].imshow(im_tg)
    ax[1].set_title('Ground-Truth', fontsize=20)
    ax[2].imshow(im_pred)
    ax[2].set_title('Prediction', fontsize=20)
    plt.tight_layout()
    plt.savefig('tmp_imgs/segm/{:03d}.png'.format(i))
    plt.close()
    
def make_segm_map(imgs, tgs, preds):
    fig, ax = plt.subplots(1, 1, figsize = (12,9))
    
    show_img = np.zeros((*imgs[0,:].shape, 3))
    print(show_img.shape)
    for k in range(3):
        show_img[:,:,k] = imgs.mean(axis=0)
        
    for j in range(tgs.shape[0]):
        tp = np.logical_and(tgs[j,:] == 1, preds[j,:] == 1)
        fn = np.logical_and(tgs[j,:] == 1, preds[j,:] == 0)
        
        show_img[tp,1] = 1
        show_img[fn,0] = 1
    
    ax.imshow(show_img)
    ax.set_axis_off()
    plt.savefig('./tmp_imgs/chicken_data_pod/segm_map.pdf', format='pdf')
    
def apply_dsegm(data):
    ckpt_folder = Path('../ckpt')

    data_folders = data['test']
    
    train_name = data['train'][0].parts[-1]
    param_path = sorted((ckpt_folder / '{}_dsegm'.format(train_name)).glob('*.ckpt'))[-1]
    print(param_path)
    
    model = pts.SegmentationModel.load_from_checkpoint(param_path)
    
    test_ds = pts.dataset.BoneSegmentationDataset(data_folders, 'test')
    test_dataloader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=1)
    
    conf_mat = np.zeros((2,2))
    
    imgs = []
    tgs = []
    preds = []
    
    for i, data in enumerate(test_dataloader):
        if i > 40:
            break
        with torch.no_grad():
            model.eval()
            logits = model(data['input'].cuda())
        prediction = torch.where(logits.sigmoid() > 0.5, 1, 0)
        prediction_numpy = prediction.detach().cpu().numpy()[0,:]
        inp_numpy = data['input'].detach().cpu().numpy()[0,:]
        tg_numpy = data['mask'].detach().cpu().numpy()[0,:]
        
        tp = np.count_nonzero(np.logical_and(tg_numpy[1,:] == 1, prediction_numpy[1,:] == 1))
        fn = np.count_nonzero(np.logical_and(tg_numpy[1,:] == 1, prediction_numpy[1,:] == 0))
        
        pred_cl = 0
        if prediction_numpy[1,:].mean() > 0:
            if tg_numpy[1,:].mean() == 0:
                pred_cl = 1
            elif tp / (tp + fn) > 0.1:
                pred_cl = 1
        tg_cl = 0
        if tg_numpy[1,:].mean() > 0:
            tg_cl = 1
            
        conf_mat[tg_cl][pred_cl] += 1
        
        if tg_cl != pred_cl:
            print('{}, {}mm: True {} | Pred {}'.format(i, float(data['FO_size']), tg_cl, pred_cl))
        
        make_comp(i, inp_numpy, tg_numpy, prediction_numpy)
        
        imgs.append(inp_numpy[0,:,:])
        tgs.append(tg_numpy[1,:,:])
        preds.append(prediction_numpy[1,:,:])
        
    imgs = np.array(imgs)
    tgs = np.array(tgs)
    preds = np.array(preds)
    make_segm_map(imgs, tgs, preds)
        
    print('Confusion matrix')
    print(conf_mat)
    
def test_dsegm(data, iteration):
    ckpt_folder = Path('../ckpt')

    data_folders = data['test']
    
    #train_name = data['train'][0].parts[-1]
    train_name = '{}_{}'.format(data['arch'], data['train'][0].parts[-1])
    print(ckpt_folder / '{}_dsegm_{:02d}'.format(train_name, iteration))
    param_path = sorted((ckpt_folder / '{}_dsegm_{:02d}'.format(train_name, iteration)).glob('*.ckpt'))[-1]
    print(param_path)
    
    model = pts.SegmentationModel.load_from_checkpoint(param_path)
    
    test_ds = pts.dataset.BoneSegmentationDataset(data_folders, 'test')
    test_dataloader = DataLoader(test_ds, batch_size=2, shuffle=False, num_workers=8)
    
    trainer = L.Trainer()
    trainer.test(model, test_dataloader)
    shutil.move('./tmp_res/tmp.csv', './tmp_res/{}_train{}_{:02d}-test{}.csv'.format(data['arch'], data['train'][0].parts[-1], iteration, data['test'][0].parts[-1]))
    
def compose_pods():
    pod_imgs = []
    for i in range(4):
        pod_imgs.append(imageio.imread('./tmp_imgs/chicken_data_pod/pod_{}.png'.format(i)))
    
    fig, ax = plt.subplots(2, 2, figsize = (12,9))
    
    ax[0,0].imshow(pod_imgs[0])
    ax[0,0].set_axis_off()

    ax[0,1].imshow(pod_imgs[1])
    ax[0,1].set_axis_off()
    ax[1,0].imshow(pod_imgs[2])
    ax[1,0].set_axis_off()
    ax[1,1].imshow(pod_imgs[3])
    ax[1,1].set_axis_off()
    
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('./tmp_imgs/chicken_data_pod/pod_comparison.png')
    
if __name__ == "__main__":
    # Albumentations OpenCV fix
    
    cv2.setNumThreads(0)
    cv2.ocl.setUseOpenCL(False)
    
    train_folder = Path('/export/scratch2/vladysla/Data/Real/POD/datasets')
    test_folder = Path('/export/scratch2/vladysla/Data/Real/POD/datasets_var')
    data = [
        {
            'train' : [train_folder / '40kV_40W_100ms_10avg', train_folder / '90kV_45W_100ms_10avg'],
            'test' : [test_folder / '40kV_40W_100ms_10avg', test_folder / '90kV_45W_100ms_10avg']
        },
        {
            'train' : [train_folder / 'gen_40kV_40W_100ms_1avg', train_folder / 'gen_90kV_45W_100ms_1avg'],
            'test' : [test_folder / '40kV_40W_100ms_1avg', test_folder / '90kV_45W_100ms_1avg'],
            'arch' : 'eff'
        },
        {
            'train' : [train_folder / 'gen_40kV_40W_50ms_1avg', train_folder / 'gen_90kV_45W_50ms_1avg'],
            'test' : [test_folder / '40kV_40W_50ms_1avg', test_folder / '90kV_45W_50ms_1avg'],
            'arch' : 'eff'
        },
        {
            'train' : [train_folder / 'gen_40kV_40W_20ms_1avg', train_folder / 'gen_90kV_45W_20ms_1avg'],
            'test' : [test_folder / '40kV_40W_20ms_1avg', test_folder / '90kV_45W_20ms_1avg'],
            'arch' : 'eff'
        },
                {
            'train' : [train_folder / 'gen_40kV_40W_100ms_1avg', train_folder / 'gen_90kV_45W_100ms_1avg'],
            'test' : [test_folder / 'gen_40kV_40W_100ms_1avg', test_folder / 'gen_90kV_45W_100ms_1avg'],
            'arch' : 'eff'
        },
        {
            'train' : [train_folder / 'gen_40kV_40W_50ms_1avg', train_folder / 'gen_90kV_45W_50ms_1avg'],
            'test' : [test_folder / 'gen_40kV_40W_50ms_1avg', test_folder / 'gen_90kV_45W_50ms_1avg'],
            'arch' : 'eff'
        },
        {
            'train' : [train_folder / 'gen_40kV_40W_20ms_1avg', train_folder / 'gen_90kV_45W_20ms_1avg'],
            'test' : [test_folder / 'gen_40kV_40W_20ms_1avg', test_folder / 'gen_90kV_45W_20ms_1avg'],
            'arch' : 'eff'
        },
        {
            'train' : [train_folder / 'gen_40kV_40W_200ms_1avg', train_folder / 'gen_90kV_45W_200ms_1avg'],
            'test' : [test_folder / 'gen_40kV_40W_200ms_1avg', test_folder / 'gen_90kV_45W_200ms_1avg']
        },
        {
            'train' : [train_folder / 'gen_40kV_40W_500ms_1avg', train_folder / 'gen_90kV_45W_500ms_1avg'],
            'test' : [test_folder / 'gen_40kV_40W_500ms_1avg', test_folder / 'gen_90kV_45W_500ms_1avg']
        }
    ]
    
    '''
    for i in range(len(data)):
    for i in range(6,9):
        test_dsegm(data[i])
    '''
    #apply_dsegm(data[1])

    #test_dsegm(data[1], 0)
    for i in range(0, 10):
    #    dsegm_train(data[1], i)
        test_dsegm(data[6], i)
    
    '''
    for i in range(8, 9):
        for j in range(10):
            dsegm_train(data[i], j)
    '''
    #dsegm_train(train_folder, ['gen_40kV_40W_500ms_1avg', 'gen_90kV_45W_500ms_1avg'])
    #dsegm_train(train_folder, ['gen_40kV_40W_200ms_1avg', 'gen_90kV_45W_200ms_1avg'])
