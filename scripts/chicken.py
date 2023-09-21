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

def check_dataset(data_folder):
    ds = pts.dataset.BoneSegmentationDataset(data_folder, 'train')
    print(len(ds))
    for i in range(len(ds)):
        plt.imshow(ds[i]['input'][0,:])
        #print(ds[50]['label'])
        #print(ds[5]['original_ID'])
        #print(ds[5]['FO_size'])
        plt.show()
    
def segm_train(root_folder, subfolder):
    log_folder = Path('../log')
    ckpt_folder = Path('../ckpt')
    
    data_folder = root_folder / subfolder
    
    train_ds = pts.dataset.BoneSegmentationDataset([data_folder], 'train')
    train_dataloader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=8)
    val_ds = pts.dataset.BoneSegmentationDataset([data_folder], 'val')
    val_dataloader = DataLoader(val_ds, batch_size=2, shuffle=False, num_workers=8)
    
    tb_logger = TensorBoardLogger(save_dir=log_folder, name='{}_segm'.format(subfolder))
    checkpoint_callback = ModelCheckpoint(dirpath=ckpt_folder / '{}_segm'.format(subfolder), save_top_k=1, monitor="val_loss")
    trainer = L.Trainer(max_epochs=500, callbacks=[checkpoint_callback], logger=[tb_logger])
    model = pts.SegmentationModel(arch = 'DeepLabV3Plus', encoder = 'tu-efficientnet_b0', num_channels = 1, num_classes = 2)
    trainer.fit(model, train_dataloader, val_dataloader)
    
def dsegm_train(root_folder, subfolders):
    log_folder = Path('../log')
    ckpt_folder = Path('../ckpt')
    
    data_folders = []
    for subfolder in subfolders:
        data_folders.append(root_folder / subfolder)
    
    train_ds = pts.dataset.BoneSegmentationDataset(data_folders, 'train')
    train_dataloader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=8)
    val_ds = pts.dataset.BoneSegmentationDataset(data_folders, 'val')
    val_dataloader = DataLoader(val_ds, batch_size=2, shuffle=False, num_workers=8)
    
    tb_logger = TensorBoardLogger(save_dir=log_folder, name='{}_dsegm'.format(subfolders[0]))
    checkpoint_callback = ModelCheckpoint(dirpath=ckpt_folder / '{}_dsegm'.format(subfolders[0]), save_top_k=1, monitor="val_loss")
    trainer = L.Trainer(max_epochs=500, callbacks=[checkpoint_callback], logger=[tb_logger])
    model = pts.SegmentationModel(arch = 'DeepLabV3Plus', encoder = 'tu-efficientnet_b0', num_channels = 2, num_classes = 2)
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
    
def apply_segm(root_folder, subfolder):
    ckpt_folder = Path('../ckpt')

    data_folder = root_folder / subfolder
    
    param_path = sorted((ckpt_folder / '{}_segm'.format(subfolder)).glob('*.ckpt'))[-1]
    print(param_path)
    
    model = pts.SegmentationModel.load_from_checkpoint(param_path)
    
    test_ds = pts.dataset.BoneSegmentationDataset([data_folder], 'test')
    test_dataloader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=1)
    
    conf_mat = np.zeros((2,2))
    
    for i, data in enumerate(test_dataloader):
        with torch.no_grad():
            model.eval()
            logits = model(data['input'].cuda())
        prediction = torch.where(logits.sigmoid() > 0.5, 1, 0)
        prediction_numpy = prediction.detach().cpu().numpy()[0,:]
        inp_numpy = data['input'].detach().cpu().numpy()[0,:]
        tg_numpy = data['mask'].detach().cpu().numpy()[0,:]
        
        tp = np.count_nonzero(np.logical_and(tg_numpy[1,:] == 1, prediction_numpy[1,:] == 1))
        fn = np.count_nonzero(np.logical_and(tg_numpy[1,:] == 1, prediction_numpy[1,:] == 0))
        
        print(i, tp, fn)
        
        pred_cl = 0
        if prediction_numpy[1,:].mean() > 0:
            if tg_numpy[1,:].mean() == 0:
                pred_cl = 1
            elif tp / (tp + fn) > 0.1:
                pred_cl = 1
            else:
                print('Wrong location')
                
        tg_cl = 0
        if tg_numpy[1,:].mean() > 0:
            tg_cl = 1
            
        conf_mat[tg_cl][pred_cl] += 1
        
        if tg_cl != pred_cl:
            print('{}, {}mm: True {} | Pred {}'.format(i, float(data['FO_size']), tg_cl, pred_cl))
        
        make_comp(i, inp_numpy, tg_numpy, prediction_numpy)
        
    print('Confusion matrix')
    print(conf_mat)
    
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
    plt.savefig('./tmp_imgs/chicken_data_pod/b21_map_3.png')
    
def apply_dsegm(root_folder, subfolders):
    ckpt_folder = Path('../ckpt')

    data_folders = []
    for subfolder in subfolders:
        data_folders.append(root_folder / subfolder)
    
    if subfolders[0] == '40kV_40W_100ms_10avg' or subfolders[0].startswith('gen'):
        param_path = sorted((ckpt_folder / '{}_dsegm'.format(subfolders[0])).glob('*.ckpt'))[-1]
    else:
        param_path = sorted((ckpt_folder / 'gen_{}_dsegm'.format(subfolders[0])).glob('*.ckpt'))[-1]
    print(param_path)
    
    model = pts.SegmentationModel.load_from_checkpoint(param_path)
    
    test_ds = pts.dataset.BoneSegmentationDataset(data_folders, 'test')
    test_dataloader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=1)
    
    conf_mat = np.zeros((2,2))
    
    imgs = []
    tgs = []
    preds = []
    
    for i, data in enumerate(test_dataloader):
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
    
def test_segm(root_folder, subfolder):
    ckpt_folder = Path('../ckpt')

    data_folder = root_folder / subfolder
    
    param_path = sorted((ckpt_folder / '{}_segm'.format(subfolder)).glob('*.ckpt'))[-1]
    print(param_path)
    
    model = pts.SegmentationModel.load_from_checkpoint(param_path)
    
    test_ds = pts.dataset.BoneSegmentationDataset([data_folder], 'test')
    test_dataloader = DataLoader(test_ds, batch_size=2, shuffle=False, num_workers=1)
    
    trainer = L.Trainer()
    trainer.test(model, test_dataloader)
    
def test_dsegm(root_folder, subfolders):
    ckpt_folder = Path('../ckpt')

    data_folders = []
    for subfolder in subfolders:
        data_folders.append(root_folder / subfolder)
    
    if subfolders[0] == '40kV_40W_100ms_10avg' or subfolders[0].startswith('gen'):
        param_path = sorted((ckpt_folder / '{}_dsegm'.format(subfolders[0])).glob('*.ckpt'))[-1]
    else:
        param_path = sorted((ckpt_folder / 'gen_{}_dsegm'.format(subfolders[0])).glob('*.ckpt'))[-1]
    print(param_path)
    
    model = pts.SegmentationModel.load_from_checkpoint(param_path)
    
    test_ds = pts.dataset.BoneSegmentationDataset(data_folders, 'test')
    test_dataloader = DataLoader(test_ds, batch_size=2, shuffle=False, num_workers=8)
    
    trainer = L.Trainer()
    trainer.test(model, test_dataloader)
    
def check_train(root_folder, subfolder):
    log_folder = Path('../log')
    ckpt_folder = Path('../ckpt')
    
    data_folder = root_folder / subfolder
    
    train_ds = pts.dataset.BoneSegmentationDataset(data_folder, 'train')
    train_dataloader = DataLoader(train_ds, batch_size=10, shuffle=True, num_workers=8)
    val_ds = pts.dataset.BoneSegmentationDataset(data_folder, 'val')
    val_dataloader = DataLoader(val_ds, batch_size=10, shuffle=False, num_workers=8)
    
    tb_logger = TensorBoardLogger(save_dir=log_folder, name='{}_class'.format(subfolder))
    checkpoint_callback = ModelCheckpoint(dirpath=ckpt_folder / '{}_class'.format(subfolder), save_top_k=1, monitor="val_loss")
    trainer = L.Trainer(max_epochs=500, callbacks=[checkpoint_callback], logger=[tb_logger])
    model = pts.ClassificationModel(arch = 'ShuffleNet', num_channels = 1, num_classes = 2)
    trainer.fit(model, train_dataloader, val_dataloader)
    
def check_test(root_folder, subfolder):
    ckpt_folder = Path('../ckpt')

    data_folder = root_folder / subfolder
    
    param_path = sorted((ckpt_folder / '{}_class'.format(subfolder)).glob('*.ckpt'))[-1]
    print(param_path)
    
    model = pts.ClassificationModel.load_from_checkpoint(param_path)
    
    test_ds = pts.dataset.BoneSegmentationDataset(data_folder, 'val')
    print('ds ', test_ds.labels)
    test_dataloader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=8)
    trainer = L.Trainer()
    trainer.test(model, test_dataloader)
    
def comp_bbox(num, inp, tg, pred):
    colors = {1 : 'm', 2 : 'r'}
    
    fig, ax = plt.subplots(1, 2, figsize=(16,9))
    ax[0].imshow(inp[0,:])
    for i in range(len(tg['boxes'])):
        b = tg['boxes'][i]
        width = b[2] - b[0]
        height = b[3] - b[1]
        rect = patches.Rectangle((b[0], b[1]), width, height, linewidth=2, edgecolor=colors[tg['labels'][i]], facecolor='none')
        ax[0].add_patch(rect)
    ax[0].set_title('Ground-Truth', fontsize = 20)
    
    ax[1].imshow(inp[0,:])
    for i in range(len(pred['boxes'])):
        b = pred['boxes'][i]
        width = b[2] - b[0]
        height = b[3] - b[1]
        rect = patches.Rectangle((b[0], b[1]), width, height, linewidth=2, edgecolor=colors[pred['labels'][i]], facecolor='none')
        ax[1].add_patch(rect)
    ax[1].set_title('Prediction', fontsize = 20)
        
    plt.tight_layout()
    plt.savefig('tmp_imgs/detection/{:03d}.png'.format(num))
    plt.close()
    #plt.show()
    
def collate_fn(batch):
        inputs = list()
        targets = list()

        for b in batch:
            inputs.append(torch.from_numpy(b[0]))
            targets.append({'boxes' : b[1], 'labels' : b[2]})

        inputs = torch.stack(inputs, dim=0)
        return inputs, targets
    
    
def check_detection(data_folder):
    train_ds = pts.dataset.BoneDetectionDataset(data_folder, 'train')
    train_dataloader = DataLoader(train_ds, batch_size=2, shuffle=False, num_workers=1, collate_fn=collate_fn)
    
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
    num_classes = 3  # 1 class (wheat) + background
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    print(model.backbone.body.conv1)
    
    model.cuda()
    for batch in train_dataloader:        
        inputs = list(image.cuda() for image in batch[0])
        targets = [{k: v.cuda() if k =='labels' else v.float().cuda() for k, v in t.items()} for t in batch[1]]
        
        res = model(inputs, targets)
        print(res)
        break
    
    '''
    i = 80
    sample = ds[i]
    tg = {'boxes' : sample['boxes'], 'labels' : sample['labels']}
    comp_bbox(i, sample['input'], tg, sample['mask'])
    '''
    
def det_train(root_folder, subfolder):
    log_folder = Path('../log')
    ckpt_folder = Path('../ckpt')
    
    data_folder = root_folder / subfolder
    
    train_ds = pts.dataset.BoneDetectionDataset(data_folder, 'train')
    train_dataloader = DataLoader(train_ds, batch_size=5, shuffle=True, num_workers=8, collate_fn=collate_fn)
    val_ds = pts.dataset.BoneDetectionDataset(data_folder, 'val')
    val_dataloader = DataLoader(val_ds, batch_size=5, shuffle=False, num_workers=8, collate_fn=collate_fn)
    
    tb_logger = TensorBoardLogger(save_dir=log_folder, name='{}_det'.format(subfolder))
    checkpoint_callback = ModelCheckpoint(dirpath=ckpt_folder / '{}_det'.format(subfolder), save_top_k=1, monitor="train_loss")
    trainer = L.Trainer(max_epochs=500, callbacks=[checkpoint_callback], logger=[tb_logger])
    model = pts.ObjectDetectionModel()
    trainer.fit(model, train_dataloader, val_dataloader)
    
def apply_det(root_folder, subfolder):
    ckpt_folder = Path('../ckpt')

    data_folder = root_folder / subfolder
    
    param_path = sorted((ckpt_folder / '{}_det'.format(subfolder)).glob('*.ckpt'))[-1]
    print(param_path)
    
    model = pts.ObjectDetectionModel.load_from_checkpoint(param_path)
    
    test_ds = pts.dataset.BoneDetectionDataset(data_folder, 'val')
    test_dataloader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=1, collate_fn=collate_fn)
    
    conf_mat = np.zeros((2,2))
    
    model.eval()
    for i, batch in enumerate(test_dataloader):        
        inp = list(image.cuda() for image in batch[0])
        res = model(inp)
        
        show_inp = batch[0][0].numpy()
        tg = {k : v.numpy() for k, v in batch[1][0].items()}
        pred = {k : v.detach().cpu().numpy() for k, v in res[0].items()}
        
        tg_cl = 0
        if np.count_nonzero(tg['labels'] == 2) > 0:
            tg_cl = 1
        pred_cl = 0
        if np.count_nonzero(np.logical_and(pred['labels'] == 2, pred['scores'] > 0.5)) > 0:
            pred_cl = 1
        
        print('{}: True {} | Pred {}'.format(i, tg_cl, pred_cl))
        if i == 6:
            print(pred)
        conf_mat[tg_cl][pred_cl] += 1
        
        comp_bbox(i, show_inp, tg, pred)
        
    print('Confusion Matrix')
    print(conf_mat)
    
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
    root_folder = Path('/export/scratch2/vladysla/Data/Real/POD/datasets_var')
    #subfolder2 = '90kV_45W_100ms_10avg'
    #subfolder = '40kV_40W_100ms_10avg'
    #subfolder = '90kV_3W_100ms_10avg'
    #subfolder = '90kV_3W_100ms_1avg'
    dual_folders = [
        ['40kV_40W_100ms_10avg', '90kV_45W_100ms_10avg'],
        ['40kV_40W_100ms_1avg', '90kV_45W_100ms_1avg'],
        ['40kV_40W_50ms_1avg', '90kV_45W_50ms_1avg'],
        ['40kV_40W_20ms_1avg', '90kV_45W_20ms_1avg'],
        ['gen_40kV_40W_100ms_1avg', 'gen_90kV_45W_100ms_1avg'],
        ['gen_40kV_40W_50ms_1avg', 'gen_90kV_45W_50ms_1avg'],
        ['gen_40kV_40W_20ms_1avg', 'gen_90kV_45W_20ms_1avg']
    ]
    '''
    dual_folders = [
        ['40kV_40W_100ms_10avg', '90kV_45W_100ms_10avg'],
        ['gen_40kV_40W_100ms_1avg', 'gen_90kV_45W_100ms_1avg'],
        ['gen_40kV_40W_50ms_1avg', 'gen_90kV_45W_50ms_1avg'],
        ['gen_40kV_40W_20ms_1avg', 'gen_90kV_45W_20ms_1avg'],
        ['gen_40kV_40W_10ms_1avg', 'gen_90kV_45W_10ms_1avg'],
        ['gen_40kV_40W_5ms_1avg', 'gen_90kV_45W_5ms_1avg'],
        ['40kV_3W_50ms_1avg', '90kV_3W_50ms_1avg']
    ]
    '''
    '''
    dual_folders = [
        ['gen_b4_40kV_40W_20ms_1avg', 'gen_b4_90kV_45W_20ms_1avg'],
        ['gen_b3_40kV_40W_20ms_1avg', 'gen_b3_90kV_45W_20ms_1avg'],
        ['gen_b2_40kV_40W_20ms_1avg', 'gen_b2_90kV_45W_20ms_1avg']
    ]
    '''
    '''
    dual_folders = [
        ['40kV_40W_100ms_10avg', '90kV_45W_100ms_10avg'],
        ['90kV_3W_100ms_10avg', '90kV_3W_100ms_10avg'],
        ['gen_90kV_3W_100ms_10avg', 'gen_90kV_3W_100ms_10avg'],
        ['90kV_3W_100ms_1avg', '90kV_3W_100ms_1avg'],
        ['gen_90kV_3W_100ms_1avg', 'gen_90kV_3W_100ms_1avg'],
        ['gen_40kV_40W_100ms_1avg', 'gen_90kV_45W_100ms_1avg'],
        ['gen_40kV_40W_50ms_1avg', 'gen_90kV_45W_50ms_1avg'],
        ['gen_40kV_40W_20ms_1avg', 'gen_90kV_45W_20ms_1avg'],
        ['gen_40kV_40W_10ms_1avg', 'gen_90kV_45W_10ms_1avg'],
        ['gen_40kV_40W_5ms_1avg', 'gen_90kV_45W_5ms_1avg'],
        ['gen_40kV_40W_3ms_1avg', 'gen_90kV_45W_3ms_1avg']
    ]
    '''
    
    # Albumentations OpenCV fix
    cv2.setNumThreads(0)
    cv2.ocl.setUseOpenCL(False)
    
    #check_train(root_folder, subfolder)
    #check_test(root_folder, subfolder)
    
    #check_dataset(root_folder / subfolder)
    #segm_train(root_folder, subfolder)
    #apply_segm(root_folder, dual_folders[1][0])
    #test_segm(root_folder, dual_folders[3][0])
    
    #dsegm_train(root_folder, dual_folders[6])
    #apply_dsegm(root_folder, dual_folders[3])
    test_dsegm(root_folder, dual_folders[0])
    
    #compose_pods()
    
    '''
    for pair in dual_folders:
        #segm_train(root_folder, pair[0])
        dsegm_train(root_folder, [pair[0], pair[1]])
    '''
    
    #check_detection(root_folder / subfolder)
    #det_train(root_folder, subfolder)
    #apply_det(root_folder, subfolder)
