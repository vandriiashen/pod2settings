import numpy as np
import pickle
import torch
import torch.nn as nn
from torchvision import models, ops
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchmetrics
import pytorch_lightning as L
import segmentation_models_pytorch as smp

#delete?
import tifffile
import matplotlib.pyplot as plt
from skimage import morphology

from pod2settings import pod

class ClassificationModel(L.LightningModule):
    def __init__(self, arch = 'ResNet18', num_channels = 1, num_classes = 4):
        super().__init__()
        self.save_hyperparameters()
        self.num_classes = num_classes
        self.num_channels = num_channels
        
        if arch == 'ResNet18':
            m = models.resnet18(weights=False)
            m.conv1 = nn.Conv2d(num_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            m.fc = nn.Linear(in_features=512, out_features=num_classes, bias=True)
        elif arch == 'ShuffleNet':
            m = models.shufflenet_v2_x0_5(weights=False)
            m.conv1[0] = nn.Conv2d(num_channels, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            m.fc = nn.Linear(in_features=1024, out_features=num_classes, bias=True)
            
        self.net = m
        
        self.val_pred = []
        self.val_tg = []
        self.test_inp = []
        self.test_pred = []
        self.test_tg = []
        self.test_id = []
        self.test_size = []
        
    def forward(self, x):
        return self.net(x)
    
    def training_step(self, batch, batch_idx):
        x = batch['input']
        tg = batch['label']
        y = self.net(x)
        loss = nn.functional.cross_entropy(y, tg)
        self.log("train_loss", loss, prog_bar=False)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x = batch['input']
        tg = batch['label']
        y = self.net(x)
        loss = nn.functional.cross_entropy(y, tg)
        
        _, pred = torch.max(y, 1)
        self.val_tg.extend(tg)
        self.val_pred.extend(pred)
        
        self.log("val_loss", loss, prog_bar=False)
        return loss
    
    def on_validation_epoch_end(self):
        self.val_pred = torch.Tensor(self.val_pred)
        self.val_tg = torch.Tensor(self.val_tg)
                
        f1 = torchmetrics.F1Score(task = 'multiclass', num_classes=self.num_classes)
        f1_score = f1(self.val_pred, self.val_tg)
        self.log("F1_val", f1_score, prog_bar=True)
        
        self.val_pred = []
        self.val_tg = []
    
    def test_step(self, batch, batch_idx):
        x = batch['input']
        tg = batch['label']
        y = self.net(x)
        # not necessary since softmax does not change max index
        #probabilities = nn.functional.softmax(y, dim=1)
        _, pred = torch.max(y, 1)
        
        #print(x)
        self.test_inp.extend(x)
        self.test_tg.extend(tg)
        self.test_pred.extend(pred)
        
    def on_test_epoch_end(self):
        self.test_pred = torch.Tensor(self.test_pred)
        self.test_tg = torch.Tensor(self.test_tg)
                
        conf_mat = torchmetrics.ConfusionMatrix(task = 'multiclass', num_classes=self.num_classes)
        mat = conf_mat(self.test_pred, self.test_tg)
        print(mat)
                
        self.test_inp = []
        self.test_pred = []
        self.test_tg = []
        self.test_id = []
        self.test_size = []
        
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)
    
class SegmentationModel(L.LightningModule):
    def __init__(self, arch, encoder, num_channels, num_classes, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.model = smp.create_model(arch, encoder_name=encoder, in_channels=num_channels, classes=num_classes, **kwargs)

        # preprocessing parameteres for image
        params = smp.encoders.get_preprocessing_params(encoder)
        # Images are already normalized
        self.register_buffer("std", torch.tensor(1.))
        self.register_buffer("mean", torch.tensor(0.))

        self.loss_fn = smp.losses.DiceLoss(smp.losses.MULTILABEL_MODE, from_logits=True)
        self.training_step_outputs = []
        self.validation_step_outputs = []
        #self.test_step_outputs = {'tg' : [], 'pred' : [], 'FO_size' : [], 'FO_th' : [], 'contrast' : [], 'snr' : [], 'att_vals' : []}
        self.test_step_outputs = {'tg' : [], 'pred' : [], 'FO_size' : [], 'FO_th' : [], 'Contrast' : [], 'Attenuation' : []}

    def forward(self, image):
        image = (image - self.mean) / self.std
        mask = self.model(image)
        return mask

    def shared_step(self, batch, stage):
        image = batch["input"]
        assert image.ndim == 4

        # Common architectures have 5 stages of downsampling by factor 2, so image dimensions should be divisible by 32
        h, w = image.shape[2:]
        assert h % 32 == 0 and w % 32 == 0
        
        mask = batch["mask"]
        assert mask.ndim == 4
        assert mask.max() <= 1.0 and mask.min() >= 0

        logits_mask = self.forward(image)
        loss = self.loss_fn(logits_mask, mask)
        prob_mask = logits_mask.sigmoid()
        pred_mask = (prob_mask > 0.5).float()
        tp, fp, fn, tn = smp.metrics.get_stats(pred_mask.long(), mask.long(), mode="binary")

        return {
            "loss": loss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }

    def shared_epoch_end(self, outputs, stage):
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])

        per_image_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")
        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")

        metrics = {
            f"{stage}_per_image_iou": per_image_iou,
            f"{stage}_dataset_iou": dataset_iou,
        }
        self.log_dict(metrics, prog_bar=True)

    def training_step(self, batch, batch_idx):
        step_out = self.shared_step(batch, "train")
        self.training_step_outputs.append(step_out)
        self.log('train_loss', step_out['loss'])
        return step_out
    
    def on_train_epoch_end(self):
        self.shared_epoch_end(self.training_step_outputs, "train")
        # remove saved step outputs to prevent memory leak
        self.training_step_outputs = []

    def validation_step(self, batch, batch_idx):
        step_out = self.shared_step(batch, "valid")
        self.validation_step_outputs.append(step_out)
        self.log('val_loss', step_out['loss'])
        return step_out
    
    def on_validation_epoch_end(self):
        self.shared_epoch_end(self.validation_step_outputs, "valid")
        self.validation_step_outputs = []
        
    def convert_pred_to_class(self, pred_mask, mask):
        is_fo_segmented = torch.max(pred_mask[1,:,:])
        if torch.max(mask[1,:,:]) > 0:
            tg_label = 1
        else:
            tg_label = 0
        
        tp = torch.count_nonzero(torch.logical_and(mask[1,:] == 1, pred_mask[1,:] == 1))
        fn = torch.count_nonzero(torch.logical_and(mask[1,:] == 1, pred_mask[1,:] == 0))
        
        if is_fo_segmented > 0:
            if tg_label == 0:
                return 1
            elif tp / (tp + fn) > 0.1:
                return 1
            else:
                return 0
        else:
            return 0
        
    def compute_attenuation_value(self, inp, segm):
        inp = inp.detach().cpu().numpy()
        segm = segm.detach().cpu().numpy()
        
        fo_segm = segm[1,:] > 0
        att_fo = inp[0,fo_segm]
        
        return att_fo.mean()
        
    def compute_contrast(self, q, segm, img_id, fo_size):
        fig, ax = plt.subplots(1, 2, figsize=(18,9))
        q = q.detach().cpu().numpy()
        
        segm = segm.detach().cpu().numpy()
        
        show_q = np.zeros((*q.shape, 3))
        vmin = 1.
        vmax = 2.5
        q = np.nan_to_num(q)
        q[q > 10.] = 10
        q[q < 0.] = 0
        for k in range(3):
            show_q[:,:,k] = (q - vmin) / (vmax - vmin)
        
        fo_segm = segm[1,:] > 0
        nb_segm = morphology.binary_dilation(fo_segm)
        for k in range(20):
            nb_segm = morphology.binary_dilation(nb_segm)
                        
        nb_segm = np.logical_and(nb_segm, segm[0,:] > 0)
        
        # Highlight FO
        show_q[morphology.binary_dilation(fo_segm, footprint=np.ones((5, 5))) ^ fo_segm,0] = 0
        show_q[morphology.binary_dilation(fo_segm, footprint=np.ones((5, 5))) ^ fo_segm,2] = 0
        # Highlight neighborhood of FO
        show_q[morphology.binary_dilation(nb_segm, footprint=np.ones((5, 5))) ^ nb_segm,0] = 0
        show_q[morphology.binary_dilation(nb_segm, footprint=np.ones((5, 5))) ^ nb_segm,1] = 0
        
        nb_segm = np.logical_and(nb_segm, np.logical_not(fo_segm))
        
        ax[0].imshow(show_q)
        
        ax[1].hist(q[nb_segm], bins=40, density=True, color='b', label = 'MO')
        ax[1].hist(q[fo_segm], bins=40, density=True, color='g', label = 'FO')
        ax[1].legend()
        mo_mean = q[nb_segm].mean()
        fo_mean = q[fo_segm].mean()
        mo_std = q[nb_segm].std()
        snr = (fo_mean - mo_mean) / mo_std
        contrast = fo_mean - mo_mean
        ax[1].set_title('{:.1f}mm: S = {:.3f} | N = {:.3f} | SNR = {:.1f}'.format(fo_size, fo_mean - mo_mean, mo_std, snr))
        
        plt.tight_layout()
        plt.savefig('tmp_imgs/snr/{}.png'.format(img_id))
        
        return contrast, snr

    def test_step(self, batch, batch_idx):
        image = batch["input"]
        assert image.ndim == 4

        # Common architectures have 5 stages of downsampling by factor 2, so image dimensions should be divisible by 32
        h, w = image.shape[2:]
        assert h % 32 == 0 and w % 32 == 0
        
        mask = batch["mask"]
        assert mask.ndim == 4
        assert mask.max() <= 1.0 and mask.min() >= 0

        logits_mask = self.forward(image)
        loss = self.loss_fn(logits_mask, mask)
        prob_mask = logits_mask.sigmoid()
        pred_mask = (prob_mask > 0.5).float()
        
        test_classes = torch.zeros([pred_mask.size()[0]], dtype=torch.uint8)
        for s in range(pred_mask.size()[0]):
            test_classes[s] = self.convert_pred_to_class(pred_mask[s,:], mask[s,:])
            
        contrast_vals = []
        snr_vals = []
        att_vals = []
        '''
        for s in range(batch["input"].size()[0]):
            if batch['label'][s] == 0:
                contrast = 0
                snr = 0
                att_val = 0
            else:
                contrast, snr = self.compute_contrast(batch["quotient"][s], batch["mask"][s], batch["img_id"][s], batch["FO_size"][s])
                att_val = self.compute_attenuation_value(batch["input"][s], batch["mask"][s])
            contrast_vals.append(contrast)
            snr_vals.append(snr)
            att_vals.append(att_val)
        '''
            
        self.test_step_outputs['tg'].extend([int(x.detach().cpu()) for x in batch['label']])
        self.test_step_outputs['pred'].extend([int(x.detach().cpu()) for x in test_classes])
        self.test_step_outputs['FO_size'].extend([float(x.detach().cpu()) for x in batch['FO_size']])
        self.test_step_outputs['FO_th'].extend([float(x.detach().cpu()) for x in batch['FO_th']])
        
        self.test_step_outputs['Contrast'].extend([float(x.detach().cpu()) for x in batch['Contrast']])
        self.test_step_outputs['Attenuation'].extend([float(x.detach().cpu()) for x in batch['Attenuation']])
        
        #self.test_step_outputs['contrast'].extend(contrast_vals)
        #self.test_step_outputs['snr'].extend(snr_vals)
        #self.test_step_outputs['att_vals'].extend(att_vals)
        
        #self.test_step_outputs.append(step_out)
        return 0. 
    
    def on_test_epoch_end(self):
        print(self.test_step_outputs)
        
        self.test_step_outputs['tg'] = np.array(self.test_step_outputs['tg'])
        self.test_step_outputs['pred'] = np.array(self.test_step_outputs['pred'])
        self.test_step_outputs['FO_size'] = np.array(self.test_step_outputs['FO_size'])
        self.test_step_outputs['FO_th'] = np.array(self.test_step_outputs['FO_th'])
        self.test_step_outputs['Contrast'] = np.array(self.test_step_outputs['Contrast'])
        self.test_step_outputs['Attenuation'] = np.array(self.test_step_outputs['Attenuation'])
        #self.test_step_outputs['snr'] = np.array(self.test_step_outputs['snr'])
        #self.test_step_outputs['att_vals'] = np.array(self.test_step_outputs['att_vals'])
        
        
        class1_select = np.logical_and(self.test_step_outputs['tg'] == 1,
                                       self.test_step_outputs['Contrast'] < 0.3)
        
        #class1_select = self.test_step_outputs['tg'] == 1
        
        correct_det = np.where(self.test_step_outputs['pred'] == self.test_step_outputs['tg'], 1, 0)[class1_select]
        fo_size = self.test_step_outputs['FO_size'][class1_select]
        fo_th = self.test_step_outputs['FO_th'][class1_select]
        contrast = self.test_step_outputs['Contrast'][class1_select]
        att_vals = self.test_step_outputs['Attenuation'][class1_select]
        #snr = self.test_step_outputs['snr'][class1_select]
        #att_vals = self.test_step_outputs['att_vals'][class1_select]
        
        print(contrast)
        
        print(fo_size.shape)
        print(correct_det.shape)
        
        print('#\tTg\tPred\tSize\tContrast')
        for i in range(self.test_step_outputs['tg'].shape[0]):
            if self.test_step_outputs['tg'][i] == self.test_step_outputs['pred'][i]:
                print(i, self.test_step_outputs['tg'][i], self.test_step_outputs['pred'][i], self.test_step_outputs['FO_th'][i], self.test_step_outputs['Contrast'][i])
            else:
                print(i, self.test_step_outputs['tg'][i], '|', self.test_step_outputs['pred'][i], self.test_step_outputs['FO_th'][i], self.test_step_outputs['Contrast'][i])
                
        conf_mat = torchmetrics.ConfusionMatrix(task = 'multiclass', num_classes=2)
        mat = conf_mat(torch.tensor(self.test_step_outputs['pred']), torch.tensor(self.test_step_outputs['tg']))
        print(mat)
        
        tg = self.test_step_outputs['tg'][class1_select]
        pred = self.test_step_outputs['pred'][class1_select]
        
        #fit_snr = pod.stat_analyze(snr, correct_det)
        fit_contrast = pod.stat_analyze(contrast, correct_det)
        fit_att = pod.stat_analyze(att_vals, correct_det)
        fit_size = pod.stat_analyze(fo_size, correct_det)
        
        print('Class 1', np.count_nonzero(class1_select))
        pod.correlate_size_snr(att_vals, contrast)
        pod.plot_histo(contrast)
        pod.plot_points(att_vals, correct_det)
        pod.plot_pod('test_s_pod_1', 'Quotient FO Signal', fit_contrast, contrast, tg, pred)
        pod.plot_pod('test_att_pod_1', 'Attenuation at 40kV', fit_att, att_vals, tg, pred)
        pod.plot_fract_pod(fit_contrast, contrast, tg, pred)
        #pod.comp_pod_arg(fit_snr, snr, fit_size, fo_size, tg, pred)
        
        self.test_step_outputs = []

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)
    
class ObjectDetectionModel(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()
        
        model = models.detection.fasterrcnn_resnet50_fpn(weights=None, min_size = 768, max_size = 960, image_mean=[0.], image_std=[1.], box_score_thresh=0.5)
        num_classes = 3
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        model.backbone.body.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        model.cuda()
        
        self.model = model

        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def forward(self, image):
        res = self.model(image)
        return res
    
    def training_step(self, batch, batch_idx):
        inputs, targets = batch

        losses = self.model(inputs, targets)
        total_loss = sum(loss for loss in losses.values())

        self.log("train_loss_classifier", losses["loss_classifier"])
        self.log("train_loss_box_reg", losses["loss_box_reg"])
        self.log("train_loss_objectness", losses["loss_objectness"])
        self.log("train_loss_rpn_box_reg", losses["loss_rpn_box_reg"])
        self.log("train_loss", total_loss)
        
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        
        pred = self.model(inputs)
        
        
        return 0.

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)
