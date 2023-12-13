import numpy as np
from pathlib import Path
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
        self.test_step_outputs = {'tg' : [], 'pred' : [], 'FO_size' : [], 'FO_th' : [], 'Contrast' : [], 'Attenuation' : [], 'Recall' : []}

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
        
    def compute_recall(self, pred_mask, mask):
        tp = torch.count_nonzero(torch.logical_and(mask[1,:] == 1, pred_mask[1,:] == 1))
        fn = torch.count_nonzero(torch.logical_and(mask[1,:] == 1, pred_mask[1,:] == 0))
        
        if tp + fn == 0:
            return 0
        else:
            return tp / (tp + fn)
        
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
        recall_list = torch.zeros([pred_mask.size()[0]], dtype=torch.float32)
        for s in range(pred_mask.size()[0]):
            test_classes[s] = self.convert_pred_to_class(pred_mask[s,:], mask[s,:])
            recall_list[s] = self.compute_recall(pred_mask[s,:], mask[s,:])
            
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
        
        self.test_step_outputs['Recall'].extend([float(x.detach().cpu()) for x in recall_list])
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
        self.test_step_outputs['Recall'] = np.array(self.test_step_outputs['Recall'])
        #self.test_step_outputs['snr'] = np.array(self.test_step_outputs['snr'])
        #self.test_step_outputs['att_vals'] = np.array(self.test_step_outputs['att_vals'])
        
        
        class1_select = np.logical_and(self.test_step_outputs['tg'] == 1,
                                       self.test_step_outputs['Contrast'] < 0.2)
        
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
        print(self.test_step_outputs['pred'].shape)
        print(class1_select.shape)
        print(self.test_step_outputs['Recall'].shape)
        recall = self.test_step_outputs['Recall'][class1_select]
        
        res_arr = np.zeros((contrast.shape[0],), 
                           dtype = [('Contrast', '<f4'), ('Target', '<i4'), ('Prediction', '<i4'), ('Recall', '<f4')])
        res_arr['Contrast'] = contrast
        res_arr['Target'] = tg
        res_arr['Prediction'] = pred
        res_arr['Recall'] = recall
        print(res_arr.dtype.names)
        print(res_arr)
        res_folder = Path('./tmp_res')
        np.savetxt(res_folder / 'tmp.csv', res_arr, delimiter=',', 
                   fmt = ('%f', '%d', '%d', '%f'), header=','.join(res_arr.dtype.names))

        #fit_snr = pod.stat_analyze(snr, correct_det)
        #fit_contrast = pod.stat_analyze(contrast, correct_det)
        #fit_att = pod.stat_analyze(att_vals, correct_det)
        #fit_size = pod.stat_analyze(fo_size, correct_det)
        
        #print('Class 1', np.count_nonzero(class1_select))
        #pod.correlate_size_snr(att_vals, contrast)
        #pod.plot_histo(contrast)
        #pod.plot_points(att_vals, correct_det)
        #pod.plot_pod('test_s_pod_1', 'Quotient FO Signal', fit_contrast, contrast, tg, pred)
        #pod.plot_pod('test_att_pod_1', 'Attenuation at 40kV', fit_att, att_vals, tg, pred)
        #pod.plot_fract_pod(fit_contrast, contrast, tg, pred)
        #pod.comp_pod_arg(fit_snr, snr, fit_size, fo_size, tg, pred)
        
        self.test_step_outputs = []

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)
