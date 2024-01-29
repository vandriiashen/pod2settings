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
        self.test_step_outputs = {'tg' : [], 'pred' : [], 'FO_size' : [], 'Contrast' : [], 'Attenuation' : [], 'Recall' : []}

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
                        
        self.test_step_outputs['tg'].extend([int(x.detach().cpu()) for x in batch['label']])
        self.test_step_outputs['pred'].extend([int(x.detach().cpu()) for x in test_classes])
        self.test_step_outputs['FO_size'].extend([float(x.detach().cpu()) for x in batch['FO_size']])
        self.test_step_outputs['Contrast'].extend([float(x.detach().cpu()) for x in batch['Contrast']])
        self.test_step_outputs['Attenuation'].extend([float(x.detach().cpu()) for x in batch['Attenuation']])
        self.test_step_outputs['Recall'].extend([float(x.detach().cpu()) for x in recall_list])
        return 0. 
    
    def on_test_epoch_end(self):
        print(self.test_step_outputs)
        
        self.test_step_outputs['tg'] = np.array(self.test_step_outputs['tg'])
        self.test_step_outputs['pred'] = np.array(self.test_step_outputs['pred'])
        self.test_step_outputs['FO_size'] = np.array(self.test_step_outputs['FO_size'])
        self.test_step_outputs['Contrast'] = np.array(self.test_step_outputs['Contrast'])
        self.test_step_outputs['Attenuation'] = np.array(self.test_step_outputs['Attenuation'])    
        self.test_step_outputs['Recall'] = np.array(self.test_step_outputs['Recall'])
        
        # Image with Contrast > 0.2 are outliers
        class1_select = np.logical_and(self.test_step_outputs['tg'] == 1,
                                       self.test_step_outputs['Contrast'] < 0.2)
                
        correct_det = np.where(self.test_step_outputs['pred'] == self.test_step_outputs['tg'], 1, 0)[class1_select]
        fo_size = self.test_step_outputs['FO_size'][class1_select]
        contrast = self.test_step_outputs['Contrast'][class1_select]
        att_vals = self.test_step_outputs['Attenuation'][class1_select]
        recall = self.test_step_outputs['Recall'][class1_select]
                
        conf_mat = torchmetrics.ConfusionMatrix(task = 'multiclass', num_classes=2)
        mat = conf_mat(torch.tensor(self.test_step_outputs['pred']), torch.tensor(self.test_step_outputs['tg']))
        print('Confusion Matrix')
        print(mat)
        
        tg = self.test_step_outputs['tg'][class1_select]
        pred = self.test_step_outputs['pred'][class1_select]
        print(self.test_step_outputs['pred'].shape)
        print(class1_select.shape)
        
        res_arr = np.zeros((contrast.shape[0],), 
                           dtype = [('FO_size', '<f4'), ('Contrast', '<f4'), ('Attenuation', '<f4'), ('Target', '<i4'), ('Prediction', '<i4'), ('Recall', '<f4')])
        res_arr['FO_size'] = fo_size
        res_arr['Contrast'] = contrast
        res_arr['Attenuation'] = att_vals
        res_arr['Target'] = tg
        res_arr['Prediction'] = pred
        res_arr['Recall'] = recall
        res_folder = Path('./tmp_res')
        np.savetxt(res_folder / 'tmp.csv', res_arr, delimiter=',', 
                   fmt = ('%f', '%f', '%f', '%d', '%d', '%f'), header=','.join(res_arr.dtype.names))
        
        self.test_step_outputs = []

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)
