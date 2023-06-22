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
        self.test_step_outputs = []

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

    def test_step(self, batch, batch_idx):
        step_out = self.shared_step(batch, "test")
        self.test_step_outputs.append(step_out)
        return step_out 
    
    def on_test_epoch_end(self):
        # Test epoch end uses F1 as a metric
        outputs = self.test_step_outputs
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])
        f1 = smp.metrics.f1_score(tp, fp, fn, tn, reduction="macro-imagewise")
        metrics = {
            "test_f1_score": f1,
        }
        self.log_dict(metrics)
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
