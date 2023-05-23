import numpy as np
import torch
import torch.nn as nn
from torchvision import models
import torchmetrics
import pytorch_lightning as L

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
            m = models.resnet18(weights=None)
            m.conv1 = nn.Conv2d(num_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            m.fc = nn.Linear(in_features=512, out_features=num_classes, bias=True)
        elif arch == 'ShuffleNet':
            m = models.shufflenet_v2_x0_5(weights=None)
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
        img_id = batch['img_id']
        fo_size = batch['fo_size']
        y = self.net(x)
        # not necessary since softmax does not change max index
        #probabilities = nn.functional.softmax(y, dim=1)
        _, pred = torch.max(y, 1)
        
        #print(x)
        self.test_inp.extend(x)
        self.test_tg.extend(tg)
        self.test_pred.extend(pred)
        self.test_id.extend(img_id)
        self.test_size.extend(fo_size)
        
    def on_test_epoch_end(self):
        self.test_pred = torch.Tensor(self.test_pred)
        self.test_tg = torch.Tensor(self.test_tg)
        self.test_id = torch.Tensor(self.test_id)
        self.test_size = torch.Tensor(self.test_size)
        
        true_pred = self.test_pred == self.test_tg
        #print(self.test_id[false_pred])
        #print(self.test_size)
        print(self.test_size.size()[0])
        '''
        for i in range(self.test_size.size()[0]):
            print('{}\t{}\t{}\t{}'.format(self.test_id[i], self.test_tg[i], self.test_pred[i], self.test_size[i]))
            if self.test_pred[i] == self.test_tg[i]:
                img = self.test_inp[i].detach().cpu().numpy()
                tifffile.imwrite('tmp_imgs/good_{}.tiff'.format(self.test_id[i]), img)
            else:
                img = self.test_inp[i].detach().cpu().numpy()
                tifffile.imwrite('tmp_imgs/bad_{}.tiff'.format(self.test_id[i]), img)
        '''
        
        true_fanbone = torch.logical_and(self.test_tg == 1, self.test_pred == 1)
        false_fanbone = torch.logical_and(self.test_tg == 1, self.test_pred != 1)
        print(self.test_size[true_fanbone])
        print(self.test_size[false_fanbone])
        print(self.test_size[true_fanbone].min())
        print(self.test_size[false_fanbone].max())
        
        x = self.test_size[self.test_tg == 1]
        tg = self.test_tg[self.test_tg == 1]
        pred = self.test_pred[self.test_tg == 1]
        y = true_pred[self.test_tg == 1]
        fit = pod.stat_analyze(x, y)
        pod.plot_points(x, y)
        pod.plot_pod(fit, x, tg, pred)
        
        f1 = torchmetrics.F1Score(task = 'multiclass', num_classes=self.num_classes)
        f1_score = f1(self.test_pred, self.test_tg)
        
        conf_mat = torchmetrics.ConfusionMatrix(task = 'multiclass', num_classes=self.num_classes)
        mat = conf_mat(self.test_pred, self.test_tg)
        print(mat)
        
        metrics = {
                'F1_test' : f1_score,
        }
        self.log_dict(metrics, prog_bar=True)
        
        self.test_inp = []
        self.test_pred = []
        self.test_tg = []
        self.test_id = []
        self.test_size = []
        
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
