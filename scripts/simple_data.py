import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import tifffile
import shutil

from torch.utils.data import DataLoader
from torchvision import models
import pytorch_lightning as L
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import pod2settings as pts

def check_dataset():
    ds = pts.dataset.SimpleDataset('train', add_noise=0.)
    for entry in ds:
        tifffile.imwrite('tmp_imgs/simple_data/simple_{}.tiff'.format(entry['img_id']), entry['input'].astype(np.float32))
        
def check_train():
    log_folder = Path('../log')
    ckpt_folder = Path('../ckpt')
    
    noise_level = 0.5
    folder_name = 'simple_data_n{:0.2f}'.format(noise_level)
    
    train_ds = pts.dataset.SimpleDataset('train', add_noise=noise_level)
    train_dataloader = DataLoader(train_ds, batch_size=10, shuffle=True, num_workers=8)
    val_ds = pts.dataset.SimpleDataset('val', add_noise=noise_level)
    val_dataloader = DataLoader(val_ds, batch_size=10, shuffle=False, num_workers=8)
    
    tb_logger = TensorBoardLogger(save_dir=log_folder, name=folder_name)
    checkpoint_callback = ModelCheckpoint(dirpath=ckpt_folder / folder_name, save_top_k=1, monitor="val_loss")
    trainer = L.Trainer(max_epochs=200, callbacks=[checkpoint_callback], logger=[tb_logger])
    model = pts.ClassificationModel(arch = 'ShuffleNet', num_channels = 1, num_classes = 3)
    trainer.fit(model, train_dataloader, val_dataloader)
    
def gen_data():
    add_noise = 0.
    
    for subset in ['train', 'val', 'test']:
        generator = pts.dataset.SimpleGenerator(subset, add_noise)
        generator.generate_all()
    
def check_test():
    ckpt_folder = Path('../ckpt')
    
    noise_level = 0.5
    folder_name = 'simple_data_n{:0.2f}'.format(noise_level)
    
    #0. noise
    #param_path = ckpt_folder / 'simple_data' / 'epoch=36-step=3330.ckpt'
    #0.1 noise
    #param_path = ckpt_folder / 'simple_data' / 'epoch=36-step=3330.ckpt'
    
    param_path = sorted((ckpt_folder / folder_name).glob('*.ckpt'))[-1]
    print(param_path)
    
    model = pts.ClassificationModel.load_from_checkpoint(param_path)
    
    test_ds = pts.dataset.SimpleDataset('test', add_noise=noise_level)
    test_dataloader = DataLoader(test_ds, batch_size=10, shuffle=False, num_workers=8)
    trainer = L.Trainer()
    trainer.test(model, test_dataloader)
    
    shutil.move('tmp_imgs/simple_data_pod/pod.png', 'tmp_imgs/simple_data_pod/pod_{:0.2f}.png'.format(noise_level))
    shutil.move('tmp_imgs/simple_data_pod/detection.png', 'tmp_imgs/simple_data_pod/detection_{:0.2f}.png'.format(noise_level))
    
if __name__ == "__main__":
    #check_dataset()
    gen_data()
    #check_train()
    #check_test()
