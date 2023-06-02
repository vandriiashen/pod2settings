import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import tifffile
import imageio.v3 as imageio
import shutil
import pickle
import statsmodels.api as sm

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
        
def check_train(noise_level = 0.):
    log_folder = Path('../log')
    ckpt_folder = Path('../ckpt')
    
    folder_name = 'simple_data_n{:0.2f}'.format(noise_level)
    data_folder = Path('../gen_data/') / folder_name
    
    train_ds = pts.dataset.SimpleDataset(data_folder, 'train')
    train_dataloader = DataLoader(train_ds, batch_size=10, shuffle=True, num_workers=8)
    val_ds = pts.dataset.SimpleDataset(data_folder, 'val')
    val_dataloader = DataLoader(val_ds, batch_size=10, shuffle=False, num_workers=8)
    
    tb_logger = TensorBoardLogger(save_dir=log_folder, name=folder_name)
    checkpoint_callback = ModelCheckpoint(dirpath=ckpt_folder / folder_name, save_top_k=1, monitor="val_loss")
    trainer = L.Trainer(max_epochs=200, callbacks=[checkpoint_callback], logger=[tb_logger])
    model = pts.ClassificationModel(arch = 'ShuffleNet', num_channels = 1, num_classes = 3)
    trainer.fit(model, train_dataloader, val_dataloader)
    
def gen_data(noise_level = 0.):
    for subset in ['train', 'val', 'test']:
        generator = pts.dataset.SimpleGenerator(subset, noise_level)
        generator.generate_all()
        
def make_examples(noise_level):
    root_folder = Path('../gen_data')
    dataset_folder = root_folder / 'simple_data_n{:0.2f}'.format(noise_level) / 'train'
    samples = [30, 112, 240]
    
    for i, s in enumerate(samples):
        img = imageio.imread(dataset_folder / str(i) / '{:04d}.tiff'.format(s))
        img -= img.min()
        img /= img.max()
        img = (img * 255).astype(np.uint8)
        imageio.imwrite('tmp_imgs/simple_data_png/n{:.2f}_{:04d}.png'.format(noise_level, s), img)
    
def check_test(noise_level = 0.1):
    ckpt_folder = Path('../ckpt')

    folder_name = 'simple_data_n{:0.2f}'.format(noise_level)
    data_folder = Path('../gen_data/') / folder_name
    
    param_path = sorted((ckpt_folder / folder_name).glob('*.ckpt'))[-1]
    print(param_path)
    
    model = pts.ClassificationModel.load_from_checkpoint(param_path)
    
    test_ds = pts.dataset.SimpleDataset(data_folder, 'test')
    test_dataloader = DataLoader(test_ds, batch_size=10, shuffle=False, num_workers=8)
    trainer = L.Trainer()
    trainer.test(model, test_dataloader)
    
    shutil.move('tmp_imgs/simple_data_pod/pod.png', 'tmp_imgs/simple_data_pod/pod_{:0.2f}.png'.format(noise_level))
    shutil.move('tmp_imgs/simple_data_pod/detection.png', 'tmp_imgs/simple_data_pod/detection_{:0.2f}.png'.format(noise_level))
    shutil.move('tmp_res/res.bin', 'tmp_res/res_n{:0.2f}.bin'.format(noise_level))
    
def compare_properties(noise_levels):
    f1_vals = []
    s90_vals = []
    for noise_level in noise_levels:
        with open('tmp_res/res_n{:0.2f}.bin'.format(noise_level), 'rb') as f:
            res = pickle.load(f)
        f1_vals.append(float(res['F1_score']))
        s90 = pts.pod.compute_s90(res['POD_fit'])
        s90_vals.append(s90[0])
    print(f1_vals)
    print(s90_vals)
    
    fig, ax = plt.subplots(1, 2, figsize = (17, 9))
    
    ax[0].plot(noise_levels, f1_vals)
    ax[0].grid(True)
    ax[0].set_xlabel('Noise level', fontsize=16)
    ax[0].set_ylabel("Detection accuracy", fontsize=16)
    
    ax[1].plot(noise_levels, s90_vals)
    ax[1].grid(True)
    ax[1].set_xlabel('Noise level', fontsize=16)
    ax[1].set_ylabel("Smallest detectable FO", fontsize=16)
    
    plt.tight_layout()
    plt.savefig('tmp_imgs/simple_data_pod/noise_inf.png')
    
if __name__ == "__main__":
    #check_dataset()
    #make_examples(1.)
    '''
    for noise_level in [1.5, 2., 0.01]:
        gen_data(noise_level = noise_level)
        check_train(noise_level = noise_level)
    '''
    gen_data(noise_level = 0.)
    #check_test(noise_level = 0.01)
    #compare_properties([0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 1.])
    '''
    for noise_level in [0., 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 1., 1.5, 2.0]:
        gen_data(noise_level = noise_level)
        check_test(noise_level = noise_level)
    '''
