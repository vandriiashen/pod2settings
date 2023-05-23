from pathlib import Path
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torchvision import models
import pytorch_lightning as L
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import pod2settings as pts

def check_dataset():
    folder = Path('/export/scratch2/vladysla/Data/Real/MEYN_chicken/meyn_standard')
    ds = pts.dataset.IndustrialDatasetGaussian(folder, 'val', 'A', add_noise = 0.1)
    print(ds[3]['input'].shape)
    plt.imshow(ds[20]['input'][0])
    plt.show()
    
def check_nn():
    #m = models.resnet18(weights=None)
    m = models.shufflenet_v2_x0_5(weights=None)
    print(m)
    #print(m.conv1[0])
    
def check_train():
    root_folder = Path('/export/scratch2/vladysla/Data/Real/MEYN_chicken/meyn_standard')
    log_folder = Path('../log')
    ckpt_folder = Path('../ckpt')
    
    train_ds = pts.dataset.IndustrialDatasetGaussian(root_folder, 'train', 'A', add_noise = 0.1)
    train_dataloader = DataLoader(train_ds, batch_size=10, shuffle=True, num_workers=8)
    val_ds = pts.dataset.IndustrialDatasetGaussian(root_folder, 'val', 'A', add_noise = 0.1)
    val_dataloader = DataLoader(val_ds, batch_size=10, shuffle=False, num_workers=8)
    
    tb_logger = TensorBoardLogger(save_dir=log_folder, name='test')
    checkpoint_callback = ModelCheckpoint(dirpath=ckpt_folder / 'test', save_top_k=1, monitor="val_loss")
    trainer = L.Trainer(max_epochs=1000, callbacks=[checkpoint_callback], logger=[tb_logger])
    model = pts.ClassificationModel(arch = 'ShuffleNet', num_channels = 1, num_classes = 4)
    trainer.fit(model, train_dataloader, val_dataloader)
    
def check_test():
    root_folder = Path('/export/scratch2/vladysla/Data/Real/MEYN_chicken/meyn_standard')
    ckpt_folder = Path('../ckpt')
    # 0.2 noise
    param_path = ckpt_folder / 'test' / 'epoch=153-step=3696.ckpt'
    # 0.1 noise
    #param_path = ckpt_folder / 'test' / 'epoch=473-step=11376.ckpt'
    # 0. noise
    #param_path = ckpt_folder / 'test' / 'epoch=844-step=20280.ckpt'
    model = pts.ClassificationModel.load_from_checkpoint(param_path)
    
    test_ds = pts.dataset.IndustrialDatasetGaussian(root_folder, 'test', 'A', add_noise = 0.2)
    test_dataloader = DataLoader(test_ds, batch_size=10, shuffle=False, num_workers=8)
    trainer = L.Trainer()
    trainer.test(model, test_dataloader)

if __name__ == "__main__":
    #pts.measure.cv_test()
    #check_dataset()
    #check_nn()
    #check_train()
    #check_test()
