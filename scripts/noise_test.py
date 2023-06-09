import numpy as np
from pathlib import Path
import imageio.v3 as imageio
import tifffile
import matplotlib.pyplot as plt

import pod2settings as pts

def check_img(W, t):
    img = imageio.imread('/export/scratch2/vladysla/Data/Real/POD/pod2settings/test/log/90kV_45W_100ms_10avg.tif')
    exp_settings = {
        'power' : W,
        'exposure_time' : t
    }
    calibration_data = {
        'count_per_pt' : 5.0,
        'sensitivity' : '/export/scratch2/vladysla/Data/Real/POD/noise_calibration_20ms/coef.tiff',
        'gaussian_noise' : '/export/scratch2/vladysla/Data/Real/POD/noise_calibration_20ms/intercept.tiff',
        'blur_sigma' : 0.8,
        'flatfield' : '/export/scratch2/vladysla/Data/Real/POD/pod2settings/ff/90kV_45W_100ms_10avg.tif'
    }
    
    noise_gen = pts.generator.NoiseGenerator(calibration_data, exp_settings)
    noisy_img = noise_gen.add_noise(img)
    noisy_ff = noise_gen.gen_flatfield()
    
    tifffile.imwrite('tmp_imgs/noise/{}W_{}ms.tiff'.format(exp_settings['power'], exp_settings['exposure_time']), noisy_img.astype(np.float32))
    tifffile.imwrite('tmp_imgs/noise_ff/{}W_{}ms.tiff'.format(exp_settings['power'], exp_settings['exposure_time']), noisy_ff.astype(np.float32))
    
def gen_sequence():
    W_list = [3, 3, 3, 3, 15, 45]
    t_list = [15, 50, 100, 1000, 20, 1000]
    for i in range(len(W_list)):
        check_img(W_list[i], t_list[i])
    
def ff_check():
    exp_settings = {
        'power' : 3,
        'exposure_time' : 50
    }
    calibration_data = {
        'count_per_pt' : 5.0,
        'sensitivity' : '/export/scratch2/vladysla/Data/Real/POD/noise_calibration_20ms/coef.tiff',
        'gaussian_noise' : '/export/scratch2/vladysla/Data/Real/POD/noise_calibration_20ms/intercept.tiff',
        'blur_sigma' : 0.8,
        'flatfield' : '/export/scratch2/vladysla/Data/Real/POD/pod2settings/ff/90kV_45W_100ms_10avg.tif'
    }
    noise_gen = pts.generator.NoiseGenerator(calibration_data, exp_settings)
    noise_gen.flatfield_check()
    
def process_real_data():
    data_folder = Path('/export/scratch2/vladysla/Data/Real/POD/voltage_cu_50um/fanbone/')
    ff_folder = Path('/export/scratch2/vladysla/Data/Real/POD/voltage_cu_50um/ff/')
    fnames = sorted(data_folder.glob('*.tif'))
    fnames.remove(data_folder / 'di_pre.tif')
    fnames.remove(data_folder / 'di_post.tif')
    
    imgs = [imageio.imread(fn) for fn in fnames]
    di = imageio.imread(data_folder / 'di_pre.tif')
    names = [fn.stem for fn in fnames]
    
    (data_folder / 'log').mkdir(exist_ok = True)
    for i in range(len(names)):
        im = imgs[i]
        ff = imageio.imread(ff_folder / '{}.tif'.format(names[i]))
        log = -np.log((im-di) / (ff-di))
        tifffile.imwrite(data_folder / 'log' / '{}.tif'.format(names[i]), log.astype(np.float32))
        
def process_ff():
    ff_folder = Path('/export/scratch2/vladysla/Data/Real/POD/voltage_cu_50um/ff/')
    fnames = sorted(ff_folder.glob('*.tif'))
    fnames.remove(ff_folder / 'di_pre.tif')
    fnames.remove(ff_folder / 'di_post.tif')
    
    di = imageio.imread(ff_folder / 'di_pre.tif').astype(np.int32)
    names = [fn.stem for fn in fnames]
    
    (ff_folder / 'cor').mkdir(exist_ok = True)
    for i in range(len(names)):
        ff = imageio.imread(ff_folder / '{}.tif'.format(names[i])).astype(np.int32)
        cor = ff - di
        tifffile.imwrite(ff_folder / 'cor' / '{}.tif'.format(names[i]), cor)

def gen_test():
    data_folder = Path('/export/scratch2/vladysla/Code/pod_settings/gen_data/simple_data_n0.00')
    out_folder = Path('/export/scratch2/vladysla/Code/pod_settings/gen_data/simple_data_Xms')
    exp_settings = {}
    calibration_data = {}
    
    gen = pts.generator.SettingsDataGenerator(data_folder, out_folder, calibration_data, exp_settings)
    gen.process_data()

if __name__ == "__main__":
    #gen_sequence()
    #ff_check()
    process_real_data()
    #process_ff()
    #gen_test()
