import numpy as np
from pathlib import Path
import imageio.v3 as imageio
import tifffile
import shutil

import pod2settings as pts

def generate_noisy_datasets(root_folder, calibration, kV, W):
    inp_folder = root_folder / '{}kV_{}W_100ms_10avg/'.format(kV, W)
    t_list = [500, 200]
    for t in t_list:
        out_folder = root_folder / 'gen_{}kV_{}W_{}ms_1avg/'.format(kV, W, t)
        exp_settings = {
            'power' : W,
            'exposure_time' : t
        }
        data_gen = pts.generator.NoisyDataGenerator(inp_folder, out_folder, calibration, exp_settings)
        data_gen.process_data()
        
if __name__ == "__main__":
    root_folder = Path('/export/scratch2/vladysla/Data/Real/POD/datasets_var/')
    base_calibration = {
        # edit depending on the voltage
        'count_per_pt' : -1,
        'sensitivity' : '/export/scratch2/vladysla/Data/Real/POD/noise_calibration_20ms/coef.tiff',
        'gaussian_noise' : '/export/scratch2/vladysla/Data/Real/POD/noise_calibration_20ms/intercept.tiff',
        'blur_sigma' : 0.8,
        'flatfield' : '/export/scratch2/vladysla/Data/Real/POD/pod2settings/ff/90kV_45W_100ms_10avg.tif'
    }
    
    kV = 40
    W = 40
    calibration = base_calibration.copy()
    calibration['count_per_pt'] = 0.575
    generate_noisy_datasets(root_folder, calibration, kV, W)
    
    kV = 90
    W = 45
    calibration = base_calibration.copy()
    calibration['count_per_pt'] = 3.86
    generate_noisy_datasets(root_folder, calibration, kV, W)
