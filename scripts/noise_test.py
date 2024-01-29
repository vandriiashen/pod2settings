import numpy as np
from pathlib import Path
from skimage.transform import downscale_local_mean
import imageio.v3 as imageio
import tifffile
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import shutil

import pod2settings as pts
    
def compare_gen_real_multiple(num = 23):
    dataset_folder = Path('/path/to/data')
    img_subpath = Path('test/RibBone/{:03d}.tiff'.format(num))
    
    fig = plt.figure(figsize = (16,6.6))

    gs = fig.add_gridspec(2, 4)
    
    noiseless = imageio.imread(dataset_folder / '40kV_40W_100ms_10avg' / img_subpath)
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(noiseless, cmap='gray', vmin=-0.1, vmax = 1.2)
    ax1.xaxis.set_tick_params(labelbottom=False)
    ax1.yaxis.set_tick_params(labelleft=False)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.patch.set_edgecolor('g')  
    ax1.patch.set_linewidth(10)
    
    t_vals = [100, 50, 20]
    for i in range(3):
        real = imageio.imread(dataset_folder / '40kV_40W_{}ms_1avg'.format(t_vals[i]) / img_subpath)
        gen = imageio.imread(dataset_folder / 'gen_40kV_40W_{}ms_1avg'.format(t_vals[i]) / img_subpath)
        ax_real = fig.add_subplot(gs[0, i+1])
        im_real = ax_real.imshow(real, cmap='gray', vmin=-0.1, vmax = 1.2)
        ax_real.xaxis.set_tick_params(labelbottom=False)
        ax_real.yaxis.set_tick_params(labelleft=False)
        ax_real.set_xticks([])
        ax_real.set_yticks([])
        ax_real.patch.set_edgecolor('g')  
        ax_real.patch.set_linewidth(6)
        
        ax_gen = fig.add_subplot(gs[1, i+1])
        im_gen = ax_gen.imshow(gen, cmap='gray', vmin=-0.1, vmax = 1.2)
        ax_gen.xaxis.set_tick_params(labelbottom=False)
        ax_gen.yaxis.set_tick_params(labelleft=False)
        ax_gen.set_xticks([])
        ax_gen.set_yticks([])
        ax_gen.patch.set_edgecolor('r')  
        ax_gen.patch.set_linewidth(6)
    
    plt.gcf().text(0.015, 0.73, 'Real', rotation=90, fontsize=16)
    plt.gcf().text(0.015, 0.2, 'Generated', rotation=90, fontsize=16)
    plt.gcf().text(0.09, 0.015, 't = 1s (High-Quality)', fontsize=16)
    plt.gcf().text(0.37, 0.015, 't = 100ms', fontsize=16)
    plt.gcf().text(0.61, 0.015, 't = 50ms', fontsize=16)
    plt.gcf().text(0.85, 0.015, 't = 20ms', fontsize=16)
    
    plt.tight_layout()
    plt.subplots_adjust(left = 0.05, bottom = 0.05)
    plt.savefig('tmp_imgs/calibration/comp_gen_multiple.pdf', format='pdf')
    plt.show()

def add_noise(inp_folder, out_folder, kV, W, t):
    exp_settings = {
        'power' : W,
        'exposure_time' : t
    }
    if kV == 40:
        calibration_data = {
            'count_per_pt' : 0.575,
            'sensitivity' : '/path/to/data',
            'gaussian_noise' : '/path/to/data',
            'blur_sigma' : 0.8,
            'flatfield' : '/path/to/data'
        }
    elif kV == 90:
        calibration_data = {
            'count_per_pt' : 3.86,
            'sensitivity' : '/path/to/data',
            'gaussian_noise' : '/path/to/data',
            'blur_sigma' : 0.8,
            'flatfield' : '/path/to/data'
        }
    
    data_gen = pts.generator.NoisyDataGenerator(inp_folder, out_folder, calibration_data, exp_settings)
    data_gen.process_data()
                
def generate_noisy_data():
    kV = 40
    W = 40
    t_list = [200, 75, 35]
    
    datasets_folder = Path('/path/to/data')
    if kV == 40:
        inp_folder = datasets_folder / '40kV_40W_100ms_10avg/'
    elif kV == 90:
        inp_folder = datasets_folder / '90kV_45W_100ms_10avg/'
    
    for t in t_list:
        out_folder = datasets_folder / 'gen_{}kV_{}W_{}ms_1avg/'.format(kV, W, t)
        add_noise(inp_folder, out_folder, kV, W, t)
    
if __name__ == "__main__":
    # This function generates noisy data for the desired values of exposure time
    generate_noisy_data()
    
    # This function produces Fig. 4
    compare_gen_real_multiple()
