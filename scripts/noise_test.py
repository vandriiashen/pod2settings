import numpy as np
from pathlib import Path
from skimage.transform import downscale_local_mean
import imageio.v3 as imageio
import tifffile
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import shutil

import pod2settings as pts
    
def compare_gen_real(num = 23):
    dataset_folder = Path('/export/scratch2/vladysla/Data/Real/POD/datasets_var')
    img_subpath = Path('test/RibBone/{:03d}.tiff'.format(num))
    noiseless = imageio.imread(dataset_folder / '40kV_40W_100ms_10avg' / img_subpath)
    real = imageio.imread(dataset_folder / '40kV_40W_50ms_1avg' / img_subpath)
    gen = imageio.imread(dataset_folder / 'gen_40kV_40W_50ms_1avg' / img_subpath)
    
    fig = plt.figure(figsize = (16,5))
    gs = fig.add_gridspec(1, 3)
    
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(noiseless, cmap='gray', vmin=-0.1, vmax = 1.2)
    ax1.set_axis_off()
    fig.colorbar(im1, ax=ax1, fraction=0.038, pad=0.04)
    ax1.set_title('(a) High-quality (t=1s)', y = -0.1, fontsize=14, weight='bold')
    
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(real, cmap='gray', vmin=-0.1, vmax = 1.2)
    ax2.set_axis_off()
    fig.colorbar(im2, ax=ax2, fraction=0.038, pad=0.04)
    ax2.set_title('(b) Measured noisy (t=50ms)', y = -0.1, fontsize=14, weight='bold')
    
    ax3 = fig.add_subplot(gs[0, 2])
    im3 = ax3.imshow(gen, cmap='gray', vmin=-0.1, vmax = 1.2)
    ax3.set_axis_off()
    fig.colorbar(im3, ax=ax3, fraction=0.038, pad=0.04)
    ax3.set_title('(c) Generated noisy (t=50ms)', y = -0.1, fontsize=14, weight='bold')
    
    plt.tight_layout()
    plt.savefig('tmp_imgs/calibration/comp_gen.pdf', format='pdf')
    plt.show()
        
def add_noise(inp_folder, out_folder, kV, W, t):
    exp_settings = {
        'power' : W,
        'exposure_time' : t
    }
    if kV == 40:
        calibration_data = {
            'count_per_pt' : 0.575,
            'sensitivity' : '/export/scratch2/vladysla/Data/Real/POD/noise_calibration_20ms/coef.tiff',
            'gaussian_noise' : '/export/scratch2/vladysla/Data/Real/POD/noise_calibration_20ms/intercept.tiff',
            'blur_sigma' : 0.8,
            'flatfield' : '/export/scratch2/vladysla/Data/Real/POD/pod2settings/ff/90kV_45W_100ms_10avg.tif'
        }
    elif kV == 90:
        calibration_data = {
            'count_per_pt' : 3.86,
            'sensitivity' : '/export/scratch2/vladysla/Data/Real/POD/noise_calibration_20ms/coef.tiff',
            'gaussian_noise' : '/export/scratch2/vladysla/Data/Real/POD/noise_calibration_20ms/intercept.tiff',
            'blur_sigma' : 0.8,
            'flatfield' : '/export/scratch2/vladysla/Data/Real/POD/pod2settings/ff/90kV_45W_100ms_10avg.tif'
        }
    
    data_gen = pts.generator.NoisyDataGenerator(inp_folder, out_folder, calibration_data, exp_settings)
    data_gen.process_data()
            
def quotient_analysis(t_name = '100ms_5000avg'):
    data_folder = Path('/export/scratch2/vladysla/Data/Real/POD/chicken/b40_2/log')
    
    ch1 = imageio.imread(data_folder / '40kV_60W_{}.tif'.format(t_name))
    ch2 = imageio.imread(data_folder / '90kV_90W_{}.tif'.format(t_name))
    q = ch1 / ch2
    
    fig, ax = plt.subplots(2, 2, figsize = (16,12))
    ax[0,0].imshow(ch1)
    ax[0,0].set_title('40kV', fontsize=20)
    ax[0,1].imshow(ch2)
    ax[0,1].set_title('90kV', fontsize=20)
    
    b_x1 = 321
    b_x2 = 328
    bg = 20
    b_y = 44
    
    ax[1,0].imshow(q, vmin=1.2, vmax=2.0)
    rect = patches.Rectangle((b_x1-bg, b_y), b_x2-b_x1+2*bg, 1, linewidth=2, edgecolor='red', facecolor='none')
    ax[1,0].add_patch(rect)
    ax[1,0].set_title('Quotient', fontsize=20)
    
    
    bg_prof1 = q[b_y,b_x1-bg:b_x1]
    bg_prof2 = q[b_y,b_x2:b_x2+bg]
    bg_prof = np.concatenate((bg_prof1, bg_prof2))
    def_prof = q[b_y,b_x1:b_x2]
    
    ref_n = q[150:250,150:250].std()
    n = bg_prof.std()
    s = def_prof.mean() - bg_prof.mean()
    snr = s/n
    print('Ref_N = {:.3f}, N = {:.3f}, S = {:.3f}, SNR = {:.1f}'.format(ref_n, n, s, snr))
    
    prof = q[b_y, b_x1-bg:b_x2+bg]
    ax[1,1].plot(prof)
    ax[1,1].set_title('Values along the red line, SNR = {:.1f}'.format(snr), fontsize=20)
    
    plt.tight_layout()
    plt.savefig('tmp_imgs/quotient/comp_{}.png'.format(t_name))
    plt.show()
    
def generate_noisy_data():
    kV = 40
    W = 40
    t_list = [200, 75, 35]
    
    #datasets_folder = Path('/export/scratch2/vladysla/Data/Real/POD/datasets_var/')
    datasets_folder = Path('/export/scratch2/vladysla/Data/Real/POD/datasets/')
    if kV == 40:
        inp_folder = datasets_folder / '40kV_40W_100ms_10avg/'
    elif kV == 90:
        inp_folder = datasets_folder / '90kV_45W_100ms_10avg/'
    
    for t in t_list:
        out_folder = datasets_folder / 'gen_{}kV_{}W_{}ms_1avg/'.format(kV, W, t)
        add_noise(inp_folder, out_folder, kV, W, t)
    
if __name__ == "__main__":
    #generate_noisy_data()
    compare_gen_real()
    #quotient_analysis()
