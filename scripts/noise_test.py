import numpy as np
from pathlib import Path
from skimage.transform import downscale_local_mean
import imageio.v3 as imageio
import tifffile
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import shutil

import pod2settings as pts

def check_img(W, t):
    #img = imageio.imread('/export/scratch2/vladysla/Data/Real/POD/pod2settings/test/log/90kV_45W_100ms_10avg.tif')
    img = imageio.imread('/export/scratch2/vladysla/Data/Real/POD/datasets/90kV_45W_100ms_10avg/train/RibBone/023.tiff')
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
    
def compare_gen_real():
    noiseless = imageio.imread('/export/scratch2/vladysla/Data/Real/POD/datasets/90kV_45W_100ms_10avg/train/RibBone/023.tiff')
    real = imageio.imread('/export/scratch2/vladysla/Data/Real/POD/datasets/90kV_3W_100ms_1avg/train/RibBone/023.tiff')
    gen = imageio.imread('tmp_imgs/noise/3W_100ms.tiff')
    
    fig = plt.figure(figsize = (16,5))
    gs = fig.add_gridspec(1, 3)
    
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(noiseless, cmap='gray', vmin=-0.1, vmax = 0.7)
    ax1.set_axis_off()
    fig.colorbar(im1, ax=ax1, fraction=0.038, pad=0.04)
    ax1.set_title('Noiseless', y = -0.1, fontsize=14, weight='bold')
    
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(real, cmap='gray', vmin=-0.1, vmax = 0.7)
    ax2.set_axis_off()
    fig.colorbar(im2, ax=ax2, fraction=0.038, pad=0.04)
    ax2.set_title('Measured noisy', y = -0.1, fontsize=14, weight='bold')
    
    ax3 = fig.add_subplot(gs[0, 2])
    im3 = ax3.imshow(gen, cmap='gray', vmin=-0.1, vmax = 0.7)
    ax3.set_axis_off()
    fig.colorbar(im3, ax=ax3, fraction=0.038, pad=0.04)
    ax3.set_title('Generated noisy', y = -0.1, fontsize=14, weight='bold')
    
    plt.tight_layout()
    plt.savefig('tmp_imgs/calibration/comp_gen.pdf', format='pdf')
    plt.show()
    
def add_noise(inp_folder, out_folder, W, t, binning):
    exp_settings = {
        'power' : W,
        'exposure_time' : t
    }
    calibration_data = {
        # 40 kV
        'count_per_pt' : 0.575,
        # 90 kV
        #'count_per_pt' : 3.86,
        'sensitivity' : '/export/scratch2/vladysla/Data/Real/POD/noise_calibration_20ms/coef.tiff',
        'gaussian_noise' : '/export/scratch2/vladysla/Data/Real/POD/noise_calibration_20ms/intercept.tiff',
        'blur_sigma' : 0.8,
        'flatfield' : '/export/scratch2/vladysla/Data/Real/POD/pod2settings/ff/90kV_45W_100ms_10avg.tif'
    }
    noise_gen = pts.generator.NoiseGenerator(calibration_data, exp_settings)
    #noise_gen.show_flatfield()

    subsets = ['train', 'val', 'test']
    class_folders = ['NoBone', 'RibBone']
    out_folder.mkdir(exist_ok = True)
    for subset in subsets:
        out_subfolder = out_folder / subset
        out_subfolder.mkdir(exist_ok = True)
        for cl in class_folders:
            out_cl_folder = out_subfolder / cl
            out_cl_folder.mkdir(exist_ok=True)
            fnames = (inp_folder / subset / cl).glob('*.tiff')
            for fname in fnames:
                img = imageio.imread(fname)
                noisy_img = noise_gen.add_noise(img)
                if binning != 1:
                    noisy_img = downscale_local_mean(noisy_img, (binning, binning))
                tifffile.imwrite(out_cl_folder / fname.name, noisy_img.astype(np.float32))
            
            print('before')
            if binning == 1:
                print('copy segm')
                shutil.copytree(inp_folder / subset / '{}_segm'.format(cl), out_subfolder / '{}_segm'.format(cl))
            else:
                (out_subfolder / '{}_segm'.format(cl)).mkdir(exist_ok=True)
                fnames = (inp_folder / subset / '{}_segm'.format(cl)).glob('*.tiff')
                for fname in fnames:
                    img = imageio.imread(fname)
                    img = downscale_local_mean(img, (binning, binning))
                    tifffile.imwrite(out_subfolder / '{}_segm'.format(cl) / fname.name, img.astype(np.uint8))
                    
            shutil.copy(inp_folder / subset / '{}.csv'.format(cl), out_subfolder / '{}.csv'.format(cl))
    
def gen_sequence():
    W_list = [3, 3, 3, 3, 15, 45]
    t_list = [15, 50, 100, 1000, 20, 1000]
    for i in range(len(W_list)):
        check_img(W_list[i], t_list[i])
    
def ff_check():
    exp_settings = {
        'power' : 3,
        'exposure_time' : 100
    }
    calibration_data = {
        'count_per_pt' : 3.86,
        'sensitivity' : '/export/scratch2/vladysla/Data/Real/POD/noise_calibration_20ms/coef.tiff',
        'gaussian_noise' : '/export/scratch2/vladysla/Data/Real/POD/noise_calibration_20ms/intercept.tiff',
        'blur_sigma' : 0.8,
        'flatfield' : '/export/scratch2/vladysla/Data/Real/POD/pod2settings/ff/90kV_45W_100ms_10avg.tif'
    }
    noise_gen = pts.generator.NoiseGenerator(calibration_data, exp_settings)
    noise_gen.flatfield_check()
    
def process_real_data():
    data_folder = Path('/export/scratch2/vladysla/Data/Real/POD/chicken/b40_2/')
    ff_folder = Path('/export/scratch2/vladysla/Data/Real/POD/chicken/ff/')
    fnames = sorted(data_folder.glob('*.tif'))
    fnames.remove(data_folder / 'di_pre.tif')
    
    imgs = [imageio.imread(fn) for fn in fnames]
    di = imageio.imread(data_folder / 'di_pre.tif')
    names = [fn.stem for fn in fnames]
    
    (data_folder / 'log').mkdir(exist_ok = True)
    for i in range(len(names)):
        im = imgs[i]
        ff = imageio.imread(ff_folder / '{}.tif'.format(names[i]))
        log = -np.log((im-di) / (ff-di))
        tifffile.imwrite(data_folder / 'log' / '{}.tif'.format(names[i]), log.astype(np.float32))
        
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
    W = 40
    t = 100
    b = 1
    
    inp_folder = Path('/export/scratch2/vladysla/Data/Real/POD/datasets_var/40kV_40W_100ms_10avg/')
    out_folder = Path('/export/scratch2/vladysla/Data/Real/POD/datasets_var/gen_40kV_{}W_{}ms_1avg/'.format(W, t))
    
    #inp_folder = Path('/export/scratch2/vladysla/Data/Real/POD/datasets_var/90kV_45W_100ms_10avg/')
    #out_folder = Path('/export/scratch2/vladysla/Data/Real/POD/datasets_var/gen_90kV_{}W_{}ms_1avg/'.format(W, t))
    
    #gen_sequence()
    #compare_gen_real()
    
    add_noise(inp_folder, out_folder, W, t, b)
    #ff_check()
    #process_real_data()
    #quotient_analysis()
    #process_ff()
    #gen_test()
