import numpy as np
from pathlib import Path
from skimage.transform import downscale_local_mean
import imageio.v3 as imageio
import tifffile
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import shutil

import pod2settings as pts

def get_obj_classes():
    classes = {
        'NoBone' : 0,
        'RibBone' : 1
    }
    return classes

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

def msd_copy(chA_folder, chB_folder, out_folder):
    class_dict = get_obj_classes()
    out_folder.mkdir(exist_ok=True)
    for subset in ['train', 'val', 'test']:
        out_subfolder = out_folder / subset
        out_subfolder.mkdir(exist_ok = True)
        (out_subfolder / 'inp').mkdir(exist_ok = True)
        (out_subfolder / 'tg').mkdir(exist_ok = True)
        count = 0
        for cl in class_dict.keys():
            chA_inp = sorted((chA_folder / subset / cl).glob('*.tiff'))
            chA_tg = sorted((chA_folder / subset / '{}_segm'.format(cl)).glob('*.tiff'))
            chB_inp = sorted((chB_folder / subset / cl).glob('*.tiff'))
            chB_tg = sorted((chB_folder / subset / '{}_segm'.format(cl)).glob('*.tiff'))
            for i in range(len(chA_inp)):
                print(count)
                print(chA_inp[i], chA_tg[i])
                chA = imageio.imread(chA_inp[i])
                chB = imageio.imread(chB_inp[i])
                img = np.zeros((2, chA.shape[0], chA.shape[1]))
                img[0,:] = chA
                img[1,:] = chB
                tifffile.imwrite(out_subfolder / 'inp' / '{:03d}.tiff'.format(count), img.astype(np.float32))
                
                tgA = imageio.imread(chA_tg[i])
                #tg = tgA
                tg = np.zeros((tgA.shape[1], tgA.shape[2]), dtype=np.uint8)
                #print((tgA[0,:] == 1).shape)
                #print((tg).shape)
                tg[tgA[0,:] == 1] = 1
                tg[tgA[1,:] == 1] = 2
                tifffile.imwrite(out_subfolder / 'tg' / '{:03d}.tiff'.format(count), tg)
                count += 1
    
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

if __name__ == "__main__":
    kV = 40
    W = 40
    t_list = [100, 50, 20]
    
    datasets_folder = Path('/export/scratch2/vladysla/Data/Real/POD/datasets_var/')
    if kV == 40:
        inp_folder = datasets_folder / '40kV_40W_100ms_10avg/'
    elif kV == 90:
        inp_folder = datasets_folder / '90kV_45W_100ms_10avg/'
    
    for t in t_list:
        out_folder = datasets_folder / 'gen_{}kV_{}W_{}ms_1avg/'.format(kV, W, t)
        add_noise(inp_folder, out_folder, kV, W, t)

    #gen_sequence()
    #compare_gen_real()
    #ff_check()
    #process_real_data()
    #quotient_analysis()
    #process_ff()
    #gen_test()
