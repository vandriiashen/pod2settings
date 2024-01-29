import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit
from pathlib import Path
import imageio.v3 as imageio
import tifffile
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.linear_model import LinearRegression

def show_mean_std(mean, std):
    fig, ax = plt.subplots(1, 2, figsize = (17, 9))
    ax[0].imshow(mean)
    ax[1].imshow(std)
    plt.show()
    
def read_noise_props(data_folder):
    fnames = sorted((data_folder / 'mean').glob('*.tiff'))
    mean_arr = []
    std_arr = []
    for subfolder in ['1W', '3W', '5W', '10W', '15W', '20W', '30W', '45W']:
        name = '{}.tiff'.format(subfolder)
        mean_arr.append(imageio.imread(data_folder / 'mean' / name))
        std_arr.append(imageio.imread(data_folder / 'std' / name))
    mean_arr = np.array(mean_arr)
    print(mean_arr[0].shape)
    print(mean_arr.shape)
    std_arr = np.array(std_arr)
    return mean_arr, std_arr

def fit_data(mean, std):
    x = mean.reshape(-1, 1)
    y = np.power(std, 2)
    
    reg = LinearRegression()
    reg.fit(x, y)
    return reg

def check_point(mean, std):
    mean_p = mean[:,500,500]
    std_p = std[:,500,500]

    reg = fit_data(mean_p[:], std_p[:])
    print(reg.coef_)
    print(reg.intercept_)
    
    x_val = np.linspace(0., mean_p.max(), 1000).reshape(-1, 1)
    
    fig, ax = plt.subplots(1, 1, figsize = (12, 9))
    ax.scatter(mean_p, np.power(std_p, 2), c = 'r', label = 'Measurements')
    ax.plot(x_val, reg.predict(x_val), c = 'b', label = 'Linear regression')
    ax.set_xlim(0.)
    ax.set_ylim(0.)
    ax.set_xlabel('Mean value', fontsize=20)
    ax.set_ylabel('Std^2', fontsize=20)
    ax.grid(True)
    plt.show()
    
def fit_detector_plane(data_folder, mean, std):
    coef_arr = np.zeros((mean.shape[1], mean.shape[2]))
    intercept_arr = np.zeros((mean.shape[1], mean.shape[2]))
    for y in tqdm(range(mean.shape[1])):
        for x in range(mean.shape[2]):
            mean_p = mean[:,y,x]
            std_p = std[:,y,x]
            reg = fit_data(mean_p, std_p)
            coef_arr[y,x] = reg.coef_
            intercept_arr[y,x] = reg.intercept_
            
    tifffile.imwrite(data_folder / 'coef.tiff', coef_arr.astype(np.float32))
    tifffile.imwrite(data_folder / 'intercept.tiff', intercept_arr.astype(np.float32))
    
def visualize_detector_plane(data_folder):
    coef_arr = tifffile.imread(data_folder / 'coef.tiff')
    intercept_arr = tifffile.imread(data_folder / 'intercept.tiff')
    df_arr = tifffile.imread(data_folder / 'di_pre.tif')
    
    fig = plt.figure(figsize = (16,4.3))
    gs = fig.add_gridspec(1, 3)
    
    plt.rcParams['text.usetex'] = True
    
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(coef_arr, cmap='gray')
    ax1.set_axis_off()
    fig.colorbar(im1, ax=ax1, fraction=0.038, pad=0.04)
    ax1.set_title(r'\textbf{(a) Gain} $g$', y = -0.1, fontsize=14, weight='bold')
    
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(intercept_arr, cmap='gray', vmin=0, vmax=500)
    ax2.set_axis_off()
    fig.colorbar(im2, ax=ax2, fraction=0.038, pad=0.04)
    ax2.set_title(r'\textbf{(b) Gaussian Variance} $\sigma_e^2$', y = -0.1, fontsize=14, weight='bold')
    
    ax3 = fig.add_subplot(gs[0, 2])
    im3 = ax3.imshow(df_arr, cmap='gray', vmin=1100, vmax=1400)
    ax3.set_axis_off()
    fig.colorbar(im3, ax=ax3, fraction=0.038, pad=0.04)
    ax3.set_title(r'\textbf{(c) Mean Darkfield} $d_e$', y = -0.1, fontsize=14, weight='bold')
    
    plt.tight_layout()
    plt.savefig('tmp_imgs/calibration/calibration.pdf', format='pdf')
    plt.show()
    
def fit_mean_vals(mean):
    power_vals = [1, 3, 5, 10, 15, 20, 30, 45]
    exp_t = [20, 20, 20, 20, 20, 20, 20, 20]
    pt = np.zeros((mean.shape[0],))
    intensity = np.zeros((mean.shape[0],))
    
    for i in range(mean.shape[0]):
        print(mean[i,500,500])
        pt[i] = power_vals[i] * exp_t[i]
        intensity[i] = mean[i,:,:].mean()
        print(intensity[i])
        
    x = pt.reshape(-1, 1)
    y = intensity
    reg = LinearRegression()
    reg.fit(x, y)
    print(reg.coef_)
    print(reg.intercept_)
    
    x_val = np.linspace(0., 5000, 1000).reshape(-1, 1)
    
    fig, ax = plt.subplots(1, 1, figsize = (12, 9))
    ax.scatter(x, y, c = 'r', label = 'Measurements')
    ax.plot(x_val, reg.predict(x_val), c = 'b', label = 'Linear regression')
    ax.set_xlim(0.)
    ax.set_ylim(0.)
    ax.set_xlabel('Power * Exposure time', fontsize=20)
    ax.set_ylabel('Intensity', fontsize=20)
    ax.grid(True)
    plt.show()
    
        
def average_proj(data_folder, subfolder):
    di = imageio.imread(data_folder / 'di_pre.tif')
    fnames = sorted((data_folder / subfolder).glob('*.tif'))
    proj_arr = np.zeros((len(fnames), *di.shape))
    for i in tqdm(range(len(fnames))):
        proj_arr[i,:,:] = imageio.imread(fnames[i])
    
    print(proj_arr.shape)
    proj_arr -= di
    
    print('before avg')
    mean = proj_arr.mean(axis = 0)
    std = proj_arr.std(axis = 0)
    print('after avg')
    
    return mean, std

def compute_correlation(data_folder):
    power_level = 5
    img = imageio.imread(data_folder / '{}W'.format(power_level) / 'scan_000010.tif')
    avg = imageio.imread(data_folder / 'mean' / '{}W.tiff'.format(power_level))
    df = imageio.imread(data_folder / 'di_post.tif')
    img = img - df - avg
        
    img = img.astype(np.float32)
    
    mask_dim = (9, 9)
    c_px = (4,4)
    mask = np.zeros((mask_dim[0],mask_dim[1], (img.shape[0]-mask_dim[0])*(img.shape[1]-mask_dim[1]) ))
    
    for j in range(mask.shape[0]):
        for i in range(mask.shape[1]):
            start_y = j
            end_y = img.shape[0] - mask_dim[0] + j
            start_x = i
            end_x = img.shape[1] - mask_dim[1] + i
            mask[j,i,:] = img[start_y:end_y,start_x:end_x].ravel()
        
    cov = np.zeros((mask_dim[0],mask_dim[1]))
    std = np.zeros((mask_dim[0],mask_dim[1]))
    for j in range(mask.shape[0]):
        for i in range(mask.shape[1]):
            std[j,i] = np.sqrt(np.power((mask[j,i,:] - mask[j,i,:].mean()), 2).mean())
            cov[j,i] = ( (mask[j,i,:] - mask[j,i,:].mean()) * (mask[c_px[0],c_px[1],:] - mask[c_px[0],c_px[1],:].mean()) ).mean()
            
    print(std)
    
    cov /= std
    cov /= std[c_px[0],c_px[1]]
    
    print(cov)
    
    return cov

def gaussian_fit(x, *p):
    sigma, offset = p
    return np.exp(-x**2 / (2 * sigma**2)) + offset

def exp_fit(x, *p):
    sigma, offset = p
    return np.exp(-x / sigma) + offset

def covariance_dist(cov):
    h, w = cov.shape
    X, Y = np.ogrid[:w, :h]
    c = (h//2, w//2)
    dist = (Y - c[0])**2 + (X - c[1])**2
    print(dist)
    
    popt, pcov = curve_fit(exp_fit, dist.ravel(), cov.ravel(), p0=[1., 0.])
    print(popt)
    fit_x = np.linspace(dist.min(), dist.max(), 100)
    fit_y = exp_fit(fit_x, *popt)
    
    plt.scatter(dist.ravel(), cov.ravel(), label = 'Correlation measurements')
    plt.plot(fit_x, fit_y, c='r', label='Fit, sigma = {:.2f}'.format(popt[0]))
    plt.legend()
    plt.xlabel('Distance')
    plt.ylabel('Correlation')
    plt.show()

def process_data(data_folder):
    (data_folder / 'mean').mkdir(exist_ok=True)
    (data_folder / 'std').mkdir(exist_ok=True)
    
    for subfolder in ['1W', '3W', '5W', '10W', '15W', '20W', '30W', '45W']:
        mean, std = average_proj(data_folder, subfolder)
        tifffile.imwrite(data_folder / 'mean' / '{}.tiff'.format(subfolder), mean.astype(np.float32))
        tifffile.imwrite(data_folder / 'std' / '{}.tiff'.format(subfolder), std.astype(np.float32))

if __name__ == '__main__':
    data_folder = Path('/path/to/data')
    process_data(data_folder)
    mean, std = read_noise_props(data_folder)
    
    fit_detector_plane(data_folder, mean, std)
    # This function produces Fig.3
    visualize_detector_plane(data_folder)
    
    compute_correlation(data_folder / 'mean' / '45W.tiff')
    cov = compute_correlation(data_folder)
    covariance_dist(cov)
