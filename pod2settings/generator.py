import numpy as np
from scipy.ndimage import gaussian_filter
from pathlib import Path
import imageio.v3 as imageio
import tifffile
import shutil

class NoiseGenerator():
    def __init__(self, calibration_data, exp_settings):
        self.exp_settings = exp_settings
        self.count_per_pt = calibration_data['count_per_pt']
        self.ff = imageio.imread(calibration_data['flatfield']).astype(np.float64)
        self.ff /= self.ff.mean()
        
        self.sensitivity = imageio.imread(calibration_data['sensitivity'])
        variance = imageio.imread(calibration_data['gaussian_noise'])
        variance[variance < 0.] = 0.
        self.gaussian_noise = np.sqrt(variance)
        
        self.blur_sigma = calibration_data['blur_sigma']
        # if blur is applied, noise properties should be corrected in a way that makes the resulting std match the measurements
        if self.blur_sigma != 0:
            self.sensitivity *= 4 * np.pi * self.blur_sigma**2
            self.gaussian_noise *= 2 * np.sqrt(np.pi) * self.blur_sigma

    def convert_to_counts(self, img):
        pt = self.exp_settings['power'] * self.exp_settings['exposure_time']
        avg_count = self.count_per_pt * pt
        adj_ff = self.ff * avg_count
        img = adj_ff * np.exp(-img)
        return img
        
    def log_cor(self, img):
        pt = self.exp_settings['power'] * self.exp_settings['exposure_time']
        avg_count = self.count_per_pt * pt
        adj_ff = self.ff * avg_count
        img = -np.log(img / adj_ff)
        return img
        
    def gen_mixed_noise(self, img):
        poisson_lambda =  img / self.sensitivity
        poisson = np.random.poisson(lam = poisson_lambda, size=img.shape) * self.sensitivity
        gaussian = np.random.normal(loc = 0., scale = self.gaussian_noise, size=img.shape)
        
        if self.blur_sigma == 0:
            res = poisson + gaussian
        else:
            res = gaussian_filter(poisson + gaussian, sigma=self.blur_sigma)
        
        return res
    
    def gen_flatfield(self):
        img = np.zeros_like(self.ff)
        img = self.convert_to_counts(img)
        img = self.gen_mixed_noise(img)
        return img
        
    def add_noise(self, img):
        img = self.convert_to_counts(img)
        img = self.gen_mixed_noise(img)
        img = self.log_cor(img)
        return img
    
    def flatfield_check(self):
        pt = 150
        avg_count = self.count_per_pt * pt
        adj_ff = self.ff * avg_count
        
        noisy_inst = np.zeros((10, *adj_ff.shape))
        for i in range(noisy_inst.shape[0]):
            noisy_inst[i,:,:] = self.gen_mixed_noise(adj_ff)
        
        print(noisy_inst[0].mean())
        print(noisy_inst[0].std())
        
        print('Mean - ', noisy_inst[:,200,200].mean())
        print('Measured std - ', noisy_inst[:,200,200].std())
        print('Expected std - ', np.sqrt(1.949*noisy_inst[:,200,200].mean() + 165.63))

def get_simple_obj_classes():
    classes = {
        '0' : 0,
        '1' : 1,
        '2' : 2
    }
    return classes

class SettingsDataGenerator():
    def __init__(self, data_folder, noisy_folder, calibration_data, exp_settings):
        self.data_folder = data_folder
        self.noisy_folder = noisy_folder
        self.calibration_data = calibration_data
        self.exp_settings = exp_settings
        
        noisy_folder.mkdir(exist_ok = True)
        self.noise_gen = NoiseGenerator(exp_settings, calibration_data)
        
    def process_data(self):
        class_dict = get_simple_obj_classes()
        for subset in ['train', 'val', 'test']:
            (self.noisy_folder / subset).mkdir(exist_ok = True)
            for obj_class in class_dict.keys():
                inp_folder = self.data_folder / subset / obj_class
                out_folder = self.noisy_folder / subset / obj_class
                out_folder.mkdir(exist_ok = True)
                
                class_data_fname = self.data_folder / subset / '{}.csv'.format(obj_class)
                shutil.copy(class_data_fname, self.noisy_folder / subset / '{}.csv'.format(obj_class))
                
                fnames = sorted(inp_folder.glob('*.tiff'))
                for fname in fnames:
                    img = imageio.imread(fname)
                    noisy_img = self.noise_gen.add_noise(img)
                    tifffile.imwrite(out_folder / fname.name, noisy_img)
                
                
                
                
        
        
        
