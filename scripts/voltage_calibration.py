import numpy as np
from pathlib import Path
import imageio.v3 as imageio
import tifffile
import matplotlib.pyplot as plt
import matplotlib.patches as patches

clay_select = [
    {
     'thickness' : 7., 
     'ul_corner'  : [480, 120],
     'width_height'  : [30, 30]
    },
    {
     'thickness' : 11., 
     'ul_corner'  : [480, 190],
     'width_height'  : [50, 50]
    },
    {
     'thickness' : 15., 
     'ul_corner'  : [470, 300],
     'width_height'  : [80, 80]
    },
    {
     'thickness' : 28., 
     'ul_corner'  : [440, 470],
     'width_height'  : [140, 130]
    }
]

# sample FO thicknes == 7.
pebble_select = [
    {
     'thickness' : 6., 
     'ul_corner'  : [300, 480],
     'width_height'  : [50, 50]
    },
    {
     'thickness' : 7., 
     'ul_corner'  : [580, 470],
     'width_height'  : [60, 40]
    },
    {
     'thickness' : 13., 
     'ul_corner'  : [440, 490],
     'width_height'  : [25, 25]
    }
]
    
def check_selections(img, selections):
    fig, ax = plt.subplots()
    ax.imshow(img)

    for select in selections:
        w = select['width_height'][0]
        h = select['width_height'][1]
        rect = patches.Rectangle(select['ul_corner'], w, h, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    plt.show()
    
def measure_attenuation(img, selections):
    res = []
    for select in selections:
        b_x = select['ul_corner'][0]
        b_y = select['ul_corner'][1]
        w = select['width_height'][0]
        h = select['width_height'][1]
        region = img[b_y:b_y+h, b_x:b_x+w]
        mean = region.mean()
        std = region.std()
        print('Thickness {:.0f} mm : {:.2f} +- {:.2f}'.format(select['thickness'], mean, std))
        res.append(mean)
    return res

def material_calibration(data_folder, select):
    fnames = sorted(data_folder.glob('*.tif'))
    print(fnames)
    voltages = [40, 50, 60, 70, 80, 90]
    
    #img = imageio.imread(fnames[0])
    #check_selections(img, select)
    #measure_attenuation(img, select)
    
    res_arr = np.zeros((len(voltages)+1, len(select) + 2), dtype = np.float32)
    res_arr[1:,0] = voltages
    
    for j in range(len(select)):
        res_arr[0,2+j] = select[j]['thickness']
    
    for i, fn in enumerate(fnames):
        img = imageio.imread(fn)
        att_vals = measure_attenuation(img, select)
        res_arr[1+i, 2:] = att_vals
    print(res_arr)
    
    np.savetxt('tmp_res/clay_calibr.csv', res_arr, delimiter=',')
    
    return res_arr

def filter_calibration(data_folder):
    #sizes = [100, 200, 500, 1000, 1500]
    sizes = [20, 100, 150, 300, 500]
    voltages = [40, 50, 60, 70, 80, 90]
    
    res_arr = np.zeros((len(voltages)+1, len(sizes) + 2), dtype = np.float32)
    res_arr[1:,0] = voltages
    res_arr[0,2:] = np.array(sizes).astype(np.float32) / 10**3 # in mm
    
    for i, size in enumerate(sizes):
        #fnames = sorted((data_folder / 'al_{}um'.format(size) / 'log').glob('*.tif'))
        fnames = sorted((data_folder / 'cu_{}um'.format(size) / 'log').glob('*.tif'))
        for j, fn in enumerate(fnames):
            img = imageio.imread(fn)
            res_arr[1+j,2+i] = img.mean()
    
    print(res_arr)
    #np.savetxt('tmp_res/al_calibr.csv', res_arr, delimiter=',')
    np.savetxt('tmp_res/cu_calibr.csv', res_arr, delimiter=',')
    
    return res_arr

def visualize_calibration(res_arr):
    x_range = res_arr[0,1:]
    voltages = res_arr[1:,0]
    
    fig, ax = plt.subplots(figsize=(12,9))
    
    for i in range(len(voltages)):
        y_vals = res_arr[1+i, 1:]
        ax.plot(x_range, y_vals, label = '{:d} kV'.format(int(voltages[i])))
    ax.grid(True)
    ax.legend(fontsize=20)
    ax.set_ylabel('Attenuation', fontsize=20)
    ax.set_xlabel('Thickness, mm', fontsize=20)
    ax.set_xlim(0.)
    ax.set_ylim(0.)
    plt.savefig('tmp_imgs/spectrum/pebble.png')
    plt.show()
    
if __name__ == "__main__":
    clay_folder = Path('/export/scratch2/vladysla/Data/Real/POD/voltage/clay/log')
    pebble_folder = Path('/export/scratch2/vladysla/Data/Real/POD/voltage/pebble/log')
    #res_arr = material_calibration(clay_folder, clay_select)
    res_arr = material_calibration(pebble_folder, pebble_select)
    visualize_calibration(res_arr)
    #res_arr = filter_calibration(Path('/export/scratch2/vladysla/Data/Real/POD/voltage_cu_50um'))
    #visualize_calibration(res_arr)
