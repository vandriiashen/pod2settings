from pathlib import Path
import numpy as np
import imageio.v3 as imageio
import tifffile
import matplotlib.pyplot as plt

data_classes = [
    {
        'src' : 'Fan Bone 1 to 25 times 4',
        'dest' : 'FanBone',
        'train' : 10,
        'val' : 3,
        'test' : 12
    },
    {
        'src' : 'NoBone 1 to 50  times 4',
        'dest' : 'NoBone',
        'train' : 20,
        'val' : 5,
        'test' : 25
    },
    {
        'src' : 'Rib Large 1 to 25 times 4',
        'dest' : 'RibBone',
        'train' : 10,
        'val' : 3,
        'test' : 12
    },
    {
        'src' : 'Rib Small 1 to 24 times 4',
        'dest' : 'RibBone',
        'train' : 10,
        'val' : 2,
        'test' : 12
    },
    {
        'src' : 'Wishbone Small 1 to 25 times 4',
        'dest' : 'WishBone',
        'train' : 10,
        'val' : 3,
        'test' : 12
    }
]
    
def make_folders(out_folder):
    out_folder.mkdir(exist_ok=True)
    train_folder = out_folder / 'train'
    val_folder = out_folder / 'val'
    test_folder = out_folder / 'test'
    train_folder.mkdir(exist_ok=True)
    val_folder.mkdir(exist_ok=True)
    test_folder.mkdir(exist_ok=True)
    for spec in data_classes:
        for subf in (train_folder, val_folder, test_folder):
            (subf / spec['dest']).mkdir(exist_ok=True)
            (subf / spec['dest'] / 'chA').mkdir(exist_ok=True)
            (subf / spec['dest'] / 'chB').mkdir(exist_ok=True)
    
def get_crop_bbox(im):
    ind = np.argwhere(im < 65535)
    y_min = ind[:,0].min()
    y_max = ind[:,0].max()
    x_min = ind[:,1].min()
    x_max = ind[:,1].max()
    return (y_min,y_max,x_min,x_max)

def copy_images(src_folder, dest_folder, sample_num, views, out_num):
    num = out_num
    for i in views:
        chA_fname = src_folder / '{:02d}_{}_CHANNEL_A.tif'.format(sample_num, i)
        chB_fname = src_folder / '{:02d}_{}_CHANNEL_B.tif'.format(sample_num, i)
        chA = imageio.imread(chA_fname)
        chB = imageio.imread(chB_fname)
        maskA = get_crop_bbox(chA)
        maskB = get_crop_bbox(chB)
        
        # the same crop
        chA = chA[maskA[0]:maskA[1], maskA[2]:maskA[3]]
        chB = chB[maskA[0]:maskA[1], maskA[2]:maskA[3]]
        assert chA.shape == chB.shape
        tifffile.imwrite(dest_folder / 'chA' / '{:04d}.tiff'.format(num), chA)
        tifffile.imwrite(dest_folder / 'chB' / '{:04d}.tiff'.format(num), chB)
        num += 1
        
def process_sample(files, sample_num, src_folder, dest_folder, out_num):
    name_start = '{:02d}_'.format(sample_num)
    select = [x for x in files if x.name.startswith(name_start)]
    select = [x for x in select if x.name.endswith('A.tif')]
    views = [x.name.partition(name_start)[2] for x in select]
    views = [x.partition('_CHANNEL')[0] for x in views]
        
    copy_images(src_folder, dest_folder, sample_num, views, out_num)
    incr = len(views)
    return incr

def process_spec(raw_data, out_folder, spec):
    src_folder = raw_data / spec['src']
    files = sorted(src_folder.glob('*.tif'))
    
    out_num = 0
    if spec['src'] == 'Rib Small 1 to 24 times 4':
        out_num = 40
    dest_folder = out_folder / 'train' / spec['dest']
    for i in range(1, spec['train']+1):
        incr = process_sample(files, i, src_folder, dest_folder, out_num)
        out_num += incr
        
    out_num = 0
    if spec['src'] == 'Rib Small 1 to 24 times 4':
        out_num = 12
    dest_folder = out_folder / 'val' / spec['dest']
    for i in range(spec['train']+1, spec['train']+spec['val']+1):
        incr = process_sample(files, i, src_folder, dest_folder, out_num)
        out_num += incr
        
    out_num = 0
    if spec['src'] == 'Rib Small 1 to 24 times 4':
        out_num = 48
    dest_folder = out_folder / 'test' / spec['dest']
    for i in range(spec['train']+spec['val']+1, spec['train']+spec['val']+spec['test']+1):
        incr = process_sample(files, i, src_folder, dest_folder, out_num)
        out_num += incr
        
def tiff2png(data_folder):
    tiff_folder = data_folder / 'chA'
    png_folder = data_folder / 'chA_png'
    png_folder.mkdir(exist_ok = True)
    tiffs = sorted(tiff_folder.glob('*.tiff'))
    for full_name in tiffs:
        img = imageio.imread(full_name)
        imageio.imwrite(png_folder / '{}.png'.format(full_name.stem), img)
        
def png2tiff(data_folder):
    tiff_folder = data_folder / 'segm'
    png_folder = data_folder / 'segm_png'
    tiff_folder.mkdir(exist_ok = True)
    pngs = sorted(png_folder.glob('*.png'))
    for full_name in pngs[:]:
        img = imageio.imread(full_name)
        gs = img[:,:,0:3].mean(axis=2)
        binary = np.where(gs > 0, 1, 0)
        tifffile.imwrite(tiff_folder / '{}.tiff'.format(full_name.stem), binary.astype(np.uint8))

if __name__ == "__main__":
    raw_data = Path('/export/scratch2/vladysla/Data/Real/MEYN_chicken/Standard')
    out_folder = Path('/export/scratch2/vladysla/Data/Real/MEYN_chicken/meyn_standard')
    
    #tiff2png(out_folder / 'test' / 'FanBone')
    png2tiff(out_folder / 'test' / 'FanBone')
    '''
    make_folders(out_folder)
    for i in range(5):
        process_spec(raw_data, out_folder, data_classes[i])
    '''
