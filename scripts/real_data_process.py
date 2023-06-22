import numpy as np
from pathlib import Path
import os
import imageio.v3 as imageio
import tifffile
import shutil
import matplotlib.pyplot as plt

num_classes = 2

class_to_name = {
    0 : "NoBone",
    1 : "RibBone"
}

dataset_names = [
    '90kV_45W_100ms_10avg',
    '90kV_15W_20ms_1avg',
    '90kV_3W_100ms_10avg',
    '90kV_3W_100ms_1avg',
    '90kV_3W_50ms_1avg',
    '90kV_3W_15ms_1avg',
    '40kV_40W_100ms_10avg',
    '40kV_3W_50ms_1avg'
]

def get_meta_information(raw_data):
    class_counts = np.zeros((num_classes,), dtype=int)
    for f in raw_data:
        stats = np.genfromtxt(f / 'stats.csv', delimiter=',', names=True)
        for i in range(num_classes):
            class_counts[i] += np.count_nonzero(stats['Bone_Class'] == i)
                        
    return class_counts

def get_fo_size(bone_class, bone_id):
    data_folder = Path('/export/scratch2/vladysla/Data/Real/POD/chicken_09_06_cu_50um')
    size_table = np.genfromtxt(data_folder / 'bone_size.csv', delimiter=',', names=True)
    
    if bone_class != 0:
        select = np.logical_and(size_table['Bone_Class'] == bone_class,
                                size_table['Bone'] == bone_id)
        size = size_table[select]['Size_mm'][0]
    else:
        size = 0.
    return size

def log_cor(src, dest, di, ff):
    img = imageio.imread(src)
    di = imageio.imread(di)
    img -= di
    img = img.astype(np.float32)
    ff = ff.astype(np.float32)
    img /= ff
    img = -np.log(img)
    mean = img.mean()
    std = img.std()
    img -= mean
    img /= std
    tifffile.imwrite(dest, img)
    
def copy_files_from_folder(inp_folder, out_data, cl, start_num, split_num):
    stats = np.genfromtxt(inp_folder / 'stats.csv', delimiter=',', names=True)
    stats = stats[stats['Bone_Class'] == cl]
    
    train_stats = {}
    val_stats = {}
    ff_dset = {}
    for dset in dataset_names:
        train_stats[dset] = open(out_data / dset / 'train' / '{}.csv'.format(class_to_name[cl]), 'a')
        val_stats[dset] = open(out_data / dset / 'val' / '{}.csv'.format(class_to_name[cl]), 'a')
        ff = imageio.imread(inp_folder / 'ff' / '{}.tif'.format(dset))
        di = imageio.imread(inp_folder / 'ff' / 'di_pre.tif')
        ff_dset[dset] = ff - di
    
    cur_id = start_num
    
    for sample_id in stats['Sample']:
        select = stats[stats['Sample'] == sample_id]
        int_id = int(select['Sample'][0])
        size = get_fo_size(int(select['Bone_Class'][0]), int(select['Bone'][0]))
        stats_string = '{},{:03d},{},{},{},{}\n'.format(cur_id, int_id, int(select['Chicken'][0]), int(select['Bone_Class'][0]), int(select['Bone'][0]), size)
        print(cur_id)
        for dset in dataset_names:
            if cur_id < split_num:
                log_cor(inp_folder / '{:03d}'.format(int_id) / '{}.tif'.format(dset),
                        out_data / dset / 'train' / class_to_name[cl] / '{:03d}.tiff'.format(cur_id),
                        inp_folder / '{:03d}'.format(int_id) / 'di_pre.tif', ff_dset[dset])
                train_stats[dset].write(stats_string)
                
                if cl != 0:
                    segm_png = imageio.imread(inp_folder / '{:03d}'.format(int_id) / 'segm.png')
                    print(segm_png.shape)
                    segm = np.copy(segm_png[:,:,0])
                    segm[segm > 0] = 1
                    segm = segm.astype(np.uint8)
                    print(segm.shape)
                    tifffile.imwrite(out_data / dset / 'train' / '{}_segm'.format(class_to_name[cl]) / '{:03d}.tiff'.format(cur_id), segm)
            else:
                log_cor(inp_folder / '{:03d}'.format(int_id) / '{}.tif'.format(dset),
                        out_data / dset / 'val' / class_to_name[cl] / '{:03d}.tiff'.format(cur_id),
                        inp_folder / '{:03d}'.format(int_id) / 'di_pre.tif', ff_dset[dset])
                val_stats[dset].write(stats_string)
                
                if cl != 0:
                    segm_png = imageio.imread(inp_folder / '{:03d}'.format(int_id) / 'segm.png')
                    print(segm_png.shape)
                    segm = np.copy(segm_png[:,:,0])
                    segm[segm > 0] = 1
                    segm = segm.astype(np.uint8)
                    print(segm.shape)
                    tifffile.imwrite(out_data / dset / 'val' / '{}_segm'.format(class_to_name[cl]) / '{:03d}.tiff'.format(cur_id), segm)
        cur_id += 1

        
    for key in train_stats.keys():
        train_stats[key].close()
    for key in val_stats.keys():
        val_stats[key].close()
    
    return cur_id
    
def create_folders(out_data):
    for dset in dataset_names:
        df = out_data / dset
        df.mkdir(exist_ok=True)
        for subset in ['train', 'val', 'test']:
            dfs = df / subset
            dfs.mkdir(exist_ok=True)
            for cl in class_to_name.values():
                cl_folder = dfs / str(cl)
                cl_folder.mkdir(exist_ok=True)
                segm_folder = dfs / '{}_segm'.format(cl)
                segm_folder.mkdir(exist_ok=True)
                with open(dfs / '{}.csv'.format(cl), 'w') as f:
                    f.write('ID,Original_ID,Chicken_ID,Bone_Class,Bone_ID,FO_size\n')
                    
def copy_all_files(raw_data, out_data):
    class_counts = get_meta_information(raw_data)
    for i in range(num_classes):
        # 20% to validation
        split_num = class_counts[i] - class_counts[i] // 5
        cur_id = 0
        for folder in raw_data:
            cur_id = copy_files_from_folder(folder, out_data, i, cur_id, split_num)
            
def tiff2png(data_folder):
    stats = np.genfromtxt(data_folder / 'stats.csv', delimiter=',', names=True)
    select = stats['Bone_Class'] == 1
    for s in stats['Sample'][select]:
        subf = data_folder / '{:03d}'.format(int(s))
        img = imageio.imread(subf / '40kV_40W_100ms_10avg.tif')
        img = img.astype(np.float32)
        di = imageio.imread(subf / 'di_pre.tif')
        ff = imageio.imread(data_folder / 'ff' / '40kV_40W_100ms_10avg.tif')
        ff = ff.astype(np.float32)
        ff_di = imageio.imread(data_folder / 'ff' / 'di_pre.tif')
        ff -= ff_di
        
        img -= di
        img /= ff
        img = -np.log(img)
        
        img -= img.min()
        img /= img.max()
        img *= 255
        img = img.astype(np.uint8)
        
        imageio.imwrite(subf / 'ref.png', img)
        
if __name__ == "__main__":
    raw_data = [Path('/export/scratch2/vladysla/Data/Real/POD/chicken_09_06_cu_50um'),
                Path('/export/scratch2/vladysla/Data/Real/POD/chicken_10_06_cu_50um'),
                Path('/export/scratch2/vladysla/Data/Real/POD/chicken')]
    out_data = Path('/export/scratch2/vladysla/Data/Real/POD/datasets')

    #get_meta_information(raw_data)
    create_folders(out_data)
    #print(get_fo_size(1, 10))
    #copy_files_from_folder(raw_data[0], out_data, 1, 0, 500)
    copy_all_files(raw_data, out_data)
    #tiff2png(raw_data[2])
