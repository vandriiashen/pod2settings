import numpy as np
from pathlib import Path
import os
import imageio.v3 as imageio
import tifffile
import shutil
from skimage import morphology
import matplotlib.pyplot as plt

num_classes = 2

class_to_name = {
    0 : "NoBone",
    1 : "RibBone"
}
exp1_dataset_names = [
    '90kV_45W_100ms_10avg',
    '40kV_40W_100ms_10avg'
]
exp2_dataset_names = [
    '90kV_45W_100ms_10avg',
    '90kV_45W_100ms_1avg',
    '90kV_45W_50ms_1avg',
    '90kV_45W_20ms_1avg',
    '40kV_40W_100ms_10avg',
    '40kV_40W_100ms_1avg',
    '40kV_40W_50ms_1avg',
    '40kV_40W_20ms_1avg'
]

def get_meta_information(raw_data):
    class_counts = np.zeros((num_classes,), dtype=int)
    for f in raw_data:
        stats = np.genfromtxt(f / 'stats.csv', delimiter=',', names=True)
        for i in range(num_classes):
            class_counts[i] += np.count_nonzero(stats['Bone_Class'] == i)
                        
    return class_counts

def get_fo_size(bone_class, bone_id):
    data_folder = Path('/path/to/data')
    size_table = np.genfromtxt(data_folder / 'bone_size.csv', delimiter=',', names=True)
    
    if bone_class != 0:
        select = np.logical_and(size_table['Bone_Class'] == bone_class,
                                size_table['Bone'] == bone_id)
        size = size_table[select]['Size_mm'][0]
    else:
        size = 0.
    return size

def log_cor(src, dest, di, ff):
    img = imageio.imread(src).astype(np.float32)
    di = imageio.imread(di).astype(np.float32)
    img -= di
    img[img < 1.] = 1.
    ff[ff < 1.] = 1.
    
    ff = ff.astype(np.float32)
    div = img / ff
    div = -np.log(div)
    
    div[:15,:] = 0.
    div[-15:,:] = 0.
    div[:,:15] = 0.
    div[:,-15:] = 0.
    
    tifffile.imwrite(dest, div)
    
def tiff2png(data_folder):
    stats = np.genfromtxt(data_folder / 'stats.csv', delimiter=',', names=True)
    select = stats['Bone_Class'] == 1
    for s in stats['Sample'][select]:
        subf = data_folder / '{:03d}'.format(int(s))
        ref_name = '40kV_40W_100ms_10avg'
        #ref_name = '40kV_60W_100ms_600avg'
        img = imageio.imread(subf / '{}.tif'.format(ref_name))
        img = img.astype(np.float32)
        di = imageio.imread(subf / 'di_pre.tif')
        ff = imageio.imread(data_folder / 'ff' / '{}.tif'.format(ref_name))
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
    
def compute_quotient(int_id, cur_id, ff_dset, inp_folder, dest):
    f1 = '40kV_40W_100ms_10avg'
    f2 = '90kV_45W_100ms_10avg'

    img_1 = imageio.imread(inp_folder / '{:03d}'.format(int_id) / '{}.tif'.format(f1)).astype(np.float32)
    di_1 = imageio.imread(inp_folder / '{:03d}'.format(int_id) / 'di_pre.tif').astype(np.float32)
    ff_1 = ff_dset[f1]
    img_1 -= di_1
    img_1[img_1 < 1.] = 1.
    ff_1[ff_1 < 1.] = 1.
    ff_1 = ff_1.astype(np.float32)
    img_1 = img_1 / ff_1
    img_1 = -np.log(img_1)
    
    img_2 = imageio.imread(inp_folder / '{:03d}'.format(int_id) / '{}.tif'.format(f2)).astype(np.float32)
    di_2 = imageio.imread(inp_folder / '{:03d}'.format(int_id) / 'di_pre.tif').astype(np.float32)
    ff_2 = ff_dset[f2]
    img_2 -= di_2
    img_2[img_2 < 1.] = 1.
    ff_2[ff_2 < 1.] = 1.
    ff_2 = ff_2.astype(np.float32)
    img_2 = img_2 / ff_2
    img_2 = -np.log(img_2)
    
    q = img_1 / img_2
    tifffile.imwrite(dest, q)
    return q
    
def quotient_features(chA, chB, segm, int_id):
    fig, ax = plt.subplots(1, 2, figsize=(18,9))
    q = np.divide(chA, chB, out=np.zeros_like(chA), where=chB!=0)
        
    show_q = np.zeros((*q.shape, 3))
    vmin = 1.
    vmax = 2.5
    q = np.nan_to_num(q)
    q[q > 10.] = 10
    q[q < 0.] = 0
    for k in range(3):
        show_q[:,:,k] = (q - vmin) / (vmax - vmin)
        
    fo_segm = segm[1,:] > 0
    nb_segm = morphology.binary_dilation(fo_segm)
    for k in range(20):
        nb_segm = morphology.binary_dilation(nb_segm)
                        
    nb_segm = np.logical_and(nb_segm, segm[0,:] > 0)
        
    # Highlight FO
    show_q[morphology.binary_dilation(fo_segm, footprint=np.ones((5, 5))) ^ fo_segm,0] = 0
    show_q[morphology.binary_dilation(fo_segm, footprint=np.ones((5, 5))) ^ fo_segm,2] = 0
    # Highlight neighborhood of FO
    show_q[morphology.binary_dilation(nb_segm, footprint=np.ones((5, 5))) ^ nb_segm,0] = 0
    show_q[morphology.binary_dilation(nb_segm, footprint=np.ones((5, 5))) ^ nb_segm,1] = 0
        
    nb_segm = np.logical_and(nb_segm, np.logical_not(fo_segm))
        
    ax[0].imshow(show_q)
        
    ax[1].hist(q[nb_segm], bins=40, density=True, color='b', label = 'MO')
    ax[1].hist(q[fo_segm], bins=40, density=True, color='g', label = 'FO')
    ax[1].legend()
    mo_mean = q[nb_segm].mean()
    fo_mean = q[fo_segm].mean()
    mo_std = q[nb_segm].std()
    snr = (fo_mean - mo_mean) / mo_std
    contrast = fo_mean - mo_mean
    att_fo = chA[fo_segm].mean()
    ax[1].set_title('S = {:.3f} | N = {:.3f} | SNR = {:.1f}'.format(fo_mean - mo_mean, mo_std, snr))
        
    plt.tight_layout()
    #plt.show()
    plt.savefig('tmp_imgs/quotient/{}.png'.format(int_id))
    
    return att_fo, contrast

def compute_properties(stats, sample_id, cur_id, cl, inp_folder, ff_dset):
    select = stats[stats['Sample'] == sample_id]
    int_id = int(select['Sample'][0])
    size = get_fo_size(int(select['Bone_Class'][0]), int(select['Bone'][0]))
        
    ref_name = '40kV_40W_100ms_10avg'
    ref_img = imageio.imread(inp_folder / '{:03d}'.format(int_id) / '{}.tif'.format(ref_name)).astype(np.float32)
    di = imageio.imread(inp_folder / '{:03d}'.format(int_id) / 'di_pre.tif').astype(np.float32)
    ref_img -= di
    ref_img = -np.log(ref_img / ff_dset[ref_name])
    segm = np.zeros((2, *(ff_dset[ref_name].shape)), dtype=np.uint8)
    (segm[0,:,:])[ref_img > 0.1] = 1
    if cl == 1:
        segm_png = imageio.imread(inp_folder / '{:03d}'.format(int_id) / 'segm.png')[:,:,0]
        (segm[1,:,:])[segm_png > 0] = 1
        
    chA = imageio.imread(inp_folder / '{:03d}'.format(int_id) / '40kV_40W_100ms_10avg.tif').astype(np.float32)
    chA -= di
    chA = -np.log(chA / ff_dset['40kV_40W_100ms_10avg'])
    chB = imageio.imread(inp_folder / '{:03d}'.format(int_id) / '90kV_45W_100ms_10avg.tif'.format(ref_name)).astype(np.float32)
    chB -= di
    chB = -np.log(chB / ff_dset['90kV_45W_100ms_10avg'])
    if cl == 1:
        att_fo, contrast = quotient_features(chA, chB, segm, int_id)
    else:
        att_fo, contrast = -1, -1
        
    stats_string = '{},{:03d},{},{},{},{},{:.3f},{:.3f}\n'.format(cur_id, int_id, int(select['Chicken'][0]), int(select['Bone_Class'][0]), int(select['Bone'][0]), size, att_fo, contrast)
    print(cur_id)
    print(stats_string)
    
    return int_id, stats_string, segm
    
    
def copy_files_from_folder(inp_folder, out_data, cl, dataset_names, start_num, split_num):
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
        int_id, stats_string, segm = compute_properties(stats, sample_id, cur_id, cl, inp_folder, ff_dset)
        
        for dset in dataset_names:
            if cur_id < split_num:
                log_cor(inp_folder / '{:03d}'.format(int_id) / '{}.tif'.format(dset),
                        out_data / dset / 'train' / class_to_name[cl] / '{:03d}.tiff'.format(cur_id),
                        inp_folder / '{:03d}'.format(int_id) / 'di_pre.tif', ff_dset[dset])
                train_stats[dset].write(stats_string)
                
                tifffile.imwrite(out_data / dset / 'train' / '{}_segm'.format(class_to_name[cl]) / '{:03d}.tiff'.format(cur_id), segm)
            else:
                log_cor(inp_folder / '{:03d}'.format(int_id) / '{}.tif'.format(dset),
                        out_data / dset / 'val' / class_to_name[cl] / '{:03d}.tiff'.format(cur_id),
                        inp_folder / '{:03d}'.format(int_id) / 'di_pre.tif', ff_dset[dset])
                val_stats[dset].write(stats_string)
                
                tifffile.imwrite(out_data / dset / 'val' / '{}_segm'.format(class_to_name[cl]) / '{:03d}.tiff'.format(cur_id), segm)
                                    
        cur_id += 1

    for key in train_stats.keys():
        train_stats[key].close()
    for key in val_stats.keys():
        val_stats[key].close()
    
    return cur_id

def copy_test_files_from_folder(inp_folder, out_data, cl, dataset_names, start_num):
    stats = np.genfromtxt(inp_folder / 'stats.csv', delimiter=',', names=True)
    stats = stats[stats['Bone_Class'] == cl]
    
    test_stats = {}
    ff_dset = {}
    for dset in dataset_names:
        test_stats[dset] = open(out_data / dset / 'test' / '{}.csv'.format(class_to_name[cl]), 'a')
            
        ff = imageio.imread(inp_folder / 'ff' / '{}.tif'.format(dset))
        di = imageio.imread(inp_folder / 'ff' / 'di_pre.tif')
        ff_dset[dset] = ff - di
        
    cur_id = start_num
    
    for sample_id in stats['Sample']:
        int_id, stats_string, segm = compute_properties(stats, sample_id, cur_id, cl, inp_folder, ff_dset)
        
        for dset in dataset_names:
            log_cor(inp_folder / '{:03d}'.format(int_id) / '{}.tif'.format(dset),
                    out_data / dset / 'test' / class_to_name[cl] / '{:03d}.tiff'.format(cur_id),
                    inp_folder / '{:03d}'.format(int_id) / 'di_pre.tif', ff_dset[dset])
            
            test_stats[dset].write(stats_string)
            
            tifffile.imwrite(out_data / dset / 'test' / '{}_segm'.format(class_to_name[cl]) / '{:03d}.tiff'.format(cur_id), segm)
                    
        cur_id += 1

    for key in test_stats.keys():
        test_stats[key].close()
        
    return cur_id
    
def create_folders(out_data, dataset_names):
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
                    f.write('ID,Original_ID,Chicken_ID,Bone_Class,Bone_ID,FO_size,Attenuation,Contrast\n')
                    
def copy_train_files(raw_data, out_data, dataset_names):
    class_counts = get_meta_information(raw_data)
    for i in range(num_classes):
        # 20% to validation
        split_num = class_counts[i] - class_counts[i] // 5
        cur_id = 0
        for folder in raw_data:
            cur_id = copy_files_from_folder(folder, out_data, i, dataset_names, cur_id, split_num)
            
def copy_test_files(raw_data, out_data, dataset_names):
    for i in range(num_classes):
        cur_id = 0
        for folder in raw_data:
            cur_id = copy_test_files_from_folder(folder, out_data, i, dataset_names, cur_id)
        
if __name__ == "__main__":
    train_data = [Path('/path/to/data')]
    out_data = Path('/path/to/data')
    
    # This function was used to create png images for manual labeling
    #tiff2png(train_data[0])
    
    create_folders(out_data, exp1_dataset_names)
    copy_train_files(train_data, out_data, exp1_dataset_names)
    copy_test_files(test_data1, out_data, exp1_dataset_names)
    
    test_data2 = [Path('/path/to/data')]
    out_test2 = Path('/path/to/data')
    
    create_folders(out_test2, exp2_dataset_names)
    copy_test_files(test_data2, out_test2, exp2_dataset_names)
