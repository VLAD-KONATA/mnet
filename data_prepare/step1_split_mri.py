# following SAINT https://github.com/cpeng93/SAINT, split train/test volume
# save volume.pt: {'image':image_h_w_s.dtype('uint16'),'spacing':(x,y,z)}, image_scale=0-4095

from medpy.io import load
import os, pickle
import numpy as np

test_set = pickle.load(open('data_prepare/test_set.pt','rb'))
test_set = {k:test_set[k] for k in sorted(test_set.keys())}

src_path = '/home/konata/Dataset/IXI-T2/origin'
tgt_path = '/home/konata/Dataset/IXI-T2/I3Net'

os.makedirs(tgt_path,exist_ok=True)
os.makedirs(os.path.join(tgt_path,'imagesTr'),exist_ok=True)
os.makedirs(os.path.join(tgt_path,'imagesTs'),exist_ok=True)

def padding(img):
    h,w,d=img.shape
    target_z=256
    pad_top = (target_z -h) // 2
    pad_bottom = target_z - h - pad_top
    pad_left=(target_z - w) // 2
    pad_right=target_z-w-pad_left
    # 使用零填充或复制边缘像素填充
    padded = np.pad(img, ((pad_top, pad_bottom),(pad_left, pad_right),(0,0)), mode='constant',constant_values = (-1000,-1000))
    #padded_slice = np.pad(slice, ((pad_top, pad_bottom),(pad_left, pad_right)), mode='constant',constant_values = 0)
    return padded

for inst in ['imagesTr_lca','imagesTs_lca']:
    #file_dir = src_path
    file_dir = os.path.join(src_path, inst)
    patients = os.listdir(file_dir)
    for id,patient in enumerate(patients):
        if not '._' in patient:
            img, header = load(os.path.join(file_dir, patient))
            #img=padding(img)
            spacing = header.get_voxel_spacing()
            '''
            """ 将图像线性映射到 [target_min, target_max] 范围 """
            target_min=0
            target_max=1
            img_min, img_max = np.min(img), np.max(img)
            normalized = (img - img_min) / (img_max - img_min)  # 映射到 [0, 1]
            img=normalized * (target_max - target_min) + target_min

            img = img.astype("float32")
            '''
            img = np.clip(img,-1024,img.max())
            img = img - img.min()
            img = np.clip(img,0,4095)
            img = img.astype("uint16")
            data = {'image': img, 'spacing': spacing}
            #if not patient.split('.')[0] in test_set:
            if inst =='imagesTr':
                pickle.dump(data, open(os.path.join(tgt_path,'imagesTr', patient.replace('.nii.gz','.pt')), 'wb'))
                print(f"{id}/{len(patients)}:volume finished, " + patient, img.shape)

            else:
                pickle.dump(data, open(os.path.join(tgt_path,'imagesTs', patient.replace('.nii.gz','.pt')), 'wb'))
                print(f"{id}/{len(patients)}:volume finished, " + patient, img.shape)

print('done')
