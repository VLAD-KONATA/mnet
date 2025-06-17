# save each slice of each volume

from medpy.io import load
import numpy as np
import nibabel as nib
import os
import pickle
datapath = '/home/konata/Dataset/IXI-T2/I3Net/imagesTr/'
volume_list = os.listdir(datapath)
savepath = '/home/konata/Dataset/IXI-T2/I3Net/slice/'
os.makedirs(savepath,exist_ok=True)

for i,volumename in enumerate(volume_list):
    volume_path = datapath + volumename
    # pt
    data = pickle.load(open(volume_path,'rb'))
    volnp = data['image'].astype("float32")
    spacing = data['spacing']

    savefile = savepath + volumename.split('.')[0]
    os.makedirs(savefile,exist_ok=True)

    for frame in range(volnp.shape[2]):
        vol_slice = volnp[:,:,frame]
        savename = savefile + '/' + volumename.split('.')[0] + '_slice_' + str(frame).zfill(3) + '.pt'
        pickle.dump({'image':vol_slice,'spacing':spacing},open(savename, 'wb'))

    print(f'{i} / {len(volume_list)}  ' + savefile)

print('done')
