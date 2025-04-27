import os 
import config
args, unparsed = config.get_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

import torch
import torch.nn as nn
import torch.nn.functional as F
from data import testSet
from util_evaluation import calc_psnr,calc_ssim
from select_model import select_model

import numpy as np
import SimpleITK as sitk

def pred_img(image,model):
    patchdims =(256,256,4)
    print(image.shape)
    strides = [patchdims[0] // 2, patchdims[1] // 2, patchdims[2] // 2]
    leng, col, row = image.shape[1], image.shape[2], image.shape[3]
    score_map = torch.zeros(size=(1, 256,256,13), dtype=torch.float32).to(device='cuda:0')
    cnt = torch.zeros(size=(leng, col, row), dtype=torch.float32).to(device='cuda:0')

    patch = 0
    ###使用了逐patch进行分割的方式；
    slice = [i for i in range(0, leng, strides[0])]
    slice[-1] = leng - patchdims[0]
    slice_ = [i for i in slice if i <= slice[-1]]
    if slice_[-1] == slice_[-2]:
        del slice_[-1]
    height = [i for i in range(0, col, strides[1])]
    height[-1] = col - patchdims[1]
    height_ = [i for i in height if i <= height[-1]]
    if height_[-1] == height_[-2]:
        del height_[-1]
    width = [i for i in range(0, row, strides[2])]
    width[-1] = row - patchdims[2]
    width_ = [i for i in width if i <= width[-1]]
    if width_[-1] == width_[-2]:
        del width_[-1]

    for i in slice_:
        for j in height_:
            for k in width_:
                patch += 1
                curpatch = image[ :, i:i + patchdims[0], j:j + patchdims[1], k:k + patchdims[2]]

                predict_seg = model(curpatch)
                predict_seg = F.softmax(predict_seg, dim=1)

                curpatchoutlabel = torch.squeeze(predict_seg)
                print(score_map.shape,curpatchoutlabel.shape)
                score_map[:, i:i + patchdims[0], j:j + patchdims[1], k:k + patchdims[2]] += curpatchoutlabel
                cnt[i:i + patchdims[0], j:j + patchdims[1], k:k + patchdims[2]] += 1

    ####--------------------------------------------------------
    labelmap0 = torch.zeros(size=[leng, col, row], dtype=torch.float32).to(device='cuda:0')
    seg_map = score_map / torch.unsqueeze(cnt, dim=0)
    for idx in range(0, leng):
        curslicelabel = torch.squeeze(seg_map[:, idx, ].argmax(axis=0))  ##一个最大投票原则；
        labelmap0[idx,] = curslicelabel
    return labelmap0

def main():
    args.ckpt_dir = 'experiments/'+args.model+'/'+args.ckpt_dir
    with open(args.ckpt_dir + '/logs_test.txt',mode='a+') as f:
        s = "\n\n\n\n\nSTART EXPERIMENT\n"
        f.write(s)
        f.write('testdata:'+args.testdata_path+'\n')
        f.write('checkpoint:'+args.ckpt+'\n')

    model = select_model(args)
    checkpoint = torch.load(args.ckpt, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['state_dict'])
    print(f'load:{args.ckpt}')
    model = model.cuda()
    model.eval()

    testset = testSet(data_root=args.testdata_path)
    dataloader = torch.utils.data.DataLoader(testset, batch_size=1,
    drop_last=False, shuffle=False, num_workers=4, pin_memory=False)

    average_psnr=0
    total_x_y_ssim=0
    total_x_z_ssim=0
    total_y_z_ssim=0


    for id, (name,volume) in enumerate(dataloader):
        # volume [bz=1,h,w,s]
        gt = volume.squeeze(0) #[h,w,s]
        psnr = 0
        x_y_ssim=0 
        x_z_ssim=0
        y_z_ssim=0
        '''
        m = (gt.shape[2]-1) % args.upscale 
        if m != 0:
            gt = gt[...,:-m]
        lr = gt[...,::args.upscale]
         '''
        lr = gt

        sr=torch.zeros(size=[gt.shape[0],gt.shape[1],gt.shape[2]*args.upscale])
        sr_cnt=torch.zeros(size=[gt.shape[0],gt.shape[1],gt.shape[2]*args.upscale])
        #sr = torch.zeros_like(gt)
        #sr_cnt = torch.zeros_like(gt)

        for tmp_s in range(lr.shape[2]-args.lr_slice_patch+1):
            tmp_lr = lr[...,tmp_s:tmp_s+args.lr_slice_patch]
            tmp_lr = tmp_lr.unsqueeze(0).cuda() #[1,s,h,w]
        
            with torch.no_grad():
                tmp_sr=pred_img(tmp_lr,model)
                #tmp_sr = model(tmp_lr)

            tmp_sr = torch.clamp(tmp_sr.squeeze(0),0,1).cpu()

            sr[...,tmp_s*args.upscale:tmp_s*args.upscale+((args.lr_slice_patch-1)*args.upscale+1)] += tmp_sr
            sr_cnt[...,tmp_s*args.upscale:tmp_s*args.upscale+((args.lr_slice_patch-1)*args.upscale+1)] += 1

        #sr = sr[...,args.upscale : -1*args.upscale]
        #sr_cnt = sr_cnt[...,args.upscale : -1*args.upscale]
        #gt = gt[...,args.upscale : -1*args.upscale]

        sr /= sr_cnt #[h,w,s]
        spacing=(0.3125,0.3125,1.925/2)
        a=sr
        a=a.cpu().numpy()
        a=a.transpose(2,0,1)
        vol_ori = sitk.ReadImage('/home/konata/Dataset/TED_MRI/T2/mask/origin/' +name[0]+'.nii.gz')
        vol_ori = sitk.Image(vol_ori)
        image = sitk.GetImageFromArray(a)
        image.SetSpacing(spacing)
        image.SetDirection(vol_ori.GetDirection())
        image.SetOrigin(vol_ori.GetOrigin())
        sitk.WriteImage(image, '/home/konata/Dataset/TED_MRI/T2/mask/origin_slice/testx4/'+name[0]+'.nii.gz')
        print(name[0]+'.nii.gz')
        print(sr.shape) # h w s



if __name__ == "__main__":
    main()
