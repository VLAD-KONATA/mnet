import os 
import time
import config
args, unparsed = config.get_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

import torch
from data import testSet
from util_evaluation import calc_psnr,calc_ssim
from select_model import select_model

import SimpleITK as sitk
mnad=False

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
    
    if mnad:
        m_items=torch.load(os.path.join(args.ckpt_dir + '/keys/', 'key.pt'))
    
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

        m = (gt.shape[2]-1) % args.upscale 
        if m != 0:
            gt = gt[...,:-m]
        lr = gt[...,::args.upscale]

        sr = torch.zeros_like(gt)
        sr_cnt = torch.zeros_like(gt)

        times=0
        begin_time=time.time()
        for tmp_s in range(lr.shape[2]-args.lr_slice_patch+1):
            tmp_lr = lr[...,tmp_s:tmp_s+args.lr_slice_patch]
            tmp_lr = tmp_lr.unsqueeze(0).cuda() #[1,s,h,w]
        
            with torch.no_grad():
                if mnad:
                    tmp_sr, _, _, m_items, softmax_score_query, softmax_score_memory, separateness_loss = model(tmp_lr,m_items,False)
                else:
                    tmp_sr,middle = model(tmp_lr)

            tmp_sr = torch.clamp(tmp_sr.squeeze(0),0,1).cpu()

            sr[...,tmp_s*args.upscale:tmp_s*args.upscale+((args.lr_slice_patch-1)*args.upscale+1)] += tmp_sr
            sr_cnt[...,tmp_s*args.upscale:tmp_s*args.upscale+((args.lr_slice_patch-1)*args.upscale+1)] += 1

        sr = sr[...,args.upscale : -1*args.upscale]
        sr_cnt = sr_cnt[...,args.upscale : -1*args.upscale]
        gt = gt[...,args.upscale : -1*args.upscale]

        sr /= sr_cnt #[h,w,s]
        end_time=time.time()
        times=end_time-begin_time
        
        if False:
            a=sr
            a=a.cpu().numpy()
            
            a=a.transpose(2,1,0)
            vol_ori = sitk.ReadImage('/home/konata/Dataset/IXI-T2/TAO_CT/origin/imagesTs/' +name[0]+'.nii.gz')
            vol_ori = sitk.Image(vol_ori)
            image = sitk.GetImageFromArray(a)
            image.SetSpacing(vol_ori.GetSpacing())
            image.SetOrigin(vol_ori.GetOrigin())
            image.SetDirection(vol_ori.GetDirection())
            sitk.WriteImage(image, '/home/konata/Dataset/IXI-T2/mnet/TAOCT_test/'+name[0]+'.nii.gz')
            print(name[0]+'.nii.gz')

        print(sr.shape) # h w s
        # print(sr.shape) # h w s
        sr = sr.cuda()
        gt = gt.cuda()
        psnr = calc_psnr(sr,gt).item()
        average_psnr += psnr

        gt = gt.cuda()
        sr = sr.cuda()
        for i in range(gt.shape[2]):
            ssim = calc_ssim(gt[:,:,i],sr[:,:,i])
            x_y_ssim += ssim
        x_y_ssim /= (i+1)
        for i in range(gt.shape[0]):
            ssim = calc_ssim(gt[i,:,:],sr[i,:,:])
            x_z_ssim += ssim
        x_z_ssim /= (i+1)
        for i in range(gt.shape[1]):
            ssim = calc_ssim(gt[:,i,:],sr[:,i,:])
            y_z_ssim += ssim
        y_z_ssim /= (i+1)
        log = r"[{} / {}] NAME:{} PSNR:{} x_y_ssim:{:.4f} x_z_ssim:{:.4f} y_z_ssim:{:.4f} time:{:.2f} "\
            .format(id+1,dataloader.__len__(),name,psnr,x_y_ssim, x_z_ssim,y_z_ssim,times)
        print(log)
        with open(args.ckpt_dir + '/logs_test.txt',mode='a+') as f:
            f.write(log+'\n') 

        total_x_y_ssim+=x_y_ssim
        total_x_z_ssim+=x_z_ssim
        total_y_z_ssim+=y_z_ssim

    average_psnr /= (id+1)
    total_x_y_ssim /= (id+1)
    total_x_z_ssim /= (id+1)
    total_y_z_ssim /= (id+1)
    print("average_psnr:",average_psnr) 
    print("average_x_y_ssim:",total_x_y_ssim) 
    print("average_x_z_ssim:",total_x_z_ssim) 
    print("average_y_z_ssim:",total_y_z_ssim) 

    with open(args.ckpt_dir + '/logs_test.txt',mode='a+') as f:
        log = r"PSNR: {} x_y_ssim: {:.6f} x_z_ssim: {:.6f} y_z_ssim: {:.6f}".format(average_psnr,total_x_y_ssim,total_x_z_ssim,total_y_z_ssim)
        f.write(log)


if __name__ == "__main__":
    main()
