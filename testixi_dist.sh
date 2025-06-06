
python -W ignore testixi_dist.py --upscale 2 --lr_slice_patch 4  --testdata_path /home/konata/Dataset/IXI-T2/I3Net/imagesTs --gpu_id '0' --model 'mamba_unet_dist'  --ckpt experiments/mamba_unet_dist/IXI_x2_2x4b/pth/0500.pth --ckpt_dir IXI_x2_2x4b  --num_workers 4
python -W ignore testixi_dist.py --upscale 2 --lr_slice_patch 4  --testdata_path /home/konata/Dataset/IXI-T2/I3Net/imagesTs --gpu_id '0' --model 'mamba_unet_dist'  --ckpt experiments/mamba_unet_dist/IXI_x2/pth/0500.pth --ckpt_dir IXI_x2  --num_workers 4
