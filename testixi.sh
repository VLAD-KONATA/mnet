python test.py \
    --upscale 2 \
    --lr_slice_patch 4 \
    --testdata_path /home/konata/Dataset/IXI-T2/I3Net/imagesTs/ \
    --gpu_id '0' \
    --model mnet \
    --ckpt experiments/mnet/IXI_x2/pth/0150.pth \
    --ckpt_dir IXI_x2 \
    --num_workers 4

python -W ignore testixi.py --upscale 2 --lr_slice_patch 4  --testdata_path /home/konata/Dataset/IXI-T2/I3Net/imagesTs --gpu_id '0' --model 'mamba_inter'  --ckpt experiments/mamba_inter/IXI_x2/pth/0050.pth --ckpt_dir IXI_x2  --num_workers 4
python -W ignore testixi.py --upscale 2 --lr_slice_patch 4  --testdata_path /home/konata/Dataset/IXI-T2/I3Net/imagesTs --gpu_id '0' --model 'mamba_intra'  --ckpt experiments/mamba_intra/IXI_x2/pth/0050.pth --ckpt_dir IXI_x2  --num_workers 4
python -W ignore testixi.py --upscale 2 --lr_slice_patch 4  --testdata_path /home/konata/Dataset/IXI-T2/I3Net/imagesTs --gpu_id '0' --model 'mamba_inter_multi'  --ckpt experiments/mamba_inter_multi/IXI_x2_8b/pth/0050.pth --ckpt_dir IXI_x2_8b  --num_workers 4
python -W ignore testixi.py --upscale 2 --lr_slice_patch 4  --testdata_path /home/konata/Dataset/IXI-T2/I3Net/imagesTs --gpu_id '0' --model 'mamba_inter_multi'  --ckpt experiments/mamba_inter_multi/IXI_x2_2x4b/pth/0050.pth --ckpt_dir IXI_x2_2x4b  --num_workers 4
python -W ignore testixi.py --upscale 2 --lr_slice_patch 4  --testdata_path /home/konata/Dataset/IXI-T2/I3Net/imagesTs --gpu_id '0' --model 'mnad_mamba'  --ckpt experiments/mnad_mamba/IXI_x2/pth/0050.pth --ckpt_dir IXI_x2  --num_workers 4
python -W ignore testixi.py --upscale 2 --lr_slice_patch 4  --testdata_path /home/konata/Dataset/IXI-T2/I3Net/imagesTs --gpu_id '0' --model 'mamba_unet'  --ckpt experiments/mamba_unet/IXI_x2/pth/0050.pth --ckpt_dir IXI_x2  --num_workers 4
python -W ignore testixi.py --upscale 2 --lr_slice_patch 4  --testdata_path /home/konata/Dataset/IXI-T2/I3Net/imagesTs --gpu_id '0' --model 'mamba_unet_multi'  --ckpt experiments/mamba_unet_multi/IXI_x2_4b/pth/0050.pth --ckpt_dir IXI_x2_4b  --num_workers 4
python -W ignore testixi.py --upscale 2 --lr_slice_patch 4  --testdata_path /home/konata/Dataset/IXI-T2/I3Net/imagesTs --gpu_id '0' --model 'mamba_unet_multi'  --ckpt experiments/mamba_unet_multi/IXI_x2_8b/pth/0050.pth --ckpt_dir IXI_x2_8b  --num_workers 4
python -W ignore testixi.py --upscale 2 --lr_slice_patch 4  --testdata_path /home/konata/Dataset/IXI-T2/I3Net/imagesTs --gpu_id '0' --model 'mamba_unet_multi'  --ckpt experiments/mamba_unet_multi/IXI_x2_2x4b/pth/0050.pth --ckpt_dir IXI_x2_2x4b  --num_workers 4
python -W ignore testixi.py --upscale 2 --lr_slice_patch 4  --testdata_path /home/konata/Dataset/IXI-T2/I3Net/imagesTs --gpu_id '0' --model 'mamba_unet_multi'  --ckpt experiments/mamba_unet_multi/IXI_x2_2x4b/pth/0500.pth --ckpt_dir IXI_x2_2x4b  --num_workers 4
python -W ignore testixi.py --upscale 2 --lr_slice_patch 4  --testdata_path /home/konata/Dataset/IXI-T2/I3Net/imagesTs --gpu_id '0' --model 'unet'  --ckpt experiments/unet/IXI_x2/pth/0050.pth --ckpt_dir IXI_x2  --num_workers 4
python -W ignore testixi.py --upscale 2 --lr_slice_patch 4  --testdata_path /home/konata/Dataset/IXI-T2/I3Net/imagesTs --gpu_id '0' --model 'mamba'  --ckpt experiments/mamba/IXI_x2/pth/0050.pth --ckpt_dir IXI_x2  --num_workers 4
python -W ignore testixi.py --upscale 2 --lr_slice_patch 4  --testdata_path /home/konata/Dataset/IXI-T2/I3Net/imagesTs --gpu_id '0' --model 'unet-unet_mamba'  --ckpt experiments/unet-unet_mamba/IXI_x2/pth/0050.pth --ckpt_dir IXI_x2  --num_workers 4
