python -W ignore main.py \
    --upscale 2 \
    --lr_slice_patch 4 \
    --traindata_path /home/konata/Dataset/IXI-T2/I3Net/slice  \
    --testdata_path /home/konata/Dataset/IXI-T2/I3Net/imagesTs \
    --ckpt_dir IXI_x2 \
    --batch_size 4 \
    --gpu_id '0' \
    --model 'mnet' \
    --num_workers 4

    python -W ignore main.py --upscale 2 --lr_slice_patch 4 --traindata_path /home/konata/Dataset/IXI-T2/I3Net/slice --testdata_path /home/konata/Dataset/IXI-T2/I3Net/imagesTs --ckpt_dir IXI_x2 --batch_size 4 --gpu_id '0' --model 'mnad' --num_workers 4
    python -W ignore main.py --upscale 2 --lr_slice_patch 4 --traindata_path /home/konata/Dataset/IXI-T2/I3Net/slice --testdata_path /home/konata/Dataset/IXI-T2/I3Net/imagesTs --ckpt_dir IXI_x2 --batch_size 4 --gpu_id '0' --model 'mamba_intra' --num_workers 4
    python -W ignore main.py --upscale 2 --lr_slice_patch 4 --traindata_path /home/konata/Dataset/IXI-T2/I3Net/slice --testdata_path /home/konata/Dataset/IXI-T2/I3Net/imagesTs --ckpt_dir IXI_x2 --batch_size 4 --gpu_id '0' --model 'mambaccc_inter' --num_workers 4
    python -W ignore main.py --upscale 2 --lr_slice_patch 4 --traindata_path /home/konata/Dataset/IXI-T2/I3Net/slice --testdata_path /home/konata/Dataset/IXI-T2/I3Net/imagesTs --ckpt_dir IXI_x2_8b --batch_size 4 --gpu_id '0' --model 'mamba_inter_multi' --num_workers 4
    python -W ignore main.py --upscale 2 --lr_slice_patch 4 --traindata_path /home/konata/Dataset/IXI-T2/I3Net/slice --testdata_path /home/konata/Dataset/IXI-T2/I3Net/imagesTs --ckpt_dir IXI_x2_2x4b --batch_size 4 --gpu_id '0' --model 'mamba_inter_multi' --num_workers 4
    python -W ignore main.py --upscale 2 --lr_slice_patch 4 --traindata_path /home/konata/Dataset/IXI-T2/I3Net/slice --testdata_path /home/konata/Dataset/IXI-T2/I3Net/imagesTs --ckpt_dir IXI_x2 --batch_size 4 --gpu_id '0' --model 'mnad_mamba' --num_workers 4
    python -W ignore main.py --upscale 2 --lr_slice_patch 4 --traindata_path /home/konata/Dataset/IXI-T2/I3Net/slice --testdata_path /home/konata/Dataset/IXI-T2/I3Net/imagesTs --ckpt_dir IXI_x2 --batch_size 4 --gpu_id '0' --model 'mamba_unet' --num_workers 4
    python -W ignore main.py --upscale 2 --lr_slice_patch 4 --traindata_path /home/konata/Dataset/IXI-T2/I3Net/slice --testdata_path /home/konata/Dataset/IXI-T2/I3Net/imagesTs --ckpt_dir IXI_x2 --batch_size 4 --gpu_id '0' --model 'mamba_unet_multi' --num_workers 4
    python -W ignore main.py --upscale 2 --lr_slice_patch 4 --traindata_path /home/konata/Dataset/IXI-T2/I3Net/slice --testdata_path /home/konata/Dataset/IXI-T2/I3Net/imagesTs --ckpt_dir IXI_x2_4b --batch_size 4 --gpu_id '0' --model 'mamba_unet_multi' --num_workers 4
    python -W ignore main.py --upscale 2 --lr_slice_patch 4 --traindata_path /home/konata/Dataset/IXI-T2/I3Net/slice --testdata_path /home/konata/Dataset/IXI-T2/I3Net/imagesTs --ckpt_dir IXI_x2_8b --batch_size 4 --gpu_id '0' --model 'mamba_unet_multi' --num_workers 4
    python -W ignore main.py --upscale 2 --lr_slice_patch 4 --traindata_path /home/konata/Dataset/IXI-T2/I3Net/slice --testdata_path /home/konata/Dataset/IXI-T2/I3Net/imagesTs --ckpt_dir IXI_x2_2x4b --batch_size 4 --gpu_id '0' --model 'mamba_unet_multi' --num_workers 4
    python -W ignore main.py --upscale 2 --lr_slice_patch 4 --traindata_path /home/konata/Dataset/IXI-T2/I3Net/slice --testdata_path /home/konata/Dataset/IXI-T2/I3Net/imagesTs --ckpt_dir IXI_x2 --batch_size 4 --gpu_id '0' --model 'unet_mamba_test' --num_workers 4
    python -W ignore main.py --upscale 2 --lr_slice_patch 4 --traindata_path /home/konata/Dataset/IXI-T2/I3Net/slice --testdata_path /home/konata/Dataset/IXI-T2/I3Net/imagesTs --ckpt_dir IXI_x2 --batch_size 4 --gpu_id '0' --model 'unet' --num_workers 4
    python -W ignore main.py --upscale 2 --lr_slice_patch 4 --traindata_path /home/konata/Dataset/IXI-T2/I3Net/slice --testdata_path /home/konata/Dataset/IXI-T2/I3Net/imagesTs --ckpt_dir IXI_x2 --batch_size 4 --gpu_id '0' --model 'mamba' --num_workers 4
    python -W ignore main.py --upscale 2 --lr_slice_patch 4 --traindata_path /home/konata/Dataset/IXI-T2/I3Net/slice --testdata_path /home/konata/Dataset/IXI-T2/I3Net/imagesTs --ckpt_dir IXI_x2 --batch_size 4 --gpu_id '0' --model 'unet-unet_mamba' --num_workers 4
