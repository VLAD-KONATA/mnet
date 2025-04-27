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

    python -W ignore main.py --upscale 2 --lr_slice_patch 4 --traindata_path /home/konata/Dataset/IXI-T2/TAO_CT/I3Net/slice --testdata_path /home/konata/Dataset/IXI-T2/TAO_CT/I3Net/imagesTs --ckpt_dir TAOCT_x2 --batch_size 4 --gpu_id '0' --model 'mamba_intra' --num_workers 4
    python -W ignore main.py --upscale 2 --lr_slice_patch 4 --traindata_path /home/konata/Dataset/IXI-T2/TAO_CT/I3Net/slice --testdata_path /home/konata/Dataset/IXI-T2/TAO_CT/I3Net/imagesTs --ckpt_dir TAOCT_x2 --batch_size 4 --gpu_id '0' --model 'mamba_inter' --num_workers 4
    python -W ignore main.py --upscale 2 --lr_slice_patch 4 --traindata_path /home/konata/Dataset/IXI-T2/TAO_CT/I3Net/slice --testdata_path /home/konata/Dataset/IXI-T2/TAO_CT/I3Net/imagesTs --ckpt_dir TAOCT_x2_8b --batch_size 4 --gpu_id '0' --model 'mamba_inter_multi' --num_workers 4
