# # 代码中已经全部设置为False,设置为True时，需要解除 注释
export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m torch.distributed.launch --nproc_per_node=4 --master_port=22345 baseline.py \
        --is_use_DDP \
        --current_dataset AISHELL-1 \
        --batch_size 5

