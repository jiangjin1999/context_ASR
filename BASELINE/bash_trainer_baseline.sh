# # 代码中已经全部设置为False,设置为True时，需要解除 注释
export CUDA_VISIBLE_DEVICES=1,2
python -m torch.distributed.launch --nproc_per_node=2 --master_port=22345 baseline.py \
        --is_use_DDP \
        --current_dataset HKUST \
        --max_seq_length 80 \
        --batch_size 60
export CUDA_VISIBLE_DEVICES=1,2
python -m torch.distributed.launch --nproc_per_node=2 --master_port=22346 baseline.py \
        --is_use_DDP \
        --current_dataset AISHELL-1 \
        --max_seq_length 40 \
        --batch_size 120


