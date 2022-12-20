# # 代码中已经全部设置为False,设置为True时，需要解除 注释
export CUDA_VISIBLE_DEVICES=1,2,3
python -m torch.distributed.launch --nproc_per_node=3 --master_port=2234 model-trainer.py \
        --is_use_DDP \
        --current_dataset AISHELL-1 \
        --train_batch_size 100 \
        --dev_batch_size 100 \
        --test_batch_size 100 \
        --max_seq_length 40
# export CUDA_VISIBLE_DEVICES=2
# python model-trainer.py \
#         --current_dataset AISHELL-1 \
#         --train_batch_size 100 \
#         --dev_batch_size 100 \
#         --test_batch_size 100 \
#         --max_seq_length 40
# export CUDA_VISIBLE_DEVICES=2
# python model-trainer.py \
#         --current_dataset HKUST \
#         --train_batch_size 50 \
#         --dev_batch_size 50 \
#         --test_batch_size 50 \
#         --max_seq_length 80
