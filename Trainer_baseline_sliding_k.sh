# export CUDA_VISIBLE_DEVICES=2
# batch_size=128
# python model-trainer.py \
#         --current_dataset AISHELL-1 \
#         --train_batch_size $batch_size \
#         --dev_batch_size 128 \
#         --test_batch_size 128 \
#         --max_seq_length 45 \
#         --is_sliding_k 2

# export CUDA_VISIBLE_DEVICES=3
# batch_size=64
# python model-trainer.py \
#         --current_dataset HKUST \
#         --train_batch_size $batch_size \
#         --dev_batch_size 64 \
#         --test_batch_size 64 \
#         --max_seq_length 80 \
#         --is_sliding_k 4 

export CUDA_VISIBLE_DEVICES=2
python model-trainer.py \
        --current_dataset LIBRISPEECH_CLEAN \
        --train_batch_size 64 \
        --dev_batch_size 64 \
        --test_batch_size 64 \
        --max_seq_length 100 \
        --is_sliding_k 4 

# export CUDA_VISIBLE_DEVICES=3
# python model-trainer.py \
#         --current_dataset LIBRISPEECH_OTHER \
#         --train_batch_size 128 \
#         --dev_batch_size 64 \
#         --test_batch_size 64 \
        # --max_seq_length 100 \
        # --is_sliding_k 4 