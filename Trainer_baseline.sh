# export CUDA_VISIBLE_DEVICES=2
# python model-trainer.py \
#         --current_dataset AISHELL-1 \
#         --train_batch_size 128 \
#         --dev_batch_size 128 \
#         --test_batch_size 128 \
#         --max_seq_length 45
export CUDA_VISIBLE_DEVICES=0
python model-trainer.py \
        --current_dataset HKUST \
        --train_batch_size 100 \
        --dev_batch_size 100 \
        --test_batch_size  100\
        --max_seq_length 80

# export CUDA_VISIBLE_DEVICES=2
# python model-trainer.py \
#         --current_dataset LIBRISPEECH_CLEAN \
#         --train_batch_size 96 \
#         --dev_batch_size 96 \
#         --test_batch_size 96 \
#         --max_seq_length 100

# export CUDA_VISIBLE_DEVICES=2
# python model-trainer.py \
#         --current_dataset LIBRISPEECH_OTHER \
#         --train_batch_size 96 \
#         --dev_batch_size 96 \
#         --test_batch_size 96 \
#         --max_seq_length 100