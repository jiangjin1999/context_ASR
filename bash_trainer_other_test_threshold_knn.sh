# Other test mode:
export CUDA_VISIBLE_DEVICES=2
python model-trainer.py \
        --current_dataset AISHELL-1 \
        --train_batch_size 80 \
        --dev_batch_size 40 \
        --test_batch_size 20 \
        --is_use_knn \
        --is_add_sos_eos \
        --is_use_threshold \
        --max_seq_length 40
# export CUDA_VISIBLE_DEVICES=2
# python model-trainer.py \
#         --current_dataset HKUST \
#         --train_batch_size 50 \
#         --dev_batch_size 22 \
#         --test_batch_size 24 \
#         --is_use_knn \
#         --is_add_sos_eos \
#         --is_use_threshold \
#         --max_seq_length 80