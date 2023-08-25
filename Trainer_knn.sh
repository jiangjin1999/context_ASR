# export CUDA_VISIBLE_DEVICES=3
# python model-trainer.py \
#         --current_dataset AISHELL-1 \
#         --train_batch_size 128 \
#         --dev_batch_size 40 \
#         --test_batch_size 20 \
#         --_num_retrieved_memories_K 64 \
#         --is_use_knn \
#         --is_add_sos_eos \
#         --max_seq_length 40
export CUDA_VISIBLE_DEVICES=3
python model-trainer.py \
        --current_dataset HKUST \
        --train_batch_size 100 \
        --dev_batch_size 22 \
        --test_batch_size 24 \
        --_num_retrieved_memories_K 256 \
        --is_use_knn \
        --is_add_sos_eos \
        --max_seq_length 80
# export CUDA_VISIBLE_DEVICES=3
# python model-trainer.py \
#         --current_dataset LIBRISPEECH_CLEAN \
#         --train_batch_size 128 \
#         --dev_batch_size 87 \
#         --test_batch_size 97 \
#         --_num_retrieved_memories_K 64\
#         --is_use_knn \
#         --is_add_sos_eos \
#         --max_seq_length 100
# export CUDA_VISIBLE_DEVICES=3
# python model-trainer.py \
#         --current_dataset LIBRISPEECH_OTHER \
#         --train_batch_size 96 \
#         --dev_batch_size 91 \
#         --test_batch_size 90 \
#         --is_use_knn \
#         --is_add_sos_eos \
#         --max_seq_length 100

