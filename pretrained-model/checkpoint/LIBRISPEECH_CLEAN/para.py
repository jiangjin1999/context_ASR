import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
path = "/home/users/jiangjin/jiangjin_bupt/ASR_CORRECTION/Context_Correction/context_seq2seq/pretrained-model/checkpoint/LIBRISPEECH_CLEAN/pytorch_model.bin"
paras = torch.load(path)


list_new = ['model.decoder.layers.5.knn_attn.q_proj.bias', 'model.decoder.layers.5.knn_attn.k_proj.weight', 'model.decoder.layers.5.knn_attn.v_proj.bias', 'model.decoder.layers.5.knn_attn.v_proj.weight', 'model.decoder.layers.5.knn_attn.k_proj.bias', 'model.decoder.layers.5.knn_attn.out_proj.weight', 'model.decoder.layers.5.knn_attn.q_proj.weight', 'model.decoder.layers.5.knn_attn.out_proj.bias']

list_need = [item.replace('knn_attn','self_attn') for item in list_new]

for i in range(len(list_need)):
    item_new = list_new[i]
    item_need = list_need[i]
    paras[item_new] = paras[item_need]

torch.save(paras, path)
a = 1