'''
对错误的情况作出分类：
对每一个句子给出一个 flag
"length": 7176

"flag_length": 7171
输入为正确-纠错后为正确：r-r 4,413
输入为正确-纠错后为错误：r-w 104
输入为错误-纠错后为错误-cer未变化：w-w-s 1,547
输入为错误-纠错后为错误-cer下降： w-w-d 334
输入为错误-纠错后为错误-cer上升： w-w-u 123
输入为错误-纠错后为正确： w-r 640

'''

import json
from datasets import load_dataset, load_metric
from torch import Tensor, tensor
from torch.nn import Module
from torch.nn.parameter import Parameter
from typing import List
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR

class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp

class CustomSchedule(object):
    def __init__(self, d_model, warmup_steps=4000, optimizer=None):
        super(CustomSchedule, self).__init__()
        self.d_model = torch.tensor(d_model, dtype=torch.float32)
        self.warmup_steps = warmup_steps
        self.steps = 1.
        self.optimizer = optimizer

    def step(self):
        arg1 = self.steps ** -0.5
        arg2 = self.steps * (self.warmup_steps ** -1.5)
        self.steps += 1.
        lr = (self.d_model ** -0.5) * min(arg1, arg2)
        for p in self.optimizer.param_groups:
            p['lr'] = lr
        return lr

class EarlyStopping(object):
    def __init__(self, mode='min', min_delta=0, patience=10, percentage=False):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta, percentage)

        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False

    def step(self, metrics):
        if self.best is None:
            self.best = metrics
            return False

        if np.isnan(metrics):
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            print('terminating because of early stopping!')
            return True

        return False

    def _init_is_better(self, mode, min_delta, percentage):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if not percentage:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - min_delta
            if mode == 'max':
                self.is_better = lambda a, best: a > best + min_delta
        else:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - (
                            best * min_delta / 100)
            if mode == 'max':
                self.is_better = lambda a, best: a > best + (
                            best * min_delta / 100)


# def data_stats_1_sent_length(labels, records):
#     if len(labels) != len(records):
#         print("labels and records were wrong ")
#     else:
#         length = len(labels)
#         flag_same = 0
#         flag_large = 0  # means label longer than records
#         flag_small = 0
#         for i in range(len(labels)):
#             label = labels[i]
#             record = records[i]
#             if len(label) == len(record):
#                 flag_same = flag_same + 1
#             elif len(label) > len(record):
#                 flag_large = flag_large + 1
#             elif len(label) > len(record):
#                 flag_small = flag_small + 1
#         len_dic = {"length": length, "label_same_len_record": flag_same, "label_large_len_record": flag_large,
#                    "label_small_len_record": flag_small}
#         return len_dic


# def myfunc_savepred(preds, labels, data_name, sample_num, CER_score, path):
#     utters_list = []
#     labels_list = []
#     records_list = []
#     WERs_list = []
#     with open(
#             '/home/users/jiangjin/jiangjin_bupt/Python/data_tmp_and_process/ASR_paired_data/Aishell-1/ASR_result/test_attention_rescoring/' + data_name + '_wer',
#             'r') as f_asr, open \
#                 (
#                 path + data_name + '.jsonl',
#                 'w', encoding='utf-8') as f_data:
#         data = f_asr.readlines()
#         for i in range(len(data)):
#             line = data[i]
#             if line == '\n':
#                 continue
#             line_split = line.split(':')
#             if line_split[0] == 'utt':
#                 utters_list.append(line_split[1])
#             elif line_split[0] == 'WER':
#                 WERs_list.append(line_split[1])
#             elif line_split[0] == 'lab':
#                 labels_list.append(line_split[1])
#             elif line_split[0] == 'rec':
#                 records_list.append(line_split[1])
#             else:
#                 continue

#         if len(utters_list) == len(labels_list) == len(records_list):
#             print("three list have same length")
#         else:
#             print("Something Wrong with three lists")

#         new_label_list = []
#         new_pred_list = []
#         new_record_list = []
#         record_with_white_list = []

#         metric = load_metric("cer")
#         if len(preds) == len(labels_list[0:sample_num]):
#             for k in range(len(labels_list[0:sample_num])):
#                 utter = utters_list[k].replace('\n', '')
#                 label = labels_list[k].replace('\n', '')
#                 record_with_white = records_list[k].replace('\n', '。')
#                 record_with_white_list.append(record_with_white)
#                 record = records_list[k].replace('\n', '')
#                 pred = preds[k]
#                 Corr_label = labels[k]
#                 ASR_Sentence_wer = WERs_list[k].replace('\n', '')
#                 Corr_label = Corr_label.replace(' ', '')
#                 label = label.replace(' ', '')
#                 label = label + '。'
#                 record = record.replace(' ', '')
#                 record = record + '。'
#                 pred = pred.replace(' ', '')
#                 new_label_list.append(label)
#                 new_pred_list.append(pred)
#                 new_record_list.append(record)
#                 if Corr_label != label:
#                     print("label in Correction not correspond with ASR")
#                 else:
#                     print("saving the prediction")

#                     Corr_Sentence_WER = metric.compute(predictions=[pred], references=[label]) * 100
#                     Corr_Sentence_WER = round(Corr_Sentence_WER, 2)
#                     ASR_WER_test = metric.compute(predictions=[record], references=[label]) * 100
#                     ASR_WER_test = round(ASR_WER_test, 2)
#                     stat_flag = None #data_stats_2_stat_flag(Corr_Sentence_WER, ASR_WER_test)
#                     temp_dic = {"utt(wave_id)": utter, "refer_sent": label, "ASR_result": record, "Cor_result": pred,
#                                 "ASR_WER": ASR_Sentence_wer, "stat_flag": stat_flag,
#                                 "ASR_CER_test": str(ASR_WER_test) + "%",
#                                 "Cor_CER": str(Corr_Sentence_WER) + '%'}
#                     data_dic = {"Correction": temp_dic}
#                     f_data.write(json.dumps(data_dic, ensure_ascii=False, indent=4))
#                     f_data.write('\n')
#             len_dic = data_stats_1_sent_length(new_label_list, new_record_list)

#             f_data.write("===========================================================================")
#             f_data.write('\n')
#             Total_ASR_CER = metric.compute(predictions=new_record_list, references=new_label_list) * 100
#             Total_ASR_CER = round(Total_ASR_CER, 2)
#             Total_Cor_CER = metric.compute(predictions=new_pred_list, references=new_label_list) * 100
#             Total_Cor_CER = round(Total_Cor_CER, 2)
#             With_white_ASR_score = metric.compute(predictions=record_with_white_list, references=labels) * 100
#             With_white_ASR_score = round(With_white_ASR_score, 2)
#             temp_all_dic = {"Total_ASR_CER": Total_ASR_CER, "Total_Cor_CER": Total_Cor_CER,
#                             "With_white_ASR_score": With_white_ASR_score, "With_white_CER_score": round(CER_score*100, 2)}
#             data_all_dic = {"ALL_CER": temp_all_dic, "len_dic": len_dic}
#             f_data.write(json.dumps(data_all_dic, ensure_ascii=False, indent=4))
#             f_data.write('\n')
#             f_data.write("===========================================================================")


# def store(data):
#     with open('data.json', 'w') as fw:
#         # 将字典转化为字符串
#         # json_str = json.dumps(data)
#         # fw.write(json_str)
#         # 上面两句等同于下面这句
#         json.dump(data, fw)


# # load json data from file
# def load():
#     with open('data.json', 'r') as f:
#         data = json.load(f)
#         return data


# def get_cuda_tensors():
#     import torch
#     import gc
#     tensors = []
#     for obj in gc.get_objects():
#         try:
#             if not torch.is_tensor(obj):
#                 continue
#             assert isinstance(obj, Tensor)
#             device = str(obj.device)
#             if 'cuda' in device:
#                 tensors.append(obj)
#         except: pass
#     return tensors


# def freeze_model(model: Module):
#     if not isinstance(model, Module):
#         return

#     for parameter in model.parameters():
#         parameter.requires_grad = False

# from typing import List
# from collections import defaultdict
# class CudaTensorMonitor:
#     def __init__(self) -> None:
#         self.pre_tensor = {}

#     def update(self, named_parameters):
#         tensors = get_cuda_tensors()
#         id_tensors = { id(tensor): tensor for tensor in tensors }
#         if len(self.pre_tensor) == 0:
#             self.pre_tensor = id_tensors
#             return
        
#         new_ids = set(id_tensors.keys()) - set(self.pre_tensor.keys())
#         removed_ids = set(self.pre_tensor.keys()) - set(id_tensors.keys())
        
#         print(f"new tensors<{len(new_ids)}> removed tensors<{len(removed_ids)}>")

#         shape_names = defaultdict(list)
#         for name, parameter in named_parameters:
#             shape_string = str(parameter.shape)
#             shape_names[shape_string].append(name)
        
#         for index, new_id in enumerate(new_ids):
#             print('========================================================================')
#             print(f"new tensor<{index}/{len(new_ids)}> -> shape: {id_tensors[new_id].shape}")
#             shape_string = str(id_tensors[new_id].shape)
#             if shape_string in shape_names:
#                 print('\t matched names: ' + "--".join(shape_names[shape_string]))
        
#         self.pre_tensor = id_tensors
        
        