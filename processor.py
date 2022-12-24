"""Base Data Processors"""
from __future__ import annotations
import os
from abc import ABC
from typing import List
import json
from boto import config
from torch.utils.data import Dataset, Subset, DataLoader
import pandas as pd
from dataclasses import dataclass

from wandb import Config


@dataclass
class TextInputExample:
    """
    Input Example for a single example
    """
    utt: str = ""
    lab: str = ""
    rec: str = ""



class DataProcessor(ABC):
    """Abstract Data Processor Class which handle the different corpus"""

    def get_train_dataset(self) -> Dataset:
        """get_train_dataset
        """
        raise NotImplementedError

    def get_test_dataset(self) -> Dataset:
        raise NotImplementedError

    def get_dev_dataset(self) -> Dataset:
        raise NotImplementedError

    def get_labels(self) -> List[str]:
        pass

    def get_train_labels(self) -> List[str]:
        return self.get_labels()

    def get_test_labels(self) -> List[str]:
        return self.get_labels()

    def get_dev_labels(self) -> List[str]:
        return self.get_labels()


class TextDataProcessor(DataProcessor):
    def __init__(self, data_dir, config) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.dataset = config.current_dataset
        self.is_use_knn = config.is_use_knn
        self.batch_size = config.train_batch_size
        self.SEGMENTS = config.SEGMENTS
        self.is_shuffle_knn = config.is_shuffle_knn
        self.is_add_sos_eos = config.is_add_sos_eos
    
    def _read(self, file: str) -> List[TextInputExample]:
        with open(file, 'r', encoding='utf-8') as f:
            data = f.readlines()
            if self.is_use_knn:
                data_key = data[0]
                data = data[1:]
                # data = data[0:10]
                data_doc = []
                for item in data:
                    if item.strip() == '<d>':
                        data_doc_tmp = []
                        data_doc.append(data_doc_tmp)
                    else:
                        data_doc_tmp.append(TextInputExample(item.strip().split('\t')[0], item.strip().split('\t')[1], item.strip().split('\t')[2]) )
                # data_doc.append(data_doc_tmp)# 得到每个doc的list
                # assert len(data_doc) in [340,20,40]
                examples = self.knn_doc_process(data_doc)
                    
            else:
                if 'LIBRISPEECH' in file:
                    # print('data processor for librispeech')
                    examples = [TextInputExample(self.en_utt_process(item.strip().split('|')[0]), item.strip().split('|')[1], item.strip().split('|')[2]) for item in data]
                else:
                    examples = [TextInputExample(item.strip().split(' ')[0], item.strip().split(' ')[1], item.strip().split(' ')[2]) for item in data]
            # return example
            # examples = examples[0:150]
            return examples
    
    def knn_doc_process(self, doc_list):
        doc_num = len(doc_list)
        # 给每个doc 补全到SEGMENTS的倍数--我们设置subsequence均为 一个句子，所以，无需补全为倍数
        # white_example = TextInputExample("white_utt", "空白案例。", "空白案例。") # 如果报错，可以统一修改为。
        
        # doc_list = [doc_item+[white_example for _ in range(self.SEGMENTS-len(doc_item)%self.SEGMENTS)] for doc_item in doc_list]

        # 把所有的doc 按照一种策略-总是给当前最短的那一个，分配给batch size个 list
        data_list = [[]for _ in range(self.batch_size)]
        data_list_length = [0 for _ in range(self.batch_size)]
        for doc_item in doc_list:
            data_list, data_list_length = self.write2min(doc_item, data_list, data_list_length)
            
        # 把数据按照 SEGMENTS 进行合并
        # data_list = [[self.MergeSEGMENTS(doc_item) for doc_item in batch_item] for batch_item in data_list]
        # for batch_item in data_list:
        #     for doc_item in batch_item:
        #         doc_item = self.MergeSEGMENTS(doc_item)
        
        # 把每个batch 维度中的doc 加总在一起 ==这里后续可以加入sos 和 eos label 来清理 knn memory
        data_list = [self.mergelist(data_item) for data_item in data_list]
        # 将所有的doc 补充到相同的大小，用空的example
        data_list = self.completion2max(data_list, data_list_length)
        
        # 根据上述的process，重新排序所有example
        examples = []
        for i in range(len(data_list[0])):
            for j in range(self.batch_size):
                examples.append(data_list[j][i])
        return examples
        
    
    def write2min(self, doc_item, data_list, data_list_length):
        min_index = data_list_length.index(min(data_list_length))
        data_list[min_index].append(doc_item)
        data_list_length[min_index] = data_list_length[min_index] + len(doc_item)
        
        return data_list, data_list_length
    
    def MergeSEGMENTS(self, doc_item):
        assert len(doc_item)%self.SEGMENTS==0
        doc_item_new = []
        for i in range(0,len(doc_item), self.SEGMENTS):
            utt: str = ""
            lab: str = ""
            rec: str = ""
            for j in range(self.SEGMENTS):
                utt = utt + doc_item[i+j].utt
                lab = lab + doc_item[i+j].lab
                rec = rec + doc_item[i+j].rec
            tmp = TextInputExample(utt, lab, rec)
            doc_item_new.append(tmp)
        return doc_item_new
    
    def mergelist(self, doc_item_list):
        doc_item_list_output = []
        if self.is_add_sos_eos:
            for item in doc_item_list:
                doc_item_list_output = doc_item_list_output + item
        else:
            for item in doc_item_list:
                doc_item_list_output = doc_item_list_output + item
    
        return doc_item_list_output

    def completion2max(self, data_list, data_list_length):
        white_example = TextInputExample("white_utt", "空白案例。", "空白案例。") # 如果报错，可以统一修改为。
        max_length = max(data_list_length)
        min_length = min(data_list_length)
        data_list = [data_doc_item + [white_example for _ in range(max_length-len(data_doc_item))] for data_doc_item in data_list]
        return data_list
            
    def en_utt_process(self, item):
        length = len(item.split('-')[2])
        output = ''
        if length == 1:
            output = item.split('-')[0] + '-' + item.split('-')[1] + '-000' +item.split('-')[2]
        elif length == 2:
            output = item.split('-')[0] + '-' + item.split('-')[1] + '-00' +item.split('-')[2]
        elif length == 3:
            output = item.split('-')[0] + '-' + item.split('-')[1] + '-0' +item.split('-')[2]
        else:
            output = item.split('-')[0] + '-' + item.split('-')[1] + '-' +item.split('-')[2]
        return output
    
    def _load_dataset(self, mode: str = 'train.txt') -> Dataset:
        file = os.path.join(self.data_dir, mode)
        examples = self._read(file)
        indices = [i for i in range(len(examples))] 
        return Subset(examples, indices) 

    def get_train_dataset(self, batch_size) -> Dataset:
        self.batch_size =batch_size
        if self.is_use_knn:
            return self._load_dataset(self.dataset+'/'+self.dataset+'_train_doc.txt')
        else:
            return self._load_dataset(self.dataset+'/'+self.dataset+'_train.txt')
        
    def get_dev_dataset(self, batch_size) -> Dataset:
        self.batch_size =batch_size
        if self.is_use_knn:
            return self._load_dataset(self.dataset+'/'+self.dataset+'_dev_doc.txt')
        else:
            return self._load_dataset(self.dataset+'/'+self.dataset+'_dev.txt')
        
    def get_test_dataset(self, batch_size) -> Dataset:
        self.batch_size =batch_size
        if self.is_use_knn:
            return self._load_dataset(self.dataset+'/'+self.dataset+'_test_doc.txt')
        else:
            return self._load_dataset(self.dataset+'/'+self.dataset+'_test.txt')
        # return self._load_dataset(self.dataset+'/'+self.dataset+'_test-other.txt')



    