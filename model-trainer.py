import os
import random
from typing import Dict, List, Optional, Tuple 
import numpy as np
import torch
import time

import re
from datasets import Metric, load_metric
import evaluate
from genericpath import exists
from loguru import logger
# from sqlalchemy import false
# from sympy import true
from tap import Tap
from torch import nn, tensor
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from transformers import (AdamW, AutoConfig, AutoModelForSeq2SeqLM,BertTokenizer,BartTokenizer,
                          AutoTokenizer, BertConfig,BartConfig, PreTrainedModel,BartForConditionalGeneration,
                          PreTrainedTokenizer, get_scheduler, set_seed)
from transformers.modeling_outputs import Seq2SeqLMOutput
from transformers.models import bart

from processor import DataProcessor, TextDataProcessor, TextInputExample
from utils import  EarlyStopping

from model.modeling_bart import (BartForContextCorretion, BartModel)

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class Config(Tap):


    seed: int = 2023
    pwd: str = '/home/jiangjin/context/'

    # KNN Code  
    # batch_size 的设定，如果原有baseline batch为100，那train为100，test dev doc的个数若小于100，则为为doc个数
    train_batch_size: int = 64
    dev_batch_size: int = 40
    test_batch_size: int = 20
    
    current_dataset: str = 'AISHELL-1' #'LIBRISPEECH_OTHER' #'LIBRISPEECH_CLEAN'
    is_use_knn: bool = False
    is_from_ckpt: bool = False
    is_shuffle_knn: bool = False
    is_offline: bool = False
    is_domain_datastore: bool = False
    max_seq_length: int = 40 # 一个句子的max length 是
    is_add_sos_eos: bool = False
    
    is_knn_gpu: bool = False
    
    is_sliding_k: int = 0
    
    gate_parameter: float = 0.5
    _num_retrieved_memories_K: int = 32
    knn_memorizing_layers: int = 5
    max_memories: int = 65000 # datastore 的最大值。
    
    num_beams: int = 4
    
    is_random_vector: bool = False
    _knn_dis_threshold: float = 0
    
 
    language: str = 'en'
    is_zh: bool = False
    metric: str = 'wer'
 
    mode: str = 'train'    
    is_pretrained: bool = True
    model_type: str = '' #'nopretrained-' # default
    shuffle: bool = False
    

    SEGMENTS: int = 1 #一个subsequence包含几个句子

    knn_memories_directory: str = ''
    datastore_path: str = ''
    mode_mode_path: str = pwd + model_type
    mode_mode_path_dataset: str = mode_mode_path + '/' + current_dataset
    best_model_dir: str = mode_mode_path_dataset + '/model-checkpoint/'
    test_result_dir: str = mode_mode_path_dataset + '/result/'
    log_path: str =mode_mode_path_dataset + '/log/'
    tensorboard_path: str =mode_mode_path_dataset + '/tensorboard/' 
    text_data_dir: str = pwd +'data/'+ language 
    pretrained_model: str = pwd + 'pretrained-model/'+language+'/BART'
    Model_config = AutoConfig.from_pretrained(pretrained_model)
    # 模型相关 参数配置
    learning_rate: float = 5e-5
    weight_decay: float = 0.02
    lr_scheduler_type: str = 'linear'
    num_warmup_steps: int = 200
    max_train_steps: int = 2000
    epochs: int = 30

    early_stop = EarlyStopping(patience=3)
    early_stop_flag: str = False
    device: str = 'cuda'
    
    def get_device(self):
        """return the device"""
        return torch.device(self.device)


class ContextContainer:
    """Context data container for training
    """

    def __init__(self) -> None:
        """init the variables"""
        self.train_step: int = 0
        self.dev_step: int = 0
        self.epoch: int = 0

        self.train_cer: float = 1000
        self.dev_cer: float = 1000
        self.best_dev_cer: float = 1000
        self.test_cer: float = 1000

        self.lr:float = 0

        self.loss = 0
        self.dev_loss = 0
        self.output_loss = 0
        self.logits = 0
        self.labels = 0



class Trainer:
    """Trainer which can handle the train/eval/test/predict stage of the model
    """

    def __init__(
        self, config: Config,
        text_processor: DataProcessor,
        text_tokenizer: AutoTokenizer,
        model: AutoModelForSeq2SeqLM,
        metric: Metric,
    ) -> None:
        
        self.config = config
        self.text_tokenizer = text_tokenizer
        self.metric = metric
        self.is_use_knn = self.config.is_use_knn
        
        embedding_size = model.get_input_embeddings().weight.shape[0]
        if len(text_tokenizer) > embedding_size:
            model.resize_token_embeddings(len(text_tokenizer))
        
        model.resize_token_embeddings(len(text_tokenizer))
        self.model = model.to(self.config.get_device())

        # 2. build text dataloader
        logger.info('init text  dataloaders ...')

        self.train_dataloader = self.create_dataloader(
            dataset=text_processor.get_train_dataset(self.config.train_batch_size)[0],
            batch_size=self.config.train_batch_size,
            shuffle=self.config.shuffle,
            collate_fn=self.convert_examples_to_features,
        )
        
        self.dev_dataloader = self.create_dataloader(
            dataset=text_processor.get_dev_dataset(self.config.dev_batch_size)[0],
            batch_size=self.config.dev_batch_size,
            shuffle=False, #self.config.shuffle,
            collate_fn=self.convert_examples_to_features,
        )
        self.test_dataloader = self.create_dataloader(
            dataset=text_processor.get_test_dataset(self.config.test_batch_size)[0],
            batch_size=self.config.test_batch_size,
            shuffle=False, #self.config.shuffle,
            collate_fn=self.convert_examples_to_features,
        )
        if self.config.is_offline is True:
            self.train_length_per_doc_num = text_processor.get_train_dataset(self.config.train_batch_size)[1]
            self.dev_length_per_doc_num = text_processor.get_dev_dataset(self.config.dev_batch_size)[1]
            self.test_length_per_doc_num = text_processor.get_test_dataset(self.config.test_batch_size)[1]
            self.train_length_per_doc_flag = [[0,len(self.train_length_per_doc_num[i])] for i in range(len(self.train_length_per_doc_num))]
            self.dev_length_per_doc_flag = [[0,len(self.dev_length_per_doc_num[0])] for i in range(len(self.dev_length_per_doc_num))]
            self.test_length_per_doc_flag = [[0,len(self.test_length_per_doc_num[0])] for i in range(len(self.test_length_per_doc_num))]
            self.train_length_per_doc = (self.train_length_per_doc_num, self.train_length_per_doc_flag)
            self.dev_length_per_doc = (self.dev_length_per_doc_num, self.dev_length_per_doc_flag)
            self.test_length_per_doc = (self.test_length_per_doc_num, self.test_length_per_doc_flag)
        
        # 3. init model related
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": config.weight_decay,
                },
                {
                    "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
            ]

        self.optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=config.learning_rate)
        
        # 最大的step就是 train_dataloader 的 长度 乘上 epochs的长度。
        self.config.max_train_steps = len(self.train_dataloader) * 10
        self.config.num_warmup_steps = len(self.train_dataloader) # 第一个epoch 进行 warmup
        self.lr_scheduler = get_scheduler(
            name=config.lr_scheduler_type,
            optimizer=self.optimizer,
            num_warmup_steps=self.config.num_warmup_steps, # 前 * step 进行warm up（即让lr 从0-设定的lr）
            num_training_steps=self.config.max_train_steps, # 最大的step
        )
        self.context_data = ContextContainer()
        self._init_output_dir()
        self.writer: SummaryWriter = SummaryWriter(self.config.tensorboard_path)
        self.train_bar: tqdm = None


    def create_dataloader(self, dataset: Dataset, batch_size, collate_fn, shuffle) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle, 
            collate_fn=collate_fn,
        )

    def convert_examples_to_features(self, examples: List[TextInputExample]):
        """convert the examples to features"""
        # for en: inputs_ids starts from 0(<s>)  and end with 2(</s>) 
        texts = [example.rec for example in examples]
        encoded_features = self.text_tokenizer.batch_encode_plus(
            texts,
            max_length=self.config.max_seq_length,
            padding='max_length',
            return_tensors='pt',
            truncation=True
        )

        labels = [example.lab for example in examples]
        label_features = self.text_tokenizer.batch_encode_plus(
            labels,
            max_length=self.config.max_seq_length,
            padding='max_length',
            return_tensors='pt',
            truncation=True
        )
        # breakpoint()
        return encoded_features['input_ids'], label_features['input_ids']

    def _init_output_dir(self):
        logger.info(f'init the output dir: {self.config.log_path}')
        if os.path.exists(self.config.log_path):
            pass
        else:
            os.makedirs(self.config.log_path)

    def _update_train_bar(self):
        infos = [f'epoch: {self.context_data.epoch}/{self.config.epochs}']

        loss = self.context_data.loss
        if torch.is_tensor(loss):
            loss = loss.detach().clone().cpu().numpy().item()
        infos.append(f'loss: <{loss}>')

        self.train_bar.update()
        self.train_bar.set_description(str(infos))

    def on_batch_start(self):
        '''handle the on batch start logits
        '''
        self.model.train()
        self.context_data.lr = self.optimizer.param_groups[0]['lr']

    def on_batch_end(self):
        """handle the on batch training is ending logits
        """
        # 1. update global step
        self.context_data.train_step += 1
        # print(self.context_data.train_step)
        self._update_train_bar()
        # self.train_bar.update()

        self.writer.add_scalar(
            'train/text-loss',
            scalar_value=self.context_data.loss,
            global_step=self.context_data.train_step,
        )
        self.writer.add_scalar(
            'train/learning-rate',
            scalar_value=self.context_data.lr,
            global_step=self.context_data.train_step,
        )
        
    def train(self):
        """the main train epoch"""
        
        if os.path.exists(self.config.best_model_dir):
            logger.info('+++++++++++++++++++++++++++++++++++++++++++++++++++++++')
            logger.info(f"Resumed from checkpoint: {self.config.best_model_dir}")
            logger.info('+++++++++++++++++++++++++++++++++++++++++++++++++++++++')
            self.model.load_state_dict(torch.load(self.config.best_model_dir+'checkpoint_best.pt'), strict=True)
        else:
            logger.info('start training ...')
            logger.info(f'  num example = {len(self.train_dataloader)}')
            logger.info(f'  num epochs = {self.config.epochs}')
            logger.info(f'  total optimization step = {self.config.max_train_steps}')

        self.on_train_start()
        for _ in range(self.config.epochs): 
            # breakpoint()           
            self.context_data.epoch += 1
            # add time the train epoch
            start_time_train = time.time()
            self.train_epoch()
            end_time_train = time.time()
            # log the time of train epoch
            logger.info('\n' )
            # logger.info(f'train epoch {self.context_data.epoch} time is {end_time_train-start_time_train}s')
            logger.info(f'train epoch {self.context_data.epoch} time is {round((end_time_train-start_time_train)/60,2)}min')

            self.on_epoch_end()
            if self.config.early_stop_flag:
                logger.info('\n -----------------------------')
                logger.info('early stopping on train epoch')
                logger.info('-----------------------------\n')
                break
    def on_train_start(self):
        '''inite the dev and test cer'''
        # self.context_data.dev_cer = self.evaluate(self.dev_dataloader)
        # self.context_data.test_cer = self.evaluate(self.test_dataloader)
        # self.writer.add_scalar(
        #     tag='dev/cer',
        #     # scalar_value=self.context_data.dev_cer,
        #     scalar_value=0.2701,
        #     global_step=self.context_data.dev_step
        # )
        # self.writer.add_scalar(
        #     tag='test/cer',
        #     # scalar_value=self.context_data.test_cer,
        #     scalar_value=0.2431,
        #     global_step=self.context_data.dev_step
        # )
        
    def offline_datastore(self, x, knn_memories, length_per_doc, mode='train'):
        # breakpoint()
        if self.config.is_domain_datastore is True:
            knn_memories.read_offline_db([0],db_index_list=[[0,self.config.max_memories//self.config.max_seq_length]]) # 当前模式，直接将db存进去。
            return knn_memories
        else:
            pass
        
        
        if self.config.language == 'zh':
            clear_memories_on_sos_token_id = 108
        else:
            clear_memories_on_sos_token_id = 1629
        token_id = clear_memories_on_sos_token_id
        clear_memory = (x == token_id).any(dim = -1)
        batch_indices = clear_memory.nonzero(as_tuple = True)
        batch_indices = batch_indices[0] 
        batch_indices_to_clear = batch_indices.tolist()
        if len(batch_indices_to_clear) == 0: # batch_indices_to_clear: 50 
            return knn_memories
        
        logger.info('\n')
        logger.info('========offline_datastore_read====================')
        knn_memories.clear_memory(batch_indices_to_clear)
        length_per_doc_num = length_per_doc[0] 
        length_per_doc_flag = length_per_doc[1] 
        assert length_per_doc_flag[0][0] <= length_per_doc_flag[0][1]
        
        batch_indices_to_read_db = batch_indices_to_clear
        
        db_index_list =[[0,0] for i in range(len(length_per_doc[0]))]
        
        for index in batch_indices_to_read_db:
            length_per_doc_flag[index][0] = length_per_doc_flag[index][0] + 1
            assert length_per_doc_flag[index][0] > 0
            start_index = sum(length_per_doc_num[index][:length_per_doc_flag[index][0]-1])
            end_index = sum(length_per_doc_num[index][:length_per_doc_flag[index][0]])

            db_index_list[index][0] = start_index
            db_index_list[index][1] = end_index
            
        knn_memories.read_offline_db(batch_indices_to_read_db,db_index_list=db_index_list)
        return knn_memories
        
        

    def train_epoch(self):
        """handle the logit of training epoch

        Args:
            epoch (int): _description_
        # """
        logger.info('\n')
        logger.info('=======================================')
        logger.info(f'training epoch<{self.context_data.epoch}> ...')
        self.train_bar = tqdm(total=len(self.train_dataloader))
        self.config.max_memories = len(self.train_dataloader) * self.config.max_seq_length
        
        knn_start_time = time.time()
        # if self.config.is_offline is True:
        #     self.config.knn_memories_directory = self.config.datastore_path
        # length_per_doc 的 flag 在每一个epoch 都需要重置。
        if self.config.is_offline is True:
            for i in range(len(self.train_length_per_doc[1])):
                self.train_length_per_doc[1][i][0] = 0
        with self.model.knn_memories_context(my_config=self.config, batch_size = self.config.train_batch_size, mode='train', knn_memories_directory=self.config.knn_memories_directory,max_memories=self.config.max_memories, is_knn_gpu = self.config.is_knn_gpu) as knn_memories: 
            knn_end_time = time.time()
            logger.info(f'Build KNN Memory and Moving index to GPU took {round((knn_end_time - knn_start_time)/60,4)} min')
            for text_batch in self.train_dataloader:
                # breakpoint()
                
                self.on_batch_start()
                
                input_ids, labels = text_batch
                attention_mask = (input_ids != self.text_tokenizer.pad_token_id).long()
                
                input_ids, labels, attention_mask = input_ids.to(
                    self.config.get_device()), labels.to(self.config.get_device()), \
                    attention_mask.to(self.config.get_device())
                    
                    
                # set the offline datastore and knn
                if self.config.is_offline is True:
                    knn_memories = self.offline_datastore(input_ids, knn_memories, self.train_length_per_doc, mode='train')

                self.optimizer.zero_grad()    

                # forward on text data
                output: Seq2SeqLMOutput = self.model(
                    input_ids=input_ids, 
                    attention_mask=attention_mask,
                    labels=labels,
                    decoder_knn_memories = knn_memories,
                    decoder_TrainerConfig  = self.config,
                    )

                self.context_data.loss = output.loss.sum().detach().cpu().numpy().item()
                self.context_data.output_loss = output.loss
                self.context_data.output_loss.backward()
                
                self.optimizer.step() 
                self.lr_scheduler.step()  
                self.optimizer.zero_grad() 
                    
                if self.config.early_stop_flag:
                    logger.info('early stopping')
                    break
                
                self.on_batch_end()
                
    def on_epoch_end(self):
        start_time_dev = time.time()
        self.context_data.dev_cer = self.evaluate(self.dev_dataloader)
        end_time_dev = time.time()
        # logger.info(f'dev epoch {self.context_data.epoch} time is {end_time_dev-start_time_dev}')
        self.config.early_stop_flag = self.config.early_stop.step(self.context_data.dev_cer)
        logger.info('\n')
        logger.info(f'dev/cer is {self.context_data.dev_cer}')
        self.writer.add_scalar(
            tag='dev/cer',
            scalar_value=self.context_data.dev_cer,
            global_step=self.context_data.train_step
        )
        self.on_evaluation_end(self.context_data.dev_cer)
        
    
    def evaluate(self, dataloader,):
        """handle the logit of evaluating

        Args:
            epoch (int): _description_
        """
        logger.info('\n')
        # logger.info('=======================================')
        logger.info(f'evaluating epoch<{self.context_data.epoch}> ...')
        self.model.eval()

        all_decoded_preds = []
        all_decoded_labels = []
        
        dev_pbar = tqdm(total=len(dataloader), desc='evaluating stage ...', leave=False)
        evalate_times = []
        
        knn_start_time = time.time()
        # torch.cuda.empty_cache()
        if self.is_use_knn:
            cut = int(self.config.dev_batch_size * 0.5)
        else:
            cut = self.config.dev_batch_size
        # breakpoint()
        # self.config.dev_batch_size = cut
        self.config.max_memories = len(dataloader) * self.config.max_seq_length  #* self.config.num_beams
        if self.config.is_offline is True:
            for i in range(len(self.dev_length_per_doc[1])):
                self.dev_length_per_doc[1][i][0] = 0
        # beam size 下的datastore 需要self.config.num_beams,倍的空间。
        with self.model.knn_memories_context(my_config=self.config, batch_size = cut, mode='dev', knn_memories_directory=self.config.knn_memories_directory,max_memories=self.config.max_memories,is_knn_gpu = self.config.is_knn_gpu) as knn_memories: 
            knn_end_time = time.time()
            logger.info(f'Build KNN Memory and Moving index to GPU took {round((knn_end_time - knn_start_time)/60,4)} min')
            for text_batch in dataloader:
                with torch.no_grad():
                    input_ids, labels = text_batch
                    input_ids, labels = input_ids[:cut,], labels[:cut,]
                    input_ids, labels = input_ids.to(
                        self.config.get_device()), labels.to(self.config.get_device())

                    if self.config.is_offline is True:
                        knn_memories = self.offline_datastore(input_ids, knn_memories, self.dev_length_per_doc, mode='dev')


                    # Start the timer
                    start_time = time.time()
                    
                    max_token = input_ids.shape[1]
                    generated_tokens: Seq2SeqLMOutput = self.model.generate(
                        input_ids=input_ids, 
                        max_length=max_token, 
                        # num_beams = self.config.num_beams,
                        decoder_knn_memories = knn_memories,
                        decoder_TrainerConfig  = self.config,
                        )
                    
                    # End the timer and compute the inference time
                    end_time = time.time()
                    inference_time = end_time - start_time
                    per_sample_inference_time = inference_time / self.config.test_batch_size
                    
                     # 在tqdm的bar上显示推理时间
                    evalate_times.append(inference_time)
                    dev_pbar.update(1)
                    info = f"dev time: {round(inference_time/60,2)}min ||per sample: {round(per_sample_inference_time/60,2)}min"
                    dev_pbar.set_description(str(info))
                    
                    generated_tokens = generated_tokens.detach().cpu().numpy()
                    labels = labels.detach().cpu().numpy() 

                    decoded_preds = self.text_tokenizer.batch_decode(
                        generated_tokens, skip_special_tokens=True)
                    decoded_labels = self.text_tokenizer.batch_decode(
                        labels, skip_special_tokens=True)
                    if self.config.language == 'en':
                        decoded_preds = [decoded_pred for decoded_pred in decoded_preds]
                        decoded_labels = [decoded_label for decoded_label in decoded_labels]
                    else:
                        decoded_preds = [decoded_pred.replace(' ','') for decoded_pred in decoded_preds]
                        decoded_labels = [decoded_label.replace(' ','') for decoded_label in decoded_labels]
                        
                    
                    all_decoded_preds = all_decoded_preds + decoded_preds
                    all_decoded_labels = all_decoded_labels + decoded_labels
        
        final_inference_times = sum(evalate_times)
        logger.info('\n')
        # logger.info(f'dev_times is {final_inference_times}s')
        # logger info minutes
        logger.info(f'dev_times pre sample is {round(final_inference_times/len(dataloader)*self.config.dev_batch_size, 2)}s')
        logger.info(f'dev_times is {round(final_inference_times/60, 2)}min')

        all_decoded_pred, all_decoded_label = [], []
        for i in range(len(all_decoded_labels)):
            item_label = all_decoded_labels[i]
            item_pred = all_decoded_preds[i]
            if item_label != '空白案例。':
                all_decoded_pred.append(item_pred)
                all_decoded_label.append(item_label)
            else:
                pass
        all_decoded_labels = all_decoded_label
        all_decoded_preds = all_decoded_pred
        assert len(all_decoded_labels)==len(all_decoded_preds)
        assert len(all_decoded_labels)==len(all_decoded_preds)
        metric_score = self.metric.compute(
            predictions=all_decoded_preds, references=all_decoded_labels)

        self.model.train()
        return metric_score

    def on_evaluation_end(self, metric_score):
        '''always save the best model'''
        if self.context_data.best_dev_cer > metric_score:
            self.save_model(self.config.best_model_dir)
            self.context_data.best_dev_cer = metric_score
            logger.info('\n')
            logger.info(f'dev/best_cer is {self.context_data.dev_cer}')
            self.writer.add_scalar(
                tag='dev/best_cer',
                scalar_value=self.context_data.best_dev_cer,
                global_step=self.context_data.train_step
            )
            start_time_test = time.time()
            # self.context_data.test_cer = self.predict('test')
            end_time_test = time.time()
            logger.info(f'test epoch {self.context_data.epoch} time is {end_time_test-start_time_test}')
            self.writer.add_scalar(
                tag='test/cer',
                scalar_value=self.context_data.test_cer,
                global_step=self.context_data.train_step
            )

    def save_model(self, path):
        if os.path.exists(path):
            pass
        else:
            os.makedirs(path)
        torch.save(self.model.state_dict(), path+'/checkpoint_best.pt')


    def predict(self, FLAG: Optional[str] = None,):
        """ predict the example
        """
        dataloader = self.test_dataloader
        logger.info('\n')
        logger.info('start predicting ...')
        if FLAG is not None:
            pass
        else:
            self.load_model(self.config.best_model_dir + 'checkpoint_best.pt')

        self.model.eval()
        # add tqdm for predict 
        
        
        all_decoded_inputs = []

        all_decoded_preds = []
        all_decoded_labels = []

        test_pbar = tqdm(total=len(dataloader), desc='predicting stage ...', leave=False)
        # Initialize a list to store the inference times
        inference_times = []
        # self.config.max_memories

        knn_start_time = time.time()
        self.config.max_memories = len(dataloader) * self.config.max_seq_length #* self.config.num_beams
        if self.config.is_offline is True:
            for i in range(len(self.test_length_per_doc[1])):
                self.test_length_per_doc[1][i][0] = 0
        with self.model.knn_memories_context(my_config=self.config, batch_size = self.config.test_batch_size, mode='test', knn_memories_directory=self.config.knn_memories_directory,max_memories=self.config.max_memories,is_knn_gpu = self.config.is_knn_gpu) as knn_memories: 
            knn_end_time = time.time()
            logger.info(f'Build KNN Memory and Moving index to GPU took {round((knn_end_time - knn_start_time)/60,4)} min')
            for text_batch in dataloader:
                with torch.no_grad():
                    input_ids, labels = text_batch
                    input_ids, labels = input_ids.to(
                        self.config.get_device()), labels.to(self.config.get_device())
                    
                    if self.config.is_offline is True:
                        knn_memories = self.offline_datastore(input_ids, knn_memories, self.test_length_per_doc, mode='test')

                    # Start the timer
                    start_time = time.time()
                    
                    max_token = input_ids.shape[1]
                    generated_tokens: Seq2SeqLMOutput = self.model.generate(
                        input_ids=input_ids, 
                        max_length=max_token, 
                        # num_beams = self.config.num_beams,
                        decoder_knn_memories = knn_memories,
                        decoder_TrainerConfig  = self.config,
                        )
                    # End the timer and compute the inference time
                    end_time = time.time()
                    inference_time = end_time - start_time
                    per_sample_inference_time = inference_time / self.config.test_batch_size
                    
                     # 在tqdm的bar上显示推理时间
                    inference_times.append(inference_time)
                    test_pbar.update(1)
                    info = f"Inference time: {round(inference_time/60,2)}min ||per sample: {round(per_sample_inference_time/60,2)}min"
                    test_pbar.set_description(str(info))
                    
                    generated_tokens = generated_tokens.detach().cpu().numpy()
                    labels = labels.detach().cpu().numpy() 
             
                    # 是否再算一遍raw
                    if FLAG is not None:
                        pass
                    else:
                        decoded_inputs = self.text_tokenizer.batch_decode(
                            input_ids, skip_special_tokens=True)
                    
                    decoded_preds = self.text_tokenizer.batch_decode(
                        generated_tokens, skip_special_tokens=True)
                    decoded_labels = self.text_tokenizer.batch_decode(
                        labels, skip_special_tokens=True)

                    # 中文英文数据格式不同的处理
                    if self.config.language == 'en':
                        if FLAG is not None:
                            pass
                        else:
                            decoded_inputs = [decoded_input for decoded_input in decoded_inputs]
                        decoded_preds = [decoded_pred for decoded_pred in decoded_preds]
                        decoded_labels = [decoded_label for decoded_label in decoded_labels]
                    else:
                        if FLAG is not None:
                            pass
                        else:
                            
                            decoded_inputs = [decoded_input.replace(' ','') for decoded_input in decoded_inputs]
                        decoded_preds = [decoded_pred.replace(' ','') for decoded_pred in decoded_preds]
                        
                        decoded_labels = [decoded_label.replace(' ','') for decoded_label in decoded_labels]
                        
                        
                    if FLAG is not None:
                        pass
                    else:
                        all_decoded_inputs = all_decoded_inputs + decoded_inputs
                    
                    all_decoded_preds = all_decoded_preds + decoded_preds
                    all_decoded_labels = all_decoded_labels + decoded_labels
        final_inference_times = sum(inference_times)
        # logger.info(f'test_times is {final_inference_times}s')
        # logger info minutes
        logger.info(f'test_times pre sample is {round(final_inference_times/len(dataloader)*self.config.test_batch_size, 2)}s')
        logger.info(f'test_times is {round(final_inference_times/60,2)}min')
        
        all_decoded_inputs = [item for item_label, item in zip(all_decoded_labels, all_decoded_inputs) if item_label!='空白案例。']
        all_decoded_pred, all_decoded_label = [], []
        for i in range(len(all_decoded_labels)):
            item_label = all_decoded_labels[i]
            item_pred = all_decoded_preds[i]
            if item_label != '空白案例。':
                all_decoded_pred.append(item_pred)
                all_decoded_label.append(item_label)
            else:
                pass
        all_decoded_labels = all_decoded_label
        all_decoded_preds = all_decoded_pred
        assert len(all_decoded_labels)==len(all_decoded_preds)
        
        if FLAG is not None:
            pass
        else:
            raw_score = self.metric.compute(
                predictions=all_decoded_inputs, references=all_decoded_labels)

        metric_score = self.metric.compute(
            predictions=all_decoded_preds, references=all_decoded_labels)

        self.save_test_result(all_decoded_preds, all_decoded_labels, self.config.current_dataset)

        self.context_data.test_cer = metric_score
        # self.writer.add_scalar(
        #     tag='test/'+self.config.current_dataset+'_cer',
        #     scalar_value=self.context_data.test_cer,
        #     global_step=self.context_data.dev_step
        # )
        if FLAG is not None:
            pass
        else:
            logger.info('\n')
            # logger.info(f'raw/cer is {raw_score}')
        logger.info('\n')
        logger.info(f'test/cer is {self.context_data.test_cer}')
        # add test cer every time evaluate test data
        self.model.train()
        return metric_score


    def load_model(self, path):
        logger.info('load model ...')
        self.model.load_state_dict(torch.load(path))

    def save_test_result(self, all_decoded_preds, all_decoded_labels, test_data_name):
        # for text_modal: add additional 'text_modal_' to distinguish
        # ['cross_modal', 'text_modal']
        if os.path.exists(self.config.test_result_dir):
            pass
        else:
            os.makedirs(self.config.test_result_dir) 
        with open(config.test_result_dir+'T_modal_'+test_data_name+'.txt', 'w') as f_result:
            data_output_list = []
            for item_pred, item_label in zip(all_decoded_preds, all_decoded_labels):
                data_output_list.append(item_pred + '\t' + item_label + '\n') 
            f_result.writelines(data_output_list)


def set_my_seed(seed):
    '''random:
        python
        Numpy'''
    # set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    # torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # import tensorflow as tf
    # tf.random.set_seed(seed)
    
    # from torch.backends import cudnn
    # cudnn.benchmark = False
    # cudnn.deterministic = True
    
def reset_config_parse(config):
    # 讲 config 中所有 由 if修改的变量 和 由其他变量定义的变量 在该函数内重新定义。
    # if config.is_pretrained is True:
        # config.model_type = 'pretrained-'
    # if config.is_random_vector:
    #     config.model_type = config.model_type + 'Other-test/random-vector/'
    #     config.knn_memories_directory = '.tmp/knn.ckpt.memories' + 'test-random-vector'
    if config.is_use_knn:
        if config.is_from_ckpt:
            if config.is_add_sos_eos is False:
                config.model_type = config.model_type + 'T-model-knn-ckpt-domain-context'
            else:
                config.model_type = config.model_type + 'T-model-knn-ckpt' 
            # config.knn_memories_directory = '.tmp/knn.ckpt.memories'
        else:
            if config.is_add_sos_eos is False:
                config.model_type = config.model_type + 'T-model-knn-domain-context'
            else:
                config.model_type = config.model_type + 'T-model-knn'
            # config.knn_memories_directory = '.tmp/knn.memories' 
        if config.is_shuffle_knn:
            config.model_type = config.model_type + '-shuffle'
            # config.knn_memories_directory = config.knn_memories_directory + '.shuffle'
        # config.knn_memories_directory = config.knn_memories_directory + '/'
    else:
        if config.is_sliding_k != 0:
            config.model_type = config.model_type + 'T-model-baseline-sliding-'+str(config.is_sliding_k)
            config.max_seq_length = config.max_seq_length * config.is_sliding_k
            config.text_data_dir: str = config.pwd +'data/'+ 'sliding-k/sliding-'+str(config.is_sliding_k)+'/' 
            config.train_batch_size = config.train_batch_size // config.is_sliding_k
            config.dev_batch_size = config.dev_batch_size // config.is_sliding_k
            config.test_batch_size = config.test_batch_size // config.is_sliding_k
        else:
            config.model_type = config.model_type + 'T-model-baseline'
        # config.knn_memories_directory = '.tmp/baseline.memories.tmp/' 
        
    if config.is_random_vector:
        config.model_type = 'Other-test/random-vector/' + config.model_type
        # config.knn_memories_directory = 'Other-test/random-vector/' + config.knn_memories_directory
    if config._knn_dis_threshold != 0 and config._knn_dis_threshold != None:
        config.model_type = 'Other-test/is_use_threshold/' + config.model_type
        # config.knn_memories_directory = 'Other-test/is_use_threshold/' + config.knn_memories_directory
    # if config.:
    #     config.model_type = config.model_type + 'test-random-vector'
    
    if config.is_shuffle_knn or config.is_use_knn is False:
        config.shuffle = True
    # breakpoint()
    if config.current_dataset in ['AISHELL-1', 'HKUST']:
        config.is_zh = True
        config.language = 'zh'
        config.metric = 'cer'
    if config.current_dataset in ['LIBRISPEECH_CLEAN', 'LIBRISPEECH_OTHER']:
        config.is_zh = False
        config.language = 'en'
        config.metric = 'wer'
    if config.is_offline is True:
        if config.is_domain_datastore is True:
            config.mode_mode_path: str = config.pwd + '/Offline_domain_context/' +config.model_type
        else:
            config.mode_mode_path: str = config.pwd + '/Offline/' +config.model_type
    else:
        config.mode_mode_path: str = config.pwd + config.model_type
        
    if 'baseline' in config.model_type:
        config.mode_mode_path_dataset: str = config.mode_mode_path + '/' + config.language+ '-' +config.current_dataset + '-baseline'
        # config.knn_memories_directory = config.knn_memories_directory + config.current_dataset + '-baseline'
        
    else:
        config.mode_mode_path_dataset: str = config.mode_mode_path + '/' + config.language+ '-' +config.current_dataset + '-layer=' + str(config.knn_memorizing_layers)+ '-K=' + str(config._num_retrieved_memories_K)
        # config.knn_memories_directory = config.knn_memories_directory + config.current_dataset + '-layer=' + str(config.knn_memorizing_layers)+ '-K=' + str(config._num_retrieved_memories_K)
    
        
    
    config.knn_memories_directory = config.mode_mode_path_dataset + '/datestore_memories/'
    config.best_model_dir: str = config.mode_mode_path_dataset + '/model-checkpoint/'
    config.test_result_dir: str = config.mode_mode_path_dataset + '/result/'
    config.log_path: str = config.mode_mode_path_dataset + '/log/'
    config.tensorboard_path: str = config.mode_mode_path_dataset + '/tensorboard/' 


    config.datastore_path: str = config.pwd + 'datastore/' + config.language + '/' + config.current_dataset + '/'
    config.text_data_dir: str = config.pwd +'data/'+ config.language 
    if config.is_sliding_k != 0:
        config.text_data_dir: str = config.pwd +'data/'+ 'sliding-k/sliding-'+str(config.is_sliding_k)+'/' 
    
    config.pretrained_model: str = config.pwd + 'pretrained-model/'+ config.language+'/BART'
    # if config.is_use_knn:
    if config.is_from_ckpt:
        config.pretrained_model: str = config.pwd + 'pretrained-model/checkpoint/'+ config.current_dataset
    #     else:
    #         config.pretrained_model: str = config.pwd + 'pretrained-model/'+ config.language+'/KNN_BART'
    config.Model_config = AutoConfig.from_pretrained(config.pretrained_model)
            
def reset_model_paras(config, MODEL_TYPE, knn_memorizing_layers,path):
    import time
    old_MODEL_TYPE = torch.load(path+'/pytorch_model.bin')
    logger.info('+++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    logger.info("Begin reset KNN model paras ")
    start_time = time.time()
    # logger.info('+++++++++++++++++++++++++++++++++++++++++++++++++++++++')

    for name, param in MODEL_TYPE.named_parameters():
        flag_string = 'model.decoder.layers.'
        if flag_string+str(knn_memorizing_layers-1)+'.knn_attn' in name:
            # breakpoint()
            old_param_name = name.replace('knn_attn', 'self_attn')
            if config.language == 'en' and config.is_from_ckpt is False:
                old_param_name = old_param_name.replace('model.','')
            old_param = old_MODEL_TYPE[old_param_name]
            param.data.copy_(old_param.data)
    del old_MODEL_TYPE
    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(f"Reset KNN model params finished in {elapsed_time:.4f} seconds")
    logger.info('+++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    return MODEL_TYPE
    
        


if __name__ == "__main__":
    
    config: Config = Config().parse_args()
    
    reset_config_parse(config)
        
    set_my_seed(config.seed)
    if os.path.exists(config.mode_mode_path_dataset):
        pass
    else:
        os.makedirs(config.mode_mode_path_dataset)

    if config.is_pretrained==True:
        CONFIG = BartConfig.from_pretrained(config.pretrained_model)
        if config.is_use_knn is True:
            CONFIG.knn_memorizing_layers = config.knn_memorizing_layers
            CONFIG.is_use_knn = True
            MODEL_TYPE = BartForContextCorretion.from_pretrained(config.pretrained_model, config=CONFIG)
            MODEL_TYPE = reset_model_paras(config, MODEL_TYPE, CONFIG.knn_memorizing_layers, config.pretrained_model)
        else:
            MODEL_TYPE = BartForContextCorretion.from_pretrained(config.pretrained_model, config=CONFIG)
    else:
        MODEL_TYPE = BartForContextCorretion(config.Model_config)
        
    if config.language=='en':
        TOKENIZER = BartTokenizer.from_pretrained(config.pretrained_model)
    elif config.language=='zh': # Follow CPT Model Card: https://huggingface.co/fnlp/bart-base-chinese
        TOKENIZER = BertTokenizer.from_pretrained(config.pretrained_model)


    trainer = Trainer(
        config,
        text_processor=TextDataProcessor(
            config.text_data_dir, config),
        text_tokenizer=TOKENIZER,
        model=MODEL_TYPE,
        metric=evaluate.load(config.metric)
    )
    
    metric=evaluate.load('MM2') # MM2: Multi-reference M2
    
    if config.mode == 'train':
        logger.add(os.path.join(config.log_path, 'train.'+config.current_dataset+'.T-model-log.txt')) 
        if not os.path.exists(config.best_model_dir):    # resume from checkpoint, not re log the config information
            logger.info(config)
        trainer.train()
    # elif config.mode == 'test':
    logger.add(os.path.join(config.log_path, 'test.'+config.current_dataset+'.T-model-log.txt'))
    trainer.predict()



