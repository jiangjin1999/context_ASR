import json
import os
import random
import shutil
from modulefinder import Module
# from re import L
from typing import Dict, List, Optional, Tuple  # 将wav2vec processor 和 model 合并
import pdb
import numpy as np
import torch
# import torchaudio
# from MeCab import Model
from datasets import Metric, load_metric
# import evaluate
from genericpath import exists
from loguru import logger
from sklearn.feature_selection import SelectFdr
from sqlalchemy import false
from sympy import true
from tap import Tap
from torch import nn, tensor
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel 
from tqdm.auto import tqdm
from transformers import (AdamW, AutoConfig, AutoModelForSeq2SeqLM,BertTokenizer,
                          AutoTokenizer, BertConfig, PreTrainedModel,BartForConditionalGeneration,
                          PreTrainedTokenizer, get_scheduler, set_seed)
from transformers.modeling_outputs import Seq2SeqLMOutput
from transformers.models import bart

from processor import DataProcessor, TextDataProcessor, TextInputExample
from utils import CustomSchedule, EarlyStopping, Similarity

from model.modeling_bart import (BartForContextCorretion, BartModel)


# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"
# torch.autograd.set_detect_anomaly(True) 


class Config(Tap):


    seed: int = 2022


    pwd: str = '/home/users/jiangjin/jiangjin_bupt/ASR_CORRECTION/Context_Correction/context_seq2seq/'#'/home/jiangjin/ASR_CORRECTION/TAP/'

    # KNN Code  
    # batch_size 的设定，如果原有baseline batch为100，那train为100，test dev doc的个数若小于100，则为为doc个数
    train_batch_size: int = 100
    dev_batch_size: int = 40
    test_batch_size: int = 20
    is_use_knn: bool = False
    is_from_ckpt: bool = False
    is_shuffle_knn: bool = False
    SEGMENTS: int = 1 #一个subsequence包含几个句子
    max_seq_length: int = 40 # 一个句子的max length 是
    
    
    


    mode: str = 'train'    
    is_use_DDP:bool = False
    

    current_dataset: str = 'AISHELL-1'#'LIBRISPEECH_OTHER'#'LIBRISPEECH'#'LIBRISPEECH_CLEAN_100'#'AIDATATANG' #['AISHELL-1', 'AIDATATANG', 'thchs'][0]
    is_pretrained: bool = True
    language: str = 'en'
    if current_dataset in ['AISHELL-1', 'HKUST', 'thchs']:
        is_zh = True
        language = 'zh'
    if language == 'en':
        max_seq_length: int = 100

    model_type: str = '' #'nopretrained-' # default
    # if is_pretrained is True:
    #     model_type = 'pretrained-'

    # model_type = model_type + 'T-model'
        
    mode_mode_path: str = pwd + model_type
    mode_mode_path_dataset: str = mode_mode_path + '/' + current_dataset
    
    best_model_dir: str = mode_mode_path_dataset + '/model-checkpoint/'
    test_result_dir: str = mode_mode_path_dataset + '/result/'
    log_path: str =mode_mode_path_dataset + '/log/'
    tensorboard_path: str =mode_mode_path_dataset + '/tensorboard/' 

    is_zh: bool = False


    text_data_dir: str = pwd +'data/'+ language 
    
    audio_feature_path: str = text_data_dir +'/' +current_dataset +'/audio-feature/wav2vec_feature.h5'

    pretrained_model: str = pwd + 'pretrained-model/'+language+'/BART'
    phoneme_model_path: str = pwd + 'pretrained-model/'+language+'/phoneme_model'
    Model_config = AutoConfig.from_pretrained(pretrained_model)

    shuffle: bool = False
    sampler = None
    # librispeech max length is 100
    # zh : 50

    learning_rate: float = 5e-5
    weight_decay: float = 0.02
    lr_scheduler_type: str = 'cosine'
    num_warmup_steps: int = 200
    max_train_steps: int = 2000
    gradient_accumulation_steps: int = 1
    epochs: int = 100
    num_batch_per_evaluation: int = 10

    # 模型相关 参数配置
    early_stop = EarlyStopping(patience=5)
    device: str = 'cuda'
    metric: str = 'cer'
    if language == 'en': metric = 'wer'
    early_stop_flag: str = False

    # arg for ddp
    local_rank = '0'

    # for CustomSchedule
    d_model = 768
    
    sim = Similarity(temp=0.05)

    def get_device(self):
        """return the device"""
        if config.is_use_DDP is True:
            return torch.device(self.device, int(local_rank))
        else:
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
        is_use_knn,
    ) -> None:
        self.config = config
        self.text_tokenizer = text_tokenizer
        self.metric = metric
        self.is_use_knn = is_use_knn


        model.resize_token_embeddings(len(text_tokenizer))

        if self.config.is_use_DDP is True:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = model.to(self.config.get_device())
            self.model = DDP(model, device_ids=[int(self.config.local_rank)], output_device=[int(self.config.local_rank)], find_unused_parameters=False)
        else:
            self.model = model.to(self.config.get_device())

        # 2. build text & audio dataloader
        if self.config.local_rank=='0':
            logger.info('init text  dataloaders ...')

        if self.config.is_use_DDP is True:
            self.train_dataloader = self.create_DDP_dataloader(
                dataset=text_processor.get_train_dataset(self.config.train_batch_size),
                batch_size=self.config.train_batch_size,
                shuffle=False,
                collate_fn=self.convert_examples_to_features,
            )
            self.dev_dataloader = self.create_dataloader(
                dataset=text_processor.get_dev_dataset(self.config.dev_batch_size),
                batch_size=self.config.dev_batch_size,
                shuffle=False,
                collate_fn=self.convert_examples_to_features,
            )
            self.test_dataloader = self.create_dataloader(
                dataset=text_processor.get_test_dataset(self.config.test_batch_size),
                batch_size=self.config.test_batch_size,
                shuffle=False,
                collate_fn=self.convert_examples_to_features,
                )
        else:   
            self.train_dataloader = self.create_dataloader(
                dataset=text_processor.get_train_dataset(self.config.train_batch_size),
                batch_size=self.config.train_batch_size,
                shuffle=self.config.shuffle,
                collate_fn=self.convert_examples_to_features,
            )
            self.dev_dataloader = self.create_dataloader(
                dataset=text_processor.get_dev_dataset(self.config.dev_batch_size),
                batch_size=self.config.dev_batch_size,
                shuffle=self.config.shuffle,
                collate_fn=self.convert_examples_to_features,
            )
            self.test_dataloader = self.create_dataloader(
                dataset=text_processor.get_test_dataset(self.config.test_batch_size),
                batch_size=self.config.test_batch_size,
                shuffle=self.config.shuffle,
                collate_fn=self.convert_examples_to_features,
            )


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
        self.config.max_train_steps = len(self.train_dataloader) * self.config.epochs 
        # self.config.num_warmup_steps = self.config.max_train_steps * 0.1
        self.lr_scheduler = get_scheduler(
            name=config.lr_scheduler_type,
            optimizer=self.optimizer,
            num_warmup_steps=config.num_warmup_steps, # 前 * step 进行warm up（即让lr 从0-设定的lr）
            num_training_steps=config.max_train_steps, # 最大的step
        )

        self.context_data = ContextContainer()
        self._init_output_dir()
        self.writer: SummaryWriter = SummaryWriter(self.config.tensorboard_path)
        self.train_bar: tqdm = None


    def create_DDP_dataloader(self, dataset: Dataset, batch_size, collate_fn, shuffle) -> DataLoader:
        self.config.sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        if self.config.is_use_DDP is True:
            return DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle, # self.config.shuffle,
                collate_fn=collate_fn,
                sampler=self.config.sampler
            )
        else:
            return DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle, # self.config.shuffle,
                collate_fn=collate_fn,
                # sampler=torch.utils.data.distributed.DistributedSampler(dataset)
            )

    def create_dataloader(self, dataset: Dataset, batch_size, collate_fn, shuffle) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle, # self.config.shuffle,
            collate_fn=collate_fn,
            # sampler=torch.utils.data.distributed.DistributedSampler(dataset)
        )

    def convert_examples_to_features(self, examples: List[TextInputExample]):
        """convert the examples to features"""
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
        return encoded_features['input_ids'], label_features['input_ids']

    def _init_output_dir(self):
        if self.config.local_rank=='0':
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
        self.train_bar.set_description('\t'.join(infos))

    def on_batch_start(self):
        '''handle the on batch start logits
        '''
        self.model.train()
        # self.phoneme_encoder.train()
        # self.audio_encoder.train()

    def on_batch_end(self):
        """handle the on batch training is ending logits
        """
        # 1. update global step
        self.context_data.train_step += 1
        # print(self.context_data.train_step)
        self._update_train_bar()

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
        

    def train_epoch(self):
        """handle the logit of training epoch

        Args:
            epoch (int): _description_
        # """
        if self.config.local_rank=='0':
            logger.info('\n')
            logger.info('=======================================')
            logger.info(f'training epoch<{self.context_data.epoch}> ...')
        self.train_bar = tqdm(total=len(self.train_dataloader))

        with self.model.knn_memories_context(batch_size = self.config.train_batch_size, mode='train') as knn_memories: 
            for text_batch in self.train_dataloader:
                
                self.on_batch_start()
                
                self.context_data.lr = self.optimizer.defaults['lr']

                input_ids, labels = text_batch
                input_ids, labels = input_ids.to(
                    self.config.get_device()), labels.to(self.config.get_device())


                self.optimizer.zero_grad()    
                
                # forward on text data
                output: Seq2SeqLMOutput = self.model(
                    input_ids=input_ids, 
                    labels=labels,
                    decoder_knn_memories = knn_memories,
                    )

                self.context_data.loss = output.loss.sum().detach().cpu().numpy().item()
                self.context_data.output_loss = output.loss
                self.context_data.output_loss.backward()
                

                self.optimizer.step() 
                self.lr_scheduler.step()  
                    
                if self.config.early_stop_flag:
                    if self.config.local_rank=='0':
                        logger.info('early stopping')
                        break

                self.on_batch_end()
    

    def evaluate(self, dataloader,):
        """handle the logit of evaluating

        Args:
            epoch (int): _description_
        """
        self.model.eval()

        all_decoded_preds = []
        all_decoded_labels = []
        # 这里因为tqdm 中包含 tqdm 所以，暂时采用logger方式
        # for text_batch in tqdm(dataloader, desc='evaluation stage ...'):
        with self.model.knn_memories_context(batch_size = self.config.dev_batch_size, mode='dev') as knn_memories: 
            for text_batch in dataloader:
                with torch.no_grad():
                    input_ids, labels = text_batch
                    input_ids, labels = input_ids.to(
                        self.config.get_device()), labels.to(self.config.get_device())
                    
                    max_token = input_ids.shape[1]
                    generated_tokens: Seq2SeqLMOutput = self.model.generate(
                        input_ids=input_ids, max_length=max_token, decoder_knn_memories = knn_memories)
                    
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

        all_decoded_labels = [item for item in all_decoded_labels if item!='。']
        all_decoded_preds = [item for item in all_decoded_preds if item!='。']
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
            if self.config.local_rank=='0':
                logger.info('\n')
                logger.info(f'dev/best_cer is {self.context_data.dev_cer}')
                self.writer.add_scalar(
                    tag='dev/best_cer',
                    scalar_value=self.context_data.best_dev_cer,
                    global_step=self.context_data.train_step
                )
            self.context_data.test_cer = self.predict('test')
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
        if self.config.is_use_DDP is True:
            torch.save(self.model.module.state_dict(), path+'/checkpoint_best.pt')
        else:
            torch.save(self.model.state_dict(), path+'/checkpoint_best.pt')

    def train(self):
        """the main train epoch"""
        if self.config.local_rank=='0':
            logger.info('start training ...')
            logger.info(f'  num example = {len(self.train_dataloader)}')
            logger.info(f'  num epochs = {self.config.epochs}')
            logger.info(f'   Total train batch size (w. parallel, distributed & accumulation) = {self.config.train_batch_size * self.config.gradient_accumulation_steps}' )
            logger.info(f"  Gradient Accumulation steps = {self.config.gradient_accumulation_steps}")
            logger.info(f'  total optimization step = {self.config.max_train_steps}')

        self.on_train_start()
        for _ in range(self.config.epochs):            
            self.context_data.epoch += 1
            if self.config.is_use_DDP is True:
                self.config.sampler.set_epoch(self.context_data.epoch)
            self.train_epoch()

            self.on_epoch_end()
            if self.config.early_stop_flag:
                if self.config.local_rank=='0':
                    logger.info('early stopping on train epoch')
                break

    def on_epoch_end(self):
        self.context_data.dev_cer = self.evaluate(self.dev_dataloader)
        self.config.early_stop_flag = self.config.early_stop.step(self.context_data.dev_cer)
        if self.config.local_rank=='0':
            logger.info('\n')
            logger.info(f'dev/cer is {self.context_data.dev_cer}')
        self.writer.add_scalar(
            tag='dev/cer',
            scalar_value=self.context_data.dev_cer,
            global_step=self.context_data.train_step
        )


        self.on_evaluation_end(self.context_data.dev_cer)
        
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

    def predict(self, FLAG: Optional[str] = None,):
        """ predict the example
            test_dataset = ['test_aidatatang', 'test_magicdata', 'test_thchs']
        """
        # self.load_model(self.config.best_model_dir)
        dataloader = self.test_dataloader
        if self.config.local_rank=='0':
            logger.info('\n')
            logger.info('start predicting ...')
        if FLAG is not None:
            pass
        else:
            self.load_model(self.config.best_model_dir + 'checkpoint_best.pt')

        self.model.eval()
        
        all_decoded_inputs = []

        all_decoded_preds = []
        all_decoded_labels = []

        with self.model.knn_memories_context(batch_size = self.config.test_batch_size, mode='test') as knn_memories: 
            for text_batch in dataloader:
                with torch.no_grad():
                    input_ids, labels = text_batch
                    input_ids, labels = input_ids.to(
                        self.config.get_device()), labels.to(self.config.get_device())
                    
                    max_token = input_ids.shape[1]
                    generated_tokens: Seq2SeqLMOutput = self.model.generate(
                        input_ids=input_ids, max_length=max_token, decoder_knn_memories = knn_memories)
                    
                    generated_tokens = generated_tokens.detach().cpu().numpy()
                    labels = labels.detach().cpu().numpy() 
             
                    if FLAG is not None:
                        pass
                    else:
                        decoded_inputs = self.text_tokenizer.batch_decode(
                            input_ids, skip_special_tokens=True)
                    
                    decoded_preds = self.text_tokenizer.batch_decode(
                        generated_tokens, skip_special_tokens=True)
                    decoded_labels = self.text_tokenizer.batch_decode(
                        labels, skip_special_tokens=True)

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
        all_decoded_inputs = [item for item in all_decoded_inputs if item!='。']
        all_decoded_labels = [item for item in all_decoded_labels if item!='。']
        all_decoded_preds = [item for item in all_decoded_preds if item!='。']
        assert len(all_decoded_labels)==len(all_decoded_preds)==len(all_decoded_inputs)
        
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
        if self.config.local_rank=='0':
            if FLAG is not None:
                pass
            else:
                logger.info('\n')
                logger.info(f'raw/cer is {raw_score}')
            logger.info('\n')
            logger.info(f'test/cer is {self.context_data.test_cer}')
        # add test cer every time evaluate test data
        self.model.train()
        return metric_score

    def load_model(self, path):
        if self.config.local_rank=='0':
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
                data_output_list.append(item_pred + ' ' + item_label + '\n') 
            f_result.writelines(data_output_list)


def set_my_seed(seed):
    '''random:
        python
        Numpy'''
    set_seed(seed)
    from torch.backends import cudnn
    cudnn.benchmark = False
    cudnn.deterministic = True
    
def reset_config_parse(config):
    # 讲 config 中所有 由 if修改的变量 和 由其他变量定义的变量 在该函数内重新定义。
    # if config.is_pretrained is True:
        # config.model_type = 'pretrained-'
    if config.is_use_knn:
        if config.is_from_ckpt:
            config.model_type = config.model_type + 'T-model-knn-ckpt'  
        else:
            config.model_type = config.model_type + 'T-model-knn'
        if config.is_shuffle_knn:
            config.model_type = config.model_type + '-shuffle'
    else:
        config.model_type = config.model_type + 'T-model-baseline'
    
        
    if config.is_shuffle_knn or config.is_use_knn is False :
        config.shuffle = True
    
    config.mode_mode_path: str = config.pwd + config.model_type
    config.mode_mode_path_dataset: str = config.mode_mode_path + '/' + config.current_dataset
    
    config.best_model_dir: str = config.mode_mode_path_dataset + '/model-checkpoint/'
    config.test_result_dir: str = config.mode_mode_path_dataset + '/result/'
    config.log_path: str = config.mode_mode_path_dataset + '/log/'
    config.tensorboard_path: str = config.mode_mode_path_dataset + '/tensorboard/' 

    if config.current_dataset in ['AISHELL-1', 'AIDATATANG', 'thchs']:
        config.is_zh = True
        config.language = 'zh'

    config.text_data_dir: str = config.pwd +'data/'+ config.language 
    
    config.audio_feature_path: str = config.text_data_dir +'/' + config.current_dataset +'/audio-feature/wav2vec_feature.h5'

    config.pretrained_model: str = config.pwd + 'pretrained-model/'+ config.language+'/BART'
    if config.is_use_knn:
        if config.is_from_ckpt:
            config.pretrained_model: str = config.pwd + 'pretrained-model/checkpoint/'+ config.current_dataset
        else:
            config.pretrained_model: str = config.pwd + 'pretrained-model/'+ config.language+'/KNN_BART'
    config.phoneme_model_path: str = config.pwd + 'pretrained-model/'+ config.language+'/phoneme_model'

    if config.language == 'en':
        config.max_seq_length: int = 100

    if config.language == 'en':
        config.audio_encoder_input_dim = 768

    if config.language == 'en': config.metric = 'wer'


if __name__ == "__main__":
    
    config: Config = Config().parse_args()
    
    reset_config_parse(config)

    if config.is_use_DDP is True:
        # 新增1:依赖
        import torch.distributed as dist
        from torch.nn.parallel import DistributedDataParallel as DDP

        # # 新增：从外面得到local_rank参数
        # import argparse
        # parser = argparse.ArgumentParser()
        # parser.add_argument("--local_rank", default=-1)
        # FLAGS = parser.parse_args()
        # local_rank = FLAGS.local_rank
        local_rank = config.local_rank

        # print(local_rank)

        # 新增：DDP backend初始化
        torch.cuda.set_device('cuda:'+str(local_rank))
        dist.init_process_group(backend='nccl')  # nccl是GPU设备上最快、最推荐的后端
    
    set_my_seed(config.seed)
    if os.path.exists(config.mode_mode_path_dataset):
        pass
    else:
        os.makedirs(config.mode_mode_path_dataset)

    if config.is_pretrained==True:
        MODEL_TYPE = BartForContextCorretion.from_pretrained(config.pretrained_model)
    else:
        MODEL_TYPE = BartForContextCorretion(config.Model_config)

    trainer = Trainer(
        config,
        text_processor=TextDataProcessor(
            config.text_data_dir, config),
        text_tokenizer=BertTokenizer.from_pretrained(config.pretrained_model),
        model=MODEL_TYPE,
        metric=load_metric(config.metric),
        is_use_knn=config.is_use_knn,
    )
    # if config.mode == 'train':
    #     logger.add(os.path.join(config.log_path, 'train.'+config.current_dataset+'.T-model-log.txt'))
    #     if config.local_rank=='0':
    #         logger.info(config)
    #     trainer.train()
    # elif config.mode == 'test':
    logger.add(os.path.join(config.log_path, 'test.'+config.current_dataset+'.T-model-log.txt'))
    # if config.local_rank=='0':
        # logger.info(config)
    trainer.predict()



