import os
import random
from typing import Dict, List, Optional, Tuple 
import numpy as np
import torch
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
from tqdm.auto import tqdm
from transformers import (AdamW, AutoConfig, AutoModelForSeq2SeqLM,BertTokenizer,BartTokenizer,
                          AutoTokenizer, BertConfig, PreTrainedModel,BartForConditionalGeneration,
                          PreTrainedTokenizer, get_scheduler, set_seed)
from transformers.modeling_outputs import Seq2SeqLMOutput
from transformers.models import bart

from processor import DataProcessor, TextDataProcessor, TextInputExample
from utils import  EarlyStopping

from model.modeling_bart import (BartForContextCorretion, BartModel)

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class Config(Tap):


    seed: int = 2022
    pwd: str = '/home/users/jiangjin/jiangjin_bupt/ASR_CORRECTION/Context_Correction/context_seq2seq/' #'/home/jiangjin/ASR_CORRECTION/TAP/' # for different machine

    # KNN Code  
    # batch_size 的设定，如果原有baseline batch为100，那train为100，test dev doc的个数若小于100，则为为doc个数
    train_batch_size: int = 50
    dev_batch_size: int = 22
    test_batch_size: int = 24
    
    current_dataset: str = 'HKUST'#'LIBRISPEECH_OTHER'#'LIBRISPEECH'#'LIBRISPEECH_CLEAN_100''
    is_use_knn: bool = True
    is_from_ckpt: bool = False
    is_shuffle_knn: bool = False
    max_seq_length: int = 80 # 一个句子的max length 是
    is_add_sos_eos: bool = False
 
    language: str = 'en'
    is_zh: bool = False
    if current_dataset in ['AISHELL-1', 'HKUST']:
        is_zh = True
        language = 'zh'
    metric: str = 'cer'
    if language == 'en': metric = 'wer'
 
    mode: str = 'train'    
    is_pretrained: bool = True
    model_type: str = '' #'nopretrained-' # default
    shuffle: bool = False
    

    SEGMENTS: int = 1 #一个subsequence包含几个句子

    knn_memories_directory: str = ''
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
    epochs: int = 100

    early_stop = EarlyStopping(patience=7)
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
        
        model.resize_token_embeddings(len(text_tokenizer))
        self.model = model.to(self.config.get_device())

        # 2. build text dataloader
        logger.info('init text  dataloaders ...')

        self.train_dataloader = self.create_dataloader(
            dataset=text_processor.get_train_dataset(self.config.train_batch_size),
            batch_size=self.config.train_batch_size,
            shuffle=self.config.shuffle,
            collate_fn=self.convert_examples_to_features,
        )
        self.dev_dataloader = self.create_dataloader(
            dataset=text_processor.get_dev_dataset(self.config.dev_batch_size),
            batch_size=self.config.dev_batch_size,
            shuffle=False, #self.config.shuffle,
            collate_fn=self.convert_examples_to_features,
        )
        self.test_dataloader = self.create_dataloader(
            dataset=text_processor.get_test_dataset(self.config.test_batch_size),
            batch_size=self.config.test_batch_size,
            shuffle=False, #self.config.shuffle,
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
        self.context_data.lr = self.optimizer.defaults['lr']

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
        
    def train(self):
        """the main train epoch"""
        logger.info('start training ...')
        logger.info(f'  num example = {len(self.train_dataloader)}')
        logger.info(f'  num epochs = {self.config.epochs}')
        logger.info(f'  total optimization step = {self.config.max_train_steps}')

        self.on_train_start()
        for _ in range(self.config.epochs):            
            self.context_data.epoch += 1
            self.train_epoch()

            self.on_epoch_end()
            if self.config.early_stop_flag:
                logger.info('early stopping on train epoch')
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

    def train_epoch(self):
        """handle the logit of training epoch

        Args:
            epoch (int): _description_
        # """
        logger.info('\n')
        logger.info('=======================================')
        logger.info(f'training epoch<{self.context_data.epoch}> ...')
        self.train_bar = tqdm(total=len(self.train_dataloader))

        with self.model.knn_memories_context(batch_size = self.config.train_batch_size, mode='train', knn_memories_directory=self.config.knn_memories_directory) as knn_memories: 
            for text_batch in self.train_dataloader:
                
                self.on_batch_start()
                
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
                    logger.info('early stopping')
                    break

                self.on_batch_end()
                
    def on_epoch_end(self):
        self.context_data.dev_cer = self.evaluate(self.dev_dataloader)
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
        self.model.eval()

        all_decoded_preds = []
        all_decoded_labels = []
        # 这里因为tqdm 中包含 tqdm 所以，暂时采用logger方式
        # for text_batch in tqdm(dataloader, desc='evaluation stage ...'):
        with self.model.knn_memories_context(batch_size = self.config.dev_batch_size, mode='dev', knn_memories_directory=self.config.knn_memories_directory) as knn_memories: 
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
        
        all_decoded_inputs = []

        all_decoded_preds = []
        all_decoded_labels = []

        with self.model.knn_memories_context(batch_size = self.config.test_batch_size, mode='test', knn_memories_directory=self.config.knn_memories_directory) as knn_memories: 
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
            logger.info(f'raw/cer is {raw_score}')
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
                data_output_list.append(item_pred + ' ' + item_label + '\n') 
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
    
    import tensorflow as tf
    tf.random.set_seed(seed)
    
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
            config.knn_memories_directory = '.tmp/knn.ckpt.memories'
        else:
            config.model_type = config.model_type + 'T-model-knn'
            config.knn_memories_directory = '.tmp/knn.memories' 
        if config.is_shuffle_knn:
            config.model_type = config.model_type + '-shuffle'
            config.knn_memories_directory = config.knn_memories_directory + '.shuffle'
        config.knn_memories_directory = config.knn_memories_directory + '/'
    else:
        config.model_type = config.model_type + 'T-model-baseline'
        config.knn_memories_directory = '.tmp/baseline.memories/'
    
    if config.is_shuffle_knn or config.is_use_knn is False:
        config.shuffle = True
    
    config.mode_mode_path: str = config.pwd + config.model_type
    config.mode_mode_path_dataset: str = config.mode_mode_path + '/' + config.current_dataset
    
    config.best_model_dir: str = config.mode_mode_path_dataset + '/model-checkpoint/'
    config.test_result_dir: str = config.mode_mode_path_dataset + '/result/'
    config.log_path: str = config.mode_mode_path_dataset + '/log/'
    config.tensorboard_path: str = config.mode_mode_path_dataset + '/tensorboard/' 

    if config.current_dataset in ['AISHELL-1', 'HKUST']:
        config.is_zh = True
        config.language = 'zh'

    config.text_data_dir: str = config.pwd +'data/'+ config.language 
    
    config.pretrained_model: str = config.pwd + 'pretrained-model/'+ config.language+'/BART'
    if config.is_use_knn:
        if config.is_from_ckpt:
            config.pretrained_model: str = config.pwd + 'pretrained-model/checkpoint/'+ config.current_dataset
        else:
            config.pretrained_model: str = config.pwd + 'pretrained-model/'+ config.language+'/KNN_BART'



if __name__ == "__main__":
    
    config: Config = Config().parse_args()
    
    reset_config_parse(config)
        
    # set_my_seed(config.seed)
    if os.path.exists(config.mode_mode_path_dataset):
        pass
    else:
        os.makedirs(config.mode_mode_path_dataset)

    if config.is_pretrained==True:
        MODEL_TYPE = BartForContextCorretion.from_pretrained(config.pretrained_model)
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
        metric=load_metric(config.metric)
    )
    if config.mode == 'train':
        logger.add(os.path.join(config.log_path, 'train.'+config.current_dataset+'.T-model-log.txt'))    
        logger.info(config)
        trainer.train()
    elif config.mode == 'test':
        logger.add(os.path.join(config.log_path, 'test.'+config.current_dataset+'.T-model-log.txt'))
        trainer.predict()



