import os
import math
import time
import torch
import faiss
from tqdm import tqdm
import faiss.contrib.torch_utils
import numpy as np
from pathlib import Path
from functools import wraps

from contextlib import ExitStack, contextmanager

from einops import rearrange, pack, unpack

# multiprocessing

from joblib import Parallel, delayed, cpu_count

# constants

FAISS_INDEX_GPU_ID = int(os.getenv('FAISS_INDEX_GPU_ID', 0))

DEFAULT_KNN_MEMORY_MEMMAP_DIRECTORY = './.tmp/knn.memories'

# helper functions

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def cast_list(val):
    return val if isinstance(val, list) else [val]

def all_el_unique(arr):
    return len(set(arr)) == len(arr)

@contextmanager
def multi_context(*cms):
    with ExitStack() as stack:
        yield [stack.enter_context(cls) for cls in cms]

def count_intersect(x, y):
    # returns an array that shows how many times an element in x is contained in tensor y
    return np.sum(rearrange(x, 'i -> i 1') == rearrange(y, 'j -> 1 j'), axis = -1)

def check_shape(tensor, pattern, **kwargs):
    return rearrange(tensor, f"{pattern} -> {pattern}", **kwargs)

# a wrapper around faiss IndexIVFFlat
# taking care of expiring old keys automagically

class KNN():
    def __init__(
        self,
        my_config,
        dim,
        max_num_entries,
        cap_num_entries = False,
        is_knn_gpu = True,
        M = 15,
        keep_stats = False
    ):
        if is_knn_gpu is True:
            quantizer = faiss.IndexFlatL2(dim)
            ncentroids = 80 # 64个中心点
            code_size = 64
            is_use_IVFPQ = False
            if is_use_IVFPQ is False:
                print("use IndexFlatL2-GPU")
                cpu_index = faiss.IndexFlatL2(dim)
                # breakpoint()
                gpu_index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 1, cpu_index)
            elif is_use_IVFPQ is True:
                cpu_index = faiss.IndexIVFPQ(quantizer, dim, ncentroids, code_size, 8) 
                cpu_index.nprobe = 32   # default nprobe is 1, try a few more
                co = faiss.GpuClonerOptions()
                co.useFloat16 = True
                gpu_index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, cpu_index, co)
            self.index = gpu_index
        else:
            # print("use IndexHNSWFlat-cpu")
            index = faiss.IndexHNSWFlat(dim, M, faiss.METRIC_INNER_PRODUCT) # dim: 向量维度 M：控制图的层数，层数  内积度量
            self.index = index
        self.max_num_entries = max_num_entries
        self.cap_num_entries = cap_num_entries
        self.is_trained = False
        self.keep_stats = keep_stats
        self.is_knn_gpu = is_knn_gpu
        self.my_config = my_config
        self.reset() # 每次创建一个knn都要reset一下

    def __del__(self):
        if hasattr(self, 'index'):
            del self.index

    def reset(self):
        self.ids = np.empty((0,), dtype = np.int32)

        if self.keep_stats:
            self.hits = np.empty((0,), dtype = np.int32)
            self.age_num_iterations = np.empty((0,), dtype = np.int32)
            self.ages_since_last_hit = np.empty((0,), dtype = np.int32)

        self.index.reset() # 清除索引中的所有向量和状态
        self.is_trained = False

    def train(self, x):
        # if x.shape[0] >= self.index.nlist:
        self.index.train(x)
        self.is_trained = True

    def add(self, x, ids):
        if not self.is_trained: # and self.index.ntotal > self.index.ncentroids:
            self.train(x)

        self.ids = np.concatenate((ids, self.ids))

        if self.keep_stats:#检查self.keep_stats的值来确定是否需要更新统计信息
            self.hits = np.concatenate((np.zeros_like(ids), self.hits))# 在ids张量的前面添加一个全零的数组，用于存储新的统计信息
            self.age_num_iterations = np.concatenate((np.zeros_like(ids), self.age_num_iterations))
            self.ages_since_last_hit = np.concatenate((np.zeros_like(ids), self.ages_since_last_hit))

        if self.cap_num_entries and len(self.ids) > self.max_num_entries:
            self.reset()# 调用reset()方法清空类中的所有存储的键值对。该方法会将类中的所有存储的键值对删除，并将统计信息重置为初始状态。
        # x = torch.tensor(x)
        return self.index.add(x)
    


    def search(
        self,
        x,
        topk,
        _knn_dis_threshold=None,
        nprobe = 8,
        return_distances = False,
        increment_hits = False,
        increment_age = True
    ):
        if not self.is_trained:
            if self.is_knn_gpu is True:
                return torch.from_numpy(np.full((x.shape[0], topk), -1))
            else:
                return np.full((x.shape[0], topk), -1)
        # self.index.
        # if self.is_knn_gpu is True:
        #     x = torch.tensor(x).cuda()
        # else:
        #     pass
        distances, indices = self.index.search(x, k = topk) # distance: 960 32 # length的每个head都会进行搜索一个32个 结果。
        # distance:返回960 个 32个distance ，以及每一个在database中对应的index
        # 加入 K 之外的阈值限制：
        if _knn_dis_threshold != 0 and _knn_dis_threshold is not None:
            _knn_dis_threshold = np.min(np.median(distances, axis = 0)) # 32个中的最小值
            indices[distances < _knn_dis_threshold] = -1 # distances 中 小于阈值的 对应位置的indices设置为-1
 
        if increment_hits and self.keep_stats:
            hits = count_intersect(self.ids, rearrange(indices, '... -> (...)'))
            self.hits += hits

            self.ages_since_last_hit += 1
            self.ages_since_last_hit *= (hits == 0)

        if increment_age and self.keep_stats:
            self.age_num_iterations += 1

        if return_distances:
            return indices, distances

        return indices

# KNN memory layer, where one can store key / value memories
# can automatically take care of a collection of faiss indices (across batch dimension)

class KNNMemory():
    def __init__(
        self,
        my_config,
        mode,
        dim,
        max_memories = 16000,
        is_knn_gpu = True,
        num_indices = 1,
        memmap_filename = './knn.memory.memmap',
        multiprocessing = True
    ):
        self.dim = dim
        self.num_indices = num_indices
        self.scoped_indices = list(range(num_indices))

        self.max_memories = max_memories
        self.shape = (num_indices, max_memories, 2, dim) # 24个datastore/ 每个datastore的最大值 /2 key values/ dim 64 是存的vector的维度
        self.db_offsets = np.zeros(num_indices, dtype = np.int32)  # 实际中的db 就是用 numpy数组来存储的。
        # breakpoint()
        if my_config.is_offline is True:
            path = my_config.datastore_path + [item for item in os.listdir(my_config.datastore_path) if mode in item][0]
            self.db = np.memmap(path, mode = 'r+', dtype = np.float32, shape = self.shape)
        else:    
            shape_string = '_'+str(self.shape[0])+'_'+str(self.shape[1])+'_'+str(self.shape[2])+'_'+str(self.shape[3])
            self.db = np.memmap(memmap_filename+shape_string+'.npy', mode = 'w+', dtype = np.float32, shape = self.shape)
        if my_config.is_domain_datastore is True:
            num_indices = 1
        else:
            pass
        self.knns = [KNN(my_config=my_config,dim = dim, max_num_entries = max_memories, cap_num_entries = True, is_knn_gpu=is_knn_gpu) for _ in range(num_indices)]

        self.my_config = my_config
        self.is_knn_gpu = is_knn_gpu
        # self.knns[1].index.add(np.array(self.db[1,:,0,:]))
        # self.knns = [self.knns[i].index.add(np.array(self.db[i,:,0,:])) for i in range(1)]
        # 将datastore中的数据，add到knn index 中
        
    
        self.n_jobs =  cpu_count() if multiprocessing else 1

    def set_scoped_indices(self, indices):
        indices = list(indices)
        assert all_el_unique(indices), f'all scoped batch indices must be unique, received: {indices}'
        assert all([0 <= i < self.num_indices for i in indices]), f'each batch index must be between 0 and less than {self.num_indices}: received {indices}'
        self.scoped_indices = indices

    @contextmanager
    def at_batch_indices(self, indices):
        prev_indices = self.scoped_indices
        self.set_scoped_indices(indices)
        yield self
        self.set_scoped_indices(prev_indices)

    def clear(self, batch_indices = None):
        if not exists(batch_indices):
            batch_indices = list(range(self.num_indices))

        batch_indices = cast_list(batch_indices)

        for index in batch_indices:
            knn = self.knns[index]
            knn.reset()

        self.db_offsets[batch_indices] = 0
        
    def offline_add(self, batch_indices = None, db_index_list=None):
        # if not exists(batch_indices):
        #     batch_indices = list(range(self.num_indices))

        batch_indices = cast_list(batch_indices)

        for index in tqdm(batch_indices):
            start_index = db_index_list[index][0]
            sentence_length = self.my_config.max_seq_length
            end_index = db_index_list[index][1]
            if self.my_config.is_domain_datastore is True:
                add_vectors = np.concatenate(np.array(self.db[:,:,0,:]), axis=0)
            else:
                add_vectors = np.array(self.db[index,start_index*sentence_length:end_index*sentence_length,0,:])
            if self.knns[index].is_trained is False:
                self.knns[index].index.train(add_vectors)
                self.knns[index].is_trained = True
            self.knns[index].index.add(add_vectors)
            self.db_offsets[index] = end_index*sentence_length
            # self.knns[index].is_trained = True
        # self.db_offsets[batch_indices] = 0


    def add(self, memories):
        check_shape(memories, 'b n kv d', d = self.dim, kv = 2, b = len(self.scoped_indices))

        memories = memories.detach().cpu().numpy()
        memories = memories[:, -self.max_memories:]
        num_memories = memories.shape[1] # 存储着，现在memory中存在多少个向量对了

        knn_insert_ids = np.arange(num_memories)

        keys = np.ascontiguousarray(memories[..., 0, :])
        knns = [self.knns[i] for i in self.scoped_indices]
        db_offsets = [self.db_offsets[i] for i in self.scoped_indices]

        # use joblib to insert new key / value memories into faiss index

        @delayed
        def knn_add(knn, key, db_offset):
            knn.add(key, ids = knn_insert_ids + db_offset)
            return knn
        # breakpoint()
        updated_knns = Parallel(n_jobs = self.n_jobs)(knn_add(*args) for args in zip(knns, keys, db_offsets))
        self.knns = updated_knns

        # add the new memories to the memmap "database"

        add_indices = (rearrange(np.arange(num_memories), 'j -> 1 j') + rearrange(self.db_offsets[list(self.scoped_indices)], 'i -> i 1')) % self.max_memories
        self.db[rearrange(np.array(self.scoped_indices), 'i -> i 1'), add_indices] = memories
        self.db.flush()#数据强制刷新到磁盘

        self.db_offsets += num_memories # 更新db_offsets
        
        

    def search(
        self,
        queries,
        topk,
        _knn_dis_threshold=None,
        nprobe = 8,
        increment_hits = True,
        increment_age = True
    ):
        # check 输入query 的第一个纬度为db的维度64，最后一个纬度为knnmemory 的维度
        check_shape(queries, 'b ... d', d = self.dim, b = len(self.scoped_indices)) # torch.Size([40, 12, 80, 64])
        # head num-12 和 
        queries, ps = pack([queries], 'b * d') # torch.Size([40, 960, 64]), [torch.Size([12, 80])]
        # 12✖️80 中的 960 是 [80:80:...:80] 这样拼起来的。 
        device = queries.device
        
        if self.is_knn_gpu is True:
            pass
        else:
            queries = queries.detach().cpu().numpy()

        all_masks = []
        all_key_values = []
        # 把 40 层的 knnmemory 放在list中 list
        if self.knns[0].my_config.is_domain_datastore is True:
            knns = [self.knns[i] for i in [0]]
        else:
            knns = [self.knns[i] for i in self.scoped_indices]

        # parallelize faiss search

        @delayed
        def knn_search(knn, query): # query.shape: 960,64 # knn batch size 个knn中的一个
            # query; 480 64, 32
            return knn.search(query, topk, _knn_dis_threshold, nprobe, increment_hits = increment_hits, increment_age = increment_age)
        # 只返回了knn的index，没有返回 distance
        if self.knns[0].my_config.is_domain_datastore is True:
            fetched_indices = Parallel(n_jobs = self.n_jobs)(knn_search(knns[0],query) for query in queries) 
        else: # 128 480() 32
            fetched_indices = Parallel(n_jobs = self.n_jobs)(knn_search(*args) for args in zip(knns, queries)) 
        # 40,960,32
        # get all the memory key / values from memmap 'database'
        # todo - remove for loop below
        
        # for item in fetched_indices:
        #     type_flag = item
        #     break
        
        for batch_index, indices_distances in zip(self.scoped_indices, fetched_indices): 
            if isinstance(indices_distances, tuple):
                indices, _ = indices_distances
            else:
                indices = indices_distances
            mask = indices !=  -1 #indices中不等于-1的为true，等于-1的为false
            if self.is_knn_gpu is True:
                db_indices = torch.where(mask, indices, torch.tensor(0))
                all_masks.append(mask.to(device))
                key_values = self.db[batch_index, db_indices.cpu().numpy() % self.max_memories] #来得到之前存储的key value
            else:
                db_indices = np.where(mask, indices, 0) #当mask 为true的时候，对应位置为incise的值，如果为fale就为0
                all_masks.append(torch.from_numpy(mask))
                key_values = self.db[batch_index, db_indices % self.max_memories] #来得到之前存储的key value
                
            # batch_index是指knn的层数中的第 batch_index层中来，映射返回值。/ 确保检索的会在，可以检索的范围。
            
        
            all_key_values.append(torch.from_numpy(key_values))

        # 
        if self.is_knn_gpu is True:
            all_masks = torch.stack(all_masks).to(device)
            all_key_values = torch.stack(all_key_values).to(device)
        else:
            all_masks = torch.stack(all_masks)
            all_key_values = torch.stack(all_key_values)
        
        all_key_values = all_key_values.masked_fill(~rearrange(all_masks, '... -> ... 1 1'), 0.)

        all_key_values, = unpack(all_key_values, ps, 'b * n kv d')
        
        all_masks, = unpack(all_masks, ps, 'b * n')
        
        all_key_values = all_key_values.to(device)
        all_masks = all_masks.to(device)

        return all_key_values, all_masks

    def __del__(self):
        if hasattr(self, 'knns'):
            for knn in self.knns:
                del knn
        del self.db

# extends list with some extra methods for collections of KNN memories

class KNNMemoryList(list):
    def cleanup(self):
        for memory in self:
            del memory

    @classmethod
    def create_memories(
        self,
        *,
        my_config,
        mode,
        batch_size,
        num_memory_layers, # 有多少层要加入KNN memory
        memories_directory = DEFAULT_KNN_MEMORY_MEMMAP_DIRECTORY,
        max_memories,
        is_knn_gpu = True
    ):  
        memories_path = Path(memories_directory)
        memories_path.mkdir(exist_ok = True, parents = True)

        # if mode == 'train':
        #     is_knn_gpu = False
        # else:
        #     is_knn_gpu = True
        def inner(*args, **kwargs):
            return self([KNNMemory(*args, my_config=my_config, mode=mode,max_memories=max_memories, is_knn_gpu=is_knn_gpu ,num_indices = batch_size, memmap_filename = str(memories_path / f'knn.memory.layer.memmap.{mode}'), **kwargs) for ind in range(num_memory_layers)])
        return inner

    @contextmanager
    def at_batch_indices(
        self,
        indices
    ):
        knn_batch_indices_contexts = [memory.at_batch_indices(indices) for memory in self]
        with multi_context(*knn_batch_indices_contexts):
            yield

    def clear_memory(
        self,
        batch_indices = None,
        memory_indices = None
    ):
        memory_indices = default(memory_indices, tuple(range(len(self))))

        for memory_index in memory_indices:
            memory = self[memory_index]
            memory.clear(batch_indices)
            
    def read_offline_db(
        self,
        batch_indices = None,
        db_index_list=None,
        memory_indices = None
    ):
        memory_indices = default(memory_indices, tuple(range(len(self))))

        for memory_index in memory_indices:
            memory = self[memory_index]
            memory.offline_add(batch_indices, db_index_list)
