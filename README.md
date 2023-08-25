# context

### 复现须修改：
修改pwd，
修改cer的计算方法

### 实现：
检查.sh中的内容
修改knn_memories_directory


datastore的存储



requirement:
typed-argument-parser



### 训练流程

1. 训练baseline模型
1.1 训练sliding K的模型
    K = 2 4 6 8 要根据max length 和512的关系来计算。
2. 训练 knn 模型
3. 结合 ckpt 训练 knn 模型
4. 结合ckpt的 datastore 训练 offline -knn
4. 结合ckpt的 datastore 训练 offline -ckpt -knn
5. 结合ckpt的 datastore 训练 offline -ckpt -knn -domain db


knn的参数调整：
在offline online 最优的实验设置上，分别来进行 K 的调参 和 gate的调参数。
layer_num 的影响。