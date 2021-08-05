'''参数加载'''
'''
Config配置文件（.ini文件），格式为：
[section]
item=value
每个section可以有多个items
'''
from configparser import ConfigParser
config = ConfigParser() #实例化
#常用函数：
config.read(filename) #读取ini文件
config.sections() #返回所有section，列表形式
config.items(section) #返回section内的所有key和value 字典形式

'''json文件，格式类似字典，例如{'item':value,'item2':{'key1':'value1','key2':'value2'}}'''
import json
jsondata=json.dumps(data) # 字典转json
data=json.loads(jsondata) # 读json数据
#从磁盘中的json文件中读取
from collections import OrderedDict
def read_json(filename):
    with filename.open('rt') as read:
        return json.load(read, object_hook=OrderedDict)


#argparse:命令行参数工具
import argparse
parser = argparse.ArgumentParser(description= )
parser.add_argument('test','-test')  # 加入参数，可选参数前加‘--’或‘-’，必选参数不加
args=parser.parse_args()

#config.py文件框架例子
from configparser import ConfigParser
class Config(ConfigParser):
    def __init__(self, config_file):
        raw_config = ConfigParser()#初始化
        raw_config.read(config_file)#读config文件
        self.cast_values(raw_config)

    def cast_values(self, raw_config):
        for section in raw_config.sections():
            for key, value in raw_config.items(section):
                '''具体操作'''



'''数据加载'''
'''
utils文件中设置读取数据的格式，数据预处理，较复杂
'''
torch.distributed.barrier() #分布式训练时，保证只有主进程处理数据集，其他进程先隐藏起来

#数据转变为tensor
from torch.utils.data import TensorDataset
dataset=TensorDataset(features)#将所有特征tensors合并为数据集，之后通过torch.utils.data.DataLoader加载数据，参数features是数据的各个特征


'''日志'''
import logging
'''
logging有5种级别的日志，等级依次升高 DEBUG < INFO < WARNING < ERROR < CRITICAL，默认等级是warning，warning以上的才会显示
'''
logging.basicConfig(format=, level=, filename=) #level设置等级开关，filename输出日志到磁盘文件
logger=logging.getLogger(name=)

logger.info("  Num Epochs = %d", config.num_train_epochs) #  “格式”，内容


'''并行化'''
#数据加载，取样器:
from torch.utils.data import DataLoader, Randomsampler, Tensordataset
from torch.utils.data.distributed import DistributedSampler

if config.local_rank == -1: #判断是否并行化取样
    train_sampler = RandomSampler(train_dataset)
else:
    train_sampler = DistributedSampler(train_dataset) #分布式采样
train_dataloader = DataLoader(train_dataset,sampler=train_sampler, batch_size= ,collate_fn=) # 数据加载器

#单机多卡:
##DataParallel(DP)
import torch
model = torch.nn.DataParallel(model, device_ids=[])
'''
缺点：单进程控制多gpu，计算效率、gpu使用率低，性能低于DDP
'''

##DistributedDataParallel(DDP)
parser = argparse.ArgumentParser()
parser.add_argument('--local_rank')
args = parser.parse_args()
torch.cuda.set_device(args.local_rank)

torch.distributed.init_process_group(backend="nccl", world_size= ) # 初始化gpu通信方式
train_sampler = DistributedSampler(train_dataset)
train_dataloader = DataLoader(train_dataset,sampler=train_sampler, batch_size= ,pin_memory=) #sampler指定抽样方式，pin_memory指定是否开启页锁定内存
if config.local_rank != -1:# 判断是否并行化
    model=torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank) #建立并行化模型


'''训练'''
#gpu,cpu搬运
if config.local_rank == -1 or config.no_cuda:
        device = torch.device(
            "cuda" if torch.cuda.is_available() and not config.no_cuda else "cpu")  #单卡，或者禁用cuda
else:
    torch.cuda.set_device(config.local_rank)
    device = torch.device("cuda", args.local_rank)

'''训练时，用.to(device)函数选择gpu或cpu训练；或者直接用.cpu(),.cuda()函数调整 '''

#优化器,lr decay
##torch
from torch import optim
optimizer = optim.Adam(params=model.parameters(),lr= ) #sgd,adamw等
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=) # StepLR,MultiStepLR等衰减方式
scheduler1 = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=, eta_min=, last_epoch=) #余弦周期类，震荡，T_max：一个周期的迭代次数，eta_min：最小学习率

##transformers
from transformers import AdamW, get_linear_schedule_with_warmup
optimizer = AdamW(model.parameters(), lr=, eps=)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=, num_training_steps=)# get_constant_schedule_with_warmup等方法，warmup是先增后减

#训练过程
loss=0
for i in epochs:
    model.train()
    optimizer.zero_grad() # 梯度清零
    loss= #计算损失loss，较复杂

    if config.n_gpu > 1:
        loss = loss.mean()  # 多卡训练时

    loss.backward() # 反向传播
    optimizer.step() # 参数更新
    scheduler.step() #学习率衰减
    total_loss+=loss.item()


