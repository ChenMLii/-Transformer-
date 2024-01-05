import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    # 根据Decoder的隐状态输出一个词，Decoder的后面两步（linear+softmax)
    # d_model是Decoder输出的大小，vocab是词典大小
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)# 全连接层进行线性变换

    # 全连接再加上一个softmax
    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)# 按照指定维度在softmax基础上再log