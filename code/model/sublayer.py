import torch
import torch.nn as nn

from parser_1 import args

# 若特征间具有不同的值范围时，因此梯度更新时，会来回震荡，经过较长的时间才能达到局部最优值或全局最优值。
# 为了解决该模型问题，我们需要归一化数据，我们确保不同的特征具有相同的值范围，这样梯度下降可以很快的收敛。
class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


# 不管是Self-Attention还是全连接层，都首先是LayerNorm，然后是Self-Attention/Dense，
# 然后是Dropout，最后是残差连接。
# 构造norm+dropout+add，这里面有很多可以重用的代码，我们把它封装成SublayerConnection。
class SublayerConnection(nn.Module):
    """
    LayerNorm + sublayer(Self-Attenion/Dense) + dropout + 残差连接
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer): 
     # 这个方法需要两个参数，一个是输入Tensor，一个是一个callable，并且这个callable可以用一个参数来调用
        "sublayer是传入的参数，参考DecoderLayer，它可以当成函数调用，这个函数的有一个输入参数"
        return x + self.dropout(sublayer(self.norm(x)))