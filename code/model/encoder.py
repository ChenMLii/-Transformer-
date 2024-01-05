import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import clones
from model.sublayer import SublayerConnection, LayerNorm


class Encoder(nn.Module):
    "Encoder就是N个SubLayer的stack，最后加上一个LayerNorm。"
    # layer = EncoderLayer
    # N = 6
    def __init__(self, layer, N):
        super(Encoder, self).__init__()# 调用父类(超类)的一个方法(init)。
        # layer是一个SubLayer，我们clone 6 个
        self.layers = clones(layer, N)
        # 再加一个LayerNorm层
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        "逐层进行处理"
        for layer in self.layers:
            x = layer(x, mask)
        # 最后进行LayerNorm
        return self.norm(x)


class EncoderLayer(nn.Module):
    "EncoderLayer由self-attn和feed forward组成"
    # 为了复用，这里的self_attn层和feed_forward层也是传入的参数，这里只构造两个SublayerConnection。
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn  # self_attn函数需要4个参数(Query的输入,Key的输入,Value的输入和Mask)
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)  # 自注意层和前向层都需要进行norm+dropout+add
        # d_model
        self.size = size
    
    # self_attn有4个参数，但是我们知道在Encoder里，前三个参数都是输入y，第四个参数是mask。
    # 这里mask是已知的，因此我们可以用lambda的技巧它变成一个参
    # 数的函数z = lambda y: self.self_attn(y, y, y, mask)，这个函数的输入是y。
    def forward(self, x, mask):
        z = lambda x: self.self_attn(x, x, x, mask)
        x = self.sublayer[0](x, z)
         # self.sublayer[0]是个callable，self.sublayer[0] (x, z)会调用self.sublayer[0].call(x, z)，
        # 然后会调用SublayerConnection.forward(x, z)，
        # 然后会调用sublayer(self.norm(x))，sublayer就是传入的参数z，因此就是z(self.norm(x))。
        return self.sublayer[1](x, self.feed_forward)