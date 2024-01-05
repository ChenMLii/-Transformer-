
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import clones
from model.sublayer import SublayerConnection, LayerNorm


class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
         # layer是一个SubLayer，我们clone N个
        self.layers = clones(layer, N)
        # 再加一个LayerNorm层
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        "逐层进行处理"
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        # 最后进行LayerNorm，后面会解释为什么最后还有一个LayerNorm。
        return self.norm(x)


class DecoderLayer(nn.Module):
    "Decoder包括self-attn, src-attn, 和feed forward "
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn  # 比EncoderLayer多了一个src-attn层。
        # 这是Decoder时attend to Encoder的输出(memory)。src-attn和self-attn的实现是一样的，
        # 只不过使用的Query，Key和Value的输入不同。
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):  # 多一个来自Encoder的memory
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))  # self-attention
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))  # encoder-decoder attention
        return self.sublayer[2](x, self.feed_forward)