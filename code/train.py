import time

import torch

from parser_1 import args
from lib.loss import SimpleLossCompute
import matplotlib.pyplot as plt
import numpy as np

def run_epoch(data, model, loss_compute, epoch):
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0

    for i , batch in enumerate(data):

        out = model(batch.src, batch.trg, batch.src_mask, batch.trg_mask)
       
        loss = loss_compute(out, batch.trg_y, batch.ntokens)

        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens

        if i % 50 == 1:
            elapsed = time.time() - start
            print("Epoch %d Batch: %d Loss: %f Tokens per Sec: %fs" % (epoch, i - 1, loss / batch.ntokens, tokens / elapsed / 1000))
            start = time.time()
            tokens = 0

    return total_loss / total_tokens


def train(data, model, criterion, optimizer):
    list_loss, list_epoch=[],[]
    for epoch in range(args.epochs):
        model.train()
        run_epoch(data.train_data, model, SimpleLossCompute(model.generator, criterion, optimizer), epoch)

        model.eval()

        print('>>>>> Evaluate')
        loss = run_epoch(data.dev_data, model, SimpleLossCompute(model.generator, criterion, None), epoch)
        print('<<<<< Evaluate loss: %f' % loss)

    torch.save(model.state_dict(), args.save_file)
