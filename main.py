import pandas as pd
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import tensorflow as tf # just for transforming tfrecord
from torch.utils.data import TensorDataset, DataLoader

import torchsnooper

from data_loader import BERTDataLoader
from data_proc import DataProc

from models import MultiheadAttention, PFFN, ResidualBlock, TransformerEncoder, BERT, BERTEmbeddings

no_cuda = False
cuda = not no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

def set_random_seed(state=1):
    # not fixed for numpy as the training samples need to be randomly sampled for multiple times
	gens = (torch.manual_seed, torch.cuda.manual_seed)
	for set_state in gens:
		set_state(state)

seed = 8964
set_random_seed(seed)


# Preprocessing
data_processer = DataProc()
df, n_item = data_processer.data_proc()

# Data Loader
dataloader = BERTDataLoader(n_item,
               max_len=50,
               mask_prob=0.2,
               neg_sample_size=100,
               data=df)

model_bert = BERT(n_item=n_item,
                  max_len=50,
                  num_head=2,
                  emb_dim=64,
                  n_stack=2,
                  ff_dim=256,
                  dropout_trn=0.1,
                  dropout_attention=0.2)

model_bert = model_bert.to(device)

loss_fn = nn.CrossEntropyLoss(ignore_index=n_item+1, reduction='none')
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model_bert.parameters()),
                             weight_decay=0.01,
                             lr=1e-2)

def train(data_loader, model, loss_fn, n_item, optimizer, device):
    for tokens, labels in tqdm(data_loader):
        tokens, labels = tokens.to(device), labels.to(device)
        seq_output = model(tokens)
        loss_all = loss_fn(seq_output.view(-1, seq_output.size(-1)), labels.view(-1))
        labels_flag = (tokens==n_item)
        loss_masked = loss_all * labels_flag.view(-1)
        loss_masked_mean = loss_masked.sum() / labels_flag.sum()
        print(loss_masked_mean)
        optimizer.zero_grad()
        loss_masked_mean.backward()
        optimizer.step()


# validation/test: nDCG@10 and RR@10 for evaluation
def valid_test(data_loader, model, n_user, device, k=10):
    model.eval()
    with torch.no_grad():
        ndcg_all, rr_all = 0, 0

        for tokens, candidates, labels in tqdm(data_loader):
            tokens, candidates, labels = tokens.to(device), candidates.to(device), labels.to(device)
            seq_output = model(tokens)
            # only check the last token in the sequence
            pred = seq_output[:, -1, :]
            # get the logits of candidate items
            recs = torch.gather(pred, 1, candidates)

            recs = recs.cpu()
            labels = labels.cpu()
            rank = recs.argsort(descending=True, dim=1)

            cut = rank[:, :k]
            hits = torch.gather(labels, 1, cut)
            position = torch.arange(2, 2+k)
            weights = 1 / torch.log2(position.float())
            ndcg = (hits.float() * weights).sum(1)
            rr = (hits.float() / torch.arange(1, 1+k)).sum(1)

            ndcg_all += ndcg.sum().item()
            rr_all += rr.sum().item()

        ndcg = ndcg_all / n_user
        rr = rr_all / n_user

    return ndcg, rr

##########################################################
            # START TRAINING
##########################################################
n_epoch = 20
# valid and test sets are fixed
tokens_valid, candidates_valid, labels_valid = dataloader.get_valid()
valid_tensor = TensorDataset(tokens_valid,
                             candidates_valid,
                             labels_valid)
valid_loader = DataLoader(valid_tensor, batch_size=64, shuffle=False)

tokens_test, candidates_test, labels_test = dataloader.get_test()
test_tensor = TensorDataset(tokens_test,
                             candidates_test,
                             labels_test)
test_loader = DataLoader(test_tensor, batch_size=64, shuffle=False)
# ndcg as the golden standard
ndcg_max = -np.inf
n_user = tokens_valid.shape[0]


for epoch in tqdm(range(n_epoch)):
    # training data preparation, generate new masked seqs each epoch
    tokens_train, labels_train = dataloader.get_train()
    train_tensor = TensorDataset(tokens_train,
                                labels_train)
    train_loader = DataLoader(train_tensor, batch_size=64, shuffle=True)

    print('Epoch: ' + str(epoch))

    # train
    train(train_loader, model_bert, loss_fn, n_item, optimizer, device)
    # valid
    ndcg_10_val, rr_10_val = valid_test(valid_loader, model_bert, n_user, device, k=10)
    if ndcg_10_val > ndcg_max:
        ndcg_max = ndcg_10_val
        ndcg_10_test, rr_10_test = valid_test(test_loader, model_bert, n_user, device, k=10)

# output test results
print('nDCG@10 = %.4f' % ndcg_10_test)
print('RR@10 = %.4f' % rr_10_test)