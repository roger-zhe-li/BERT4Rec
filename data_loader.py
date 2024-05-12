import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

class BERTDataLoader(TensorDataset):
    def __init__(self, n_item, max_len, mask_prob, neg_sample_size, data):
        self.max_len = max_len
        self.mask_prob = mask_prob
        self.data = data
        self.n_item = n_item
        self.n_user = data['user'].count()
        self.neg_sample_size = neg_sample_size
        # mask pad tokens in attention.
        self.pad_token = n_item + 1
        # mask for training: token = n_item  (real items range from 0 to n_item - 1, after considering padding it should be 1 to n_item)
        self.mask_token = n_item


    def get_train(self):
        n_user = self.n_user
        mask_prob = self.mask_prob
        mask_token = self.mask_token
        pad_token = self.pad_token
        max_len = self.max_len

        user_index = list(range(n_user))
        tokens, labels = [], []

        for index in user_index:
            token, label = [], []
            user = self.data['user'][index]
            seq = self.data['movies'][index]

            # in training, item ids should start from from 1
            for item in seq[:-2]:
                prob = np.random.rand()
                if prob < mask_prob:
                    token.append(mask_token)
                else:
                    token.append(item)
                label.append(item)

            token = token[-max_len: ]
            label = label[-max_len: ]
            pad_len = max_len - len(token)

            token = [pad_token] * pad_len + token
            label = [pad_token] * pad_len + label

            tokens.append(token)
            labels.append(label)

        return torch.LongTensor(tokens), torch.LongTensor(labels)


    def get_valid(self):
        n_user = self.n_user
        mask_prob = self.mask_prob
        mask_token = self.mask_token
        pad_token = self.pad_token
        max_len = self.max_len
        neg_sample_size = self.neg_sample_size

        user_index = list(range(n_user))
        tokens, candidates, labels = [], [], []

        for index in tqdm(user_index):
            token, candidate, label = [], [], []
            user = self.data['user'][index]
            seq = self.data['movies'][index]

            # test sample is the last but one token in a sequence
            for item in seq[:-2]:
                token.append(item)
            token.append(mask_token)
            pad_len = max_len - len(token)

            candidate.append(seq[-2])

            # random negative sampling
            # random sample from all items except the real one, even samples appeared before in the seq are considered for another watching
            neg_pool = user_index[: index] + user_index[index+1: ]
            neg_samples = np.random.choice(neg_pool, size=neg_sample_size, replace=False).tolist()
            candidate = candidate + neg_samples

            token = [pad_token] * pad_len + token

            label = [1] + [0] * neg_sample_size

            tokens.append(token)
            candidates.append(candidate)
            labels.append(label)

        return torch.LongTensor(tokens), torch.LongTensor(candidates), torch.LongTensor(labels)


    def get_test(self):
        n_user = self.n_user
        mask_prob = self.mask_prob
        mask_token = self.mask_token
        pad_token = self.pad_token
        max_len = self.max_len
        neg_sample_size = self.neg_sample_size

        user_index = list(range(n_user))
        tokens, candidates, labels = [], [], []

        for index in tqdm(user_index):
            token, candidate, label = [], [], []
            user = self.data['user'][index]
            seq = self.data['movies'][index]

            # test sample is the last token in a sequence
            for item in seq[:-1]:
                token.append(item)
            token.append(mask_token)
            pad_len = max_len - len(token)
            candidate.append(seq[-1])

            # random negative sampling
            # random sample from all items except the real one, even samples appeared before in the seq are considered for another watching
            neg_pool = user_index[: index] + user_index[index+1: ]
            neg_samples = np.random.choice(neg_pool, size=neg_sample_size, replace=False).tolist()
            candidate = candidate + neg_samples

            token = [pad_token] * pad_len + token

            label = [1] + [0] * neg_sample_size

            tokens.append(token)
            candidates.append(candidate)
            labels.append(label)

        return torch.LongTensor(tokens), torch.LongTensor(candidates), torch.LongTensor(labels)
