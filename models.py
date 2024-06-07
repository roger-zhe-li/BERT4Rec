import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# @torchsnooper.snoop()
class MultiheadAttention(nn.Module):
    def __init__(self, n_item, emb_dim, num_head):
        super(MultiheadAttention, self).__init__()

        assert emb_dim % num_head == 0, 'emb_dim -- num_heads combination is invalid'

        self.emb_dim = emb_dim
        self.num_head = num_head
        self.n_item = n_item

        self.W_q = nn.Linear(emb_dim, emb_dim, bias=False)
        self.W_k = nn.Linear(emb_dim, emb_dim, bias=False)
        self.W_v = nn.Linear(emb_dim, emb_dim, bias=False)
        self.W_o = nn.Linear(emb_dim, emb_dim, bias=False)

        self.scale = math.sqrt(emb_dim / num_head)

    def forward(self, q, k, v, mask=None, dropout=None):
        batch_size = q.shape[0]
        n_item = self.n_item

        query = self.W_q(q)  # batch_size * seq_length * emb_dim
        key = self.W_k(k)
        value = self.W_v(v)

        # batch_size * seq_length * num_heads * dim_head
        query = query.view(batch_size, query.shape[1], self.num_head, self.emb_dim // self.num_head).permute(0, 2, 1, 3)
        key = key.view(batch_size, key.shape[1], self.num_head, self.emb_dim // self.num_head).permute(0, 2, 1, 3)
        value = value.view(batch_size, value.shape[1], self.num_head, self.emb_dim // self.num_head).permute(0, 2, 1, 3)

        # attention aim: batch_size * num_heads * seq_length * seq_length
        attention_logits = torch.matmul(query, key.permute(0, 1, 3, 2)) / self.scale
        if mask is not None:
            # set masked seq positions (False) as an outlier number, so that its attention is 0
            attention_logits = attention_logits.masked_fill(mask==0, -8964)

        if dropout is not None:
            attention_logits = nn.Dropout(p=dropout)(attention_logits)

        attention_logits = F.softmax(attention_logits, dim=-1)
        attention_value = torch.matmul(attention_logits, value).contiguous()

        # multi-head concatenation
        multihead_attention_concat = attention_value.view(batch_size, -1, self.emb_dim)

        # multi-head output
        multihead_attention = self.W_o(multihead_attention_concat)
        # print('attention done')
        return multihead_attention


# @torchsnooper.snoop()
class PFFN(nn.Module):
    def __init__(self, emb_dim, ff_dim):
        super(PFFN, self).__init__()
        self.emb_dim = emb_dim
        self.ff_dim = ff_dim

        self.FFN_1 = nn.Linear(emb_dim, ff_dim, bias=True)
        self.FFN_2 = nn.Linear(ff_dim, emb_dim, bias=True)

    def forward(self, x, dropout=None):
        output_1 = self.FFN_1(x)
        output_1 = nn.GELU()(output_1)

        if dropout is not None:
            output_1 = nn.Dropout(p=dropout)(output_1)
        output_2 = self.FFN_2(output_1)
        # print('pffn done')
        return output_2


# @torchsnooper.snoop()
class ResidualBlock(nn.Module):
    def __init__(self, emb_dim):
        super(ResidualBlock, self).__init__()
        self.layer_norm = nn.LayerNorm(emb_dim)

    def forward(self, layer, sublayer, dropout=0.2):
        assert dropout is not None, 'has to get a dropout layer'

        sublayer = nn.Dropout(p=dropout)(sublayer)
        output_add = layer + sublayer
        # layer normalization: for normalizing the features in one sample.
        # dim 0 is batch size, dim 1 is for sequence tokens, only dim 2 is for features
        output_ln = self.layer_norm(output_add)
        return output_ln


# @torchsnooper.snoop()
class TransformerEncoder(nn.Module):
    def __init__(self, n_item, emb_dim, ff_dim, num_head, dropout_trn, dropout_attention):
        super(TransformerEncoder, self).__init__()

        # variables
        self.emb_dim = emb_dim
        self.ff_dim = ff_dim
        self.num_head = num_head
        self.dropout_trn = dropout_trn
        self.dropout_attention = dropout_attention
        self.n_item = n_item

        # netword architectures
        self.multihead_attention_model = MultiheadAttention(n_item=n_item, emb_dim=emb_dim, num_head=num_head)
        self.pffn_model = PFFN(emb_dim=emb_dim, ff_dim=ff_dim)
        self.residual_mh_model = ResidualBlock(emb_dim=emb_dim)
        self.residual_pffn_model = ResidualBlock(emb_dim=emb_dim)

    def forward(self, x, mask=None):
        dropout_trn = self.dropout_trn
        dropout_attention = self.dropout_attention

        # residual block 1: multihead attention
        MH_out = self.multihead_attention_model(x, x, x, mask=mask, dropout=dropout_attention)
        MH_res_output = self.residual_mh_model(x, MH_out, dropout=dropout_trn)

        # residual block 2: pffn
        PFFN_out = self.pffn_model(MH_res_output, dropout=dropout_trn)
        PFFN_res_output = self.residual_pffn_model(MH_res_output, PFFN_out, dropout=dropout_trn)

        return PFFN_res_output


# @torchsnooper.snoop()
class BERTEmbeddings(nn.Module):
    def __init__(self, emb_dim, vocab_size, max_len):
        super(BERTEmbeddings, self).__init__()

        self.emb_dim = emb_dim
        # how many items in total
        self.vocab_size = vocab_size
        # max length in a sequence
        self.max_len = max_len
        # embeddings for items (tokens): vocab * emb_dim
        self.token_embeddings = nn.Embedding(vocab_size, emb_dim, padding_idx=vocab_size - 1)
        # embeddings for position: here trainable, not sinusoid
        self.position_embeddings = nn.Embedding(max_len, emb_dim)

    def forward(self, seq, dropout=0.1):
        batch_size, seq_length = seq.shape[0], seq.shape[1]
        # todo: to_device
        # each position is with a certain embedding, which is learnable
        position_ids = torch.arange(seq.shape[1], dtype=torch.long, device=seq.device).expand(batch_size, seq_length)

        # batch_size * seq_length * emb_dim
        token_embeddings = self.token_embeddings(seq)
        position_embeddings = self.position_embeddings(position_ids)

        # print(token_embeddings.shape, position_embeddings.shape)

        embeddings = token_embeddings + position_embeddings
        if dropout is not None:
            embeddings = nn.Dropout(p=dropout)(embeddings)
        # print('embeddings done')
        return embeddings


# @torchsnooper.snoop()
class BERT(nn.Module):
    def __init__(self, n_item, max_len, num_head, emb_dim, n_stack, ff_dim, dropout_trn, dropout_attention):
        super(BERT, self).__init__()
        self.n_item = n_item
        # 2 additional tokens in addition to the item list: mask and pad
        self.vocab_size = n_item + 2
        self.num_head = num_head
        self.emb_dim = emb_dim
        self.max_len = max_len
        self.n_stack = n_stack
        self.ff_dim = ff_dim
        self.dropout_trn = dropout_trn
        self.dropout_attention = dropout_attention

        self.embedding = BERTEmbeddings(emb_dim, self.vocab_size, max_len)
        self.transformer_encoders = nn.ModuleList(
            [
                TransformerEncoder(
                    n_item,
                    emb_dim,
                    ff_dim,
                    num_head,
                    dropout_trn,
                    dropout_attention
                )
                for _ in range(n_stack)
            ]
        )

        self.output_1 = nn.Linear(emb_dim, emb_dim, bias=True)
        self.output_2 = nn.Linear(emb_dim, n_item + 1, bias=True)  # pad token is not taken into account

    def forward(self, seq):
        dropout_trn = self.dropout_trn
        num_head = self.num_head
        n_item = self.n_item
        transformer_encoders = self.transformer_encoders
        batch_size, seq_length = seq.shape[0], seq.shape[1]

        # mask all padded tokens to make short seqs to the length of max_len
        # mask dim is identical with attention: [batch_size, num_head, seq_len, seq_len]
        # all other items in the seq should have no attention to the padded tokens
        # also apply to all heads
        mask = (seq != n_item + 1).unsqueeze(1).unsqueeze(-1).repeat(1, num_head, 1, seq_length)
        # print(sum(mask))

        seq_embedding = self.embedding(seq, dropout=dropout_trn)
        for transformer_encoder in transformer_encoders:
            seq_embedding = transformer_encoder(seq_embedding, mask=mask)

        # output seq_embbeding dim: batch_size * max_len * emb_dim

        seq_output_int = self.output_1(seq_embedding)
        seq_output_int = nn.GELU()(seq_output_int)
        seq_output = self.output_2(seq_output_int)

        return seq_output
