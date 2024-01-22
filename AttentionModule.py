import numpy as np
import torch
import torch.nn as nn
import logging
import torch.nn.init as init
import torch.nn.functional as F



class Attention(nn.Module):
    def __init__(self, embed_dim):
        super(Attention, self).__init__()

        self.neigh_att_v = nn.Linear(2 * embed_dim, embed_dim, bias=False)
        self.neigh_att_k = nn.Linear(2 * embed_dim, embed_dim, bias=False)
        self.neigh_att_q = nn.Linear(2 * embed_dim, embed_dim, bias=False)

        self.dropout = nn.Dropout(0.3)
        self.dense = nn.Linear(embed_dim, embed_dim)
        self.LayerNorm = nn.LayerNorm(embed_dim, eps=1e-12)

        self.neigh_att_u = nn.Linear(embed_dim, 1, bias=False)

    def forward(self, rel, tail, mask=None):
        concat = torch.cat((rel, tail), dim=-1)  # (batch, max, 2*embed_dim)
        value = self.neigh_att_v(concat)
        key = self.neigh_att_k(concat)
        query = self.neigh_att_q(concat)
        scalar = tail.size(2) ** -0.5  # dim ** -0.5

        score = torch.matmul(key, query.transpose(1, 2)) * scalar

        score = score.masked_fill_(mask, -1e9)
        score = torch.softmax(score, dim=-1)
        score = self.dropout(score)
        out = torch.matmul(score, value)

        return out


class Attention_Module(nn.Module):
    def __init__(self, args, embed, num_symbols, embedding_size, use_pretrain=True, finetune=True, dropout_rate=0.3):
        super(Attention_Module, self).__init__()

        self.args = args
        self.embedding_size = embedding_size
        self.pad_idx = num_symbols
        self.pad_tensor = torch.tensor([self.pad_idx], requires_grad=False).to(args.device)

        self.symbol_emb = nn.Embedding(num_symbols + 1, self.embedding_size, padding_idx=self.pad_idx)
        if use_pretrain:
            logging.info('LOADING KB EMBEDDINGS')
            self.symbol_emb.weight.data.copy_(torch.from_numpy(embed))
            if not finetune:
                logging.info('FIX KB EMBEDDING')
                self.symbol_emb.weight.requires_grad = False

        self.dropout = nn.Dropout(dropout_rate)

        self.LeakyRelu = nn.LeakyReLU()

        self.layer_norm = nn.LayerNorm(self.embedding_size)

        self.Linear_tail = nn.Linear(self.embedding_size, self.embedding_size, bias=False)  #
        self.Linear_head = nn.Linear(self.embedding_size, self.embedding_size, bias=False)

        self.rel_w = nn.Bilinear(self.embedding_size, self.embedding_size, 2 * self.embedding_size, bias=False)  # 2*


        self.layer_norm1 = nn.LayerNorm(2 * self.embedding_size)

        self.attentions = [Attention(self.embedding_size) for _ in range(self.args.multi_head)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)  # 加入pytorch的Module模块

        self.attention_w = nn.Linear(self.args.multi_head * self.embedding_size, self.embedding_size, bias=False)


    def forward(self, entity_pairs, entity_meta):
        entity = self.dropout(self.symbol_emb(entity_pairs))  # (few/b, 2, dim)
        entity_left, entity_right = torch.split(entity, 1, dim=1)  # (few/b, 1, dim)
        entity_left = entity_left.squeeze(1)  # (few/b, dim)
        entity_right = entity_right.squeeze(1)  # (few/b, dim)

        entity_left_connections, entity_left_degrees, entity_right_connections, entity_right_degrees = entity_meta

        relations_left = entity_left_connections[:, :, 0].squeeze(-1)
        entities_left = entity_left_connections[:, :, 1].squeeze(-1)
        rel_embeds_left = self.dropout(self.symbol_emb(relations_left))  # (few/b, max, dim)
        ent_embeds_left = self.dropout(self.symbol_emb(entities_left))  # (few/b, max, dim)

        pad_matrix_left = self.pad_tensor.expand_as(relations_left)
        mask_matrix_left = torch.eq(relations_left, pad_matrix_left).unsqueeze(2)  # (b, max)
        mask_matrix_left = mask_matrix_left | (mask_matrix_left.transpose(2, 1)) # (b, max, max)

        relations_right = entity_right_connections[:, :, 0].squeeze(-1)
        entities_right = entity_right_connections[:, :, 1].squeeze(-1)
        rel_embeds_right = self.dropout(self.symbol_emb(relations_right))  # (few/b, max, dim)
        ent_embeds_right = self.dropout(self.symbol_emb(entities_right))  # (few/b, max, dim)

        pad_matrix_right = self.pad_tensor.expand_as(relations_right)
        mask_matrix_right = torch.eq(relations_right, pad_matrix_right).unsqueeze(2)
        mask_matrix_right = mask_matrix_right | (mask_matrix_right.transpose(2, 1))

        nei_att_left = torch.cat([att(rel_embeds_left, ent_embeds_left, mask_matrix_left) for att in self.attentions],
                                 dim=-1)
        nei_att_left = self.attention_w(nei_att_left).sum(dim=1)
        nei_att_left = torch.relu(nei_att_left + self.Linear_head(entity_left))
        nei_att_left = self.layer_norm(nei_att_left + entity_left)

        nei_att_right = torch.cat(
            [att(rel_embeds_right, ent_embeds_right, mask_matrix_right) for att in self.attentions], dim=-1)
        nei_att_right = self.attention_w(nei_att_right).sum(dim=1)
        nei_att_right = torch.relu(nei_att_right + self.Linear_head(entity_right))
        nei_att_right = self.layer_norm(nei_att_right + entity_right)


        output = torch.stack((nei_att_left, nei_att_right), dim=1)


        return output
