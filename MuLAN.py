import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math
from torch.autograd import Variable
from AttentionModule import Attention_Module
from Aggerator import *


class MuLAN(nn.Module):
    def __init__(self, args, num_symbols, embedding_size, embed, use_pretrain, finetune):
        super(MuLAN, self).__init__()

        self.args = args
        self.entity_encoder = Attention_Module(self.args, embed=embed, num_symbols=num_symbols,
                                               embedding_size=embedding_size,
                                               use_pretrain=use_pretrain, finetune=finetune)

        self.relation_encoder = mutillevelAttention(few=self.args.few, embed_dim=embedding_size)

    def forward(self, support, support_meta, query, query_meta,
                false=None, false_meta=None, is_train=True):

        if is_train:

            support_rep = self.entity_encoder(support, support_meta)
            query_rep = self.entity_encoder(query, query_meta)
            false_rep = self.entity_encoder(false, false_meta)

            positive_score, rel_loss = self.relation_encoder(support_rep, query_rep)
            negative_score, _ = self.relation_encoder(support_rep, false_rep)

        else:
            support_rep = self.entity_encoder(support, support_meta)
            query_rep = self.entity_encoder(query, query_meta)
            positive_score, _ = self.relation_encoder(support_rep, query_rep)  # (1, emb_dim) (128,100)

            negative_score = None
            rel_loss = None

        return positive_score, negative_score, rel_loss
