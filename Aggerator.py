import torch
import torch.nn as nn
import torch.nn.functional as F

class mutillevelAttention(nn.Module):
    def __init__(self, few, embed_dim, dropout=0.5):
        super(mutillevelAttention, self).__init__()
        self.few_shots = few
        self.embed_dim = embed_dim

        self.proj = nn.Linear(self.embed_dim * 4, self.embed_dim)
        self.enhance_lstm = nn.LSTM(self.embed_dim, self.embed_dim, 1, batch_first=True, bidirectional=True)

        if (self.few_shots % 2) != 0:
            kerns = self.few_shots
        else:
            kerns = self.few_shots - 1

        self.conv1 = nn.Conv2d(1, 32, (kerns, 1), padding=(kerns // 2, 0))
        self.conv2 = nn.Conv2d(32, 64, (kerns, 1), padding=(kerns // 2, 0))
        self.conv_final = nn.Conv2d(64, 1, (self.few_shots, 1), stride=(kerns, 1))

        self.dropout = nn.Dropout(dropout)

        self.enhance_w = nn.Linear(self.embed_dim * 4, self.embed_dim)
        self.enhance_u = nn.Linear(self.embed_dim * 4, self.embed_dim)

        self.pair_dropout = nn.Dropout(0.2)

    def forward(self, support, query):
        """
        :param support: [few_shot, 2, dim]
        :param query: [batch_size, 2, dim]
        :return:
        """
        batch_size = query.size(0)

        support = support.view(1, self.few_shots * 2, self.embed_dim).expand(batch_size, self.few_shots * 2,
                                                                             self.embed_dim).contiguous()

        enhance_support, enhance_query = self.local_matching(support, query)

        # reduce dimensionality
        enhance_support = self.proj(enhance_support)
        enhance_query = self.proj(enhance_query)
        enhance_support = F.leaky_relu(enhance_support)
        enhance_query = F.leaky_relu(enhance_query)

        enhance_support = enhance_support.view(batch_size, 2 * self.few_shots, self.embed_dim)
        enhance_query = enhance_query.view(batch_size, 2, self.embed_dim)

        support_lstm, _ = self.enhance_lstm(enhance_support)  # 123 * 6 * 200
        query_lstm, _ = self.enhance_lstm(enhance_query)  # 123 * 2 * 200

        # Local aggregation: max & avg
        enhance_support, enhance_query = self.local_aggregation(support_lstm, query_lstm)
        # enhance_support size:torch.Size([128, 3, 100])
        # enhance_query size:torch.Size([128, 100])

        entity_pair_att = enhance_support @ enhance_query.unsqueeze(1).transpose(1, 2)
        entity_pair_att = self.pair_dropout(F.softmax(entity_pair_att, 1))
        one_enhance_support = (
                enhance_support.transpose(1, 2) @ entity_pair_att).squeeze()  # torch.Size([128, 100])

        fea_att_score = enhance_support.view(batch_size, 1, self.few_shots, self.embed_dim * 1)
        fea_att_score = F.relu(self.conv1(fea_att_score))  
        fea_att_score = F.relu(self.conv2(fea_att_score)) 
        fea_att_score = self.dropout(fea_att_score)
        fea_att_score = self.conv_final(fea_att_score) 
        fea_att_score = F.relu(fea_att_score)
        fea_att_score = fea_att_score.view(batch_size, self.embed_dim * 1)  # torch.Size([128, 100])

        matching_scores = (fea_att_score * (torch.pow(one_enhance_support - enhance_query, 2))).sum(dim=1)
        matching_scores = - matching_scores

        rel_loss = (torch.pow(enhance_support - one_enhance_support.unsqueeze(1), 2)).sum(dim=2).mean(dim=1)

        return matching_scores, rel_loss.mean()

    def local_matching(self, support, query):
        support_, query_ = self.CoAttention(support, query)
        enhance_query = self.fuse(query, query_, 2)
        enhance_support = self.fuse(support, support_, 2)

        return enhance_support, enhance_query

    def CoAttention(self, support, query):
        # support: torch.Size([128, 6, 100])
        # query.transpose(1, 2): torch.Size([128, 100, 2])
        att = support @ query.transpose(1, 2)
        # att: torch.Size([128, 6, 2])
        support_ = F.softmax(att, 2) @ query
        query_ = F.softmax(att.transpose(1, 2), 2) @ support
        # support_:torch.Size([128, 6, 100])
        # query_:torch.Size([128, 2, 100])
        return support_, query_

    def fuse(self, m1, m2, dim):
        return torch.cat([m1, m2, torch.abs(m1 - m2), m1 * m2], dim)

    def local_aggregation(self, enhance_support, enhance_query):

        max_enhance_query, _ = torch.max(enhance_query, dim=1)
        mean_enhance_query = torch.mean(enhance_query, dim=1)
        enhance_query = torch.cat([max_enhance_query, mean_enhance_query], 1)
        enhance_query = self.enhance_w(enhance_query)

        enhance_support = enhance_support.view(enhance_support.size(0), self.few_shots, 2,
                                               self.embed_dim * 2)

        max_enhance_support, _ = torch.max(enhance_support, dim=2)
        mean_enhance_support = torch.mean(enhance_support, dim=2)
        enhance_support = torch.cat([max_enhance_support, mean_enhance_support], 2)
        enhance_support = self.enhance_u(enhance_support)

        return enhance_support, enhance_query
