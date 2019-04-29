import torch
import math
from torch import nn
from torch.nn import init, functional
import numpy as np


class ChannelModel(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()
        self.se_dim = kwargs['se_dim']  # dim of sentence embedding
        self.p_producer = nn.Sequential(
            nn.Dropout(kwargs['dropout']),
            nn.Linear(self.se_dim * 3, 1024),   # Chú ý hệ số 3 (xem d_s_feat trong hàm forward)
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        self.temperature = 1  # for annealing
        self.reset_parameters()

    def reset_parameters(self):
        for name, param in list(self.p_producer.named_parameters()):
            if 'bias' in name:
                init.constant_(param, 0.0)
            elif 'weight' in name:
                init.kaiming_normal_(param)

    def forward(self, D, S):  # TODO batch with torch.bmm
        '''
        To calculate P(D|S) given the sentence embeddings of D and S. 
        D: [n, Ds], where n is # sentences of document
        S: [m, Ds], where m is # sentences of summary
        Ds là embedding_dim của sentence
        '''
        n, m = D.size(0), S.size(0)
        S_T = torch.transpose(S, 0, 1).contiguous()  # [Ds, m]
        att_weight = functional.softmax(
            torch.mm(D, S_T) / self.temperature,    # size = (n,m)
            dim=1
        )  # [n, m]
        att_S = torch.mm(att_weight, S)  # [n, Ds], với mỗi câu d_i trong n câu, ta có vector tổ hợp từ m câu tóm tắt

        ## --------------------------------------------------------------- ##
        d_s_feat = torch.cat([D, att_S, torch.mul(D, att_S)], dim=1)
        prob_vector = self.p_producer(d_s_feat)  # P(d_i|S) in each row, size = [n,1]
        ## --------------------------------------------------------------- ##

        # P(D|S) on the independent assumption. TODO: 0.5 trick not elegant;
        log_prob = torch.sum(torch.log(0.5 + prob_vector))  # scalar
        addition = {
            'prob_vector': prob_vector,
            'att_weight': att_weight,
            'att_S': att_S
        }
        return log_prob, addition
