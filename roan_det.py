# Copyright (c) 2018-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from params import Params
from dataset import Dataset
from relation_emb import Rel_time_emb

class RoAN_DET(torch.nn.Module):
    
    def __init__(self, dataset, params):
        super(RoAN_DET, self).__init__()
        self.dataset = dataset
        self.params = params
        
        self.ent_embs      = nn.Embedding(dataset.numEnt(), params.s_emb_dim).cuda()
        self.rel_embs      = nn.Embedding(dataset.numRel(), params.emb_dim).cuda()
        
        self.create_time_embedds()
        
        self.time_nl = torch.sin
        
        nn.init.xavier_uniform_(self.ent_embs.weight)
        nn.init.xavier_uniform_(self.rel_embs.weight)
        
        self.sigm = torch.nn.Sigmoid()
        self.tanh = nn.Tanh()
    
    def create_time_embedds(self):
            
        self.m_freq = nn.Embedding(self.dataset.numEnt(), self.params.t_emb_dim).cuda()
        self.d_freq = nn.Embedding(self.dataset.numEnt(), self.params.t_emb_dim).cuda()
        self.y_freq = nn.Embedding(self.dataset.numEnt(), self.params.t_emb_dim).cuda()

        nn.init.xavier_uniform_(self.m_freq.weight)
        nn.init.xavier_uniform_(self.d_freq.weight)
        nn.init.xavier_uniform_(self.y_freq.weight)

        self.m_phi = nn.Embedding(self.dataset.numEnt(), self.params.t_emb_dim).cuda()
        self.d_phi = nn.Embedding(self.dataset.numEnt(), self.params.t_emb_dim).cuda()
        self.y_phi = nn.Embedding(self.dataset.numEnt(), self.params.t_emb_dim).cuda()

        nn.init.xavier_uniform_(self.m_phi.weight)
        nn.init.xavier_uniform_(self.d_phi.weight)
        nn.init.xavier_uniform_(self.y_phi.weight)

        self.m_amp = nn.Embedding(self.dataset.numEnt(), self.params.t_emb_dim).cuda()
        self.d_amp = nn.Embedding(self.dataset.numEnt(), self.params.t_emb_dim).cuda()
        self.y_amp = nn.Embedding(self.dataset.numEnt(), self.params.t_emb_dim).cuda()

        nn.init.xavier_uniform_(self.m_amp.weight)
        nn.init.xavier_uniform_(self.d_amp.weight)
        nn.init.xavier_uniform_(self.y_amp.weight)

        self.Rel_emb = Rel_time_emb(self.dataset, self.params)


    def get_time_embedd(self, entities, year, month, day):
        
        y = self.y_amp(entities)*self.time_nl(self.y_freq(entities)*year + self.y_phi(entities))
        m = self.m_amp(entities)*self.time_nl(self.m_freq(entities)*month + self.m_phi(entities))
        d = self.d_amp(entities)*self.time_nl(self.d_freq(entities)*day + self.d_phi(entities))
        
        return y+m+d

    def getEmbeddings(self, batch, ent_type, train_or_test):
        heads, rels, tails, years, months, days, yearsid, monthsid, daysid, hiss = batch
        years = years.view(-1,1)
        months = months.view(-1,1)
        days = days.view(-1,1)

        h,r,t = self.ent_embs(heads), self.rel_embs(rels), self.ent_embs(tails)
        
        h_t = self.get_time_embedd(heads, years, months, days)
        t_t = self.get_time_embedd(tails, years, months, days)
        
        h = torch.cat((h,h_t), 1)
        t = torch.cat((t,t_t), 1)

        pre_rel_emb = self.Rel_emb.getRelEmbeddings(batch, ent_type, train_or_test)
        r = (1-self.params.alp)*r + self.params.alp*pre_rel_emb

        return h,r,t
    
    #def forward(self, heads, rels, tails, years, months, days):
    def forward(self, batch1, batch2=None, train_or_test="train", ent_type="subs"):
        if batch2 == None:
            h_embs, r_embs, t_embs = self.getEmbeddings(batch1, ent_type, train_or_test)
            scores = h_embs + r_embs - t_embs
        else:
            h_embs, r_embs, t_embs = self.getEmbeddings(batch1, "objs", train_or_test)
            scores1 = h_embs + r_embs - t_embs

            h_embs, r_embs, t_embs = self.getEmbeddings(batch2, "subs", train_or_test)
            scores2 = h_embs + r_embs - t_embs

            scores = torch.cat((scores1, scores2), 0)
            
        scores = F.dropout(scores, p=self.params.dropout, training=self.training)
        scores = -torch.norm(scores, dim = 1)
        return scores
        
