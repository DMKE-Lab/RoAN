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

class RoAN_DES(torch.nn.Module):

    def __init__(self, dataset, params):
        super(RoAN_DES, self).__init__()
        self.dataset = dataset
        self.params = params
        
        self.ent_embs_h = nn.Embedding(dataset.numEnt(), params.s_emb_dim).cuda()
        self.ent_embs_t = nn.Embedding(dataset.numEnt(), params.s_emb_dim).cuda()
        self.rel_embs_f = nn.Embedding(dataset.numRel(), params.emb_dim).cuda()
        self.rel_embs_i = nn.Embedding(dataset.numRel(), params.emb_dim).cuda()
        
        self.create_time_embedds()

        self.time_nl = torch.sin
        
        nn.init.xavier_uniform_(self.ent_embs_h.weight)
        nn.init.xavier_uniform_(self.ent_embs_t.weight)
        nn.init.xavier_uniform_(self.rel_embs_f.weight)
        nn.init.xavier_uniform_(self.rel_embs_i.weight)
    
    def create_time_embedds(self):

        # frequency embeddings for the entities
        self.m_freq_h = nn.Embedding(self.dataset.numEnt(), self.params.t_emb_dim).cuda()
        self.m_freq_t = nn.Embedding(self.dataset.numEnt(), self.params.t_emb_dim).cuda()
        self.d_freq_h = nn.Embedding(self.dataset.numEnt(), self.params.t_emb_dim).cuda()
        self.d_freq_t = nn.Embedding(self.dataset.numEnt(), self.params.t_emb_dim).cuda()
        self.y_freq_h = nn.Embedding(self.dataset.numEnt(), self.params.t_emb_dim).cuda()
        self.y_freq_t = nn.Embedding(self.dataset.numEnt(), self.params.t_emb_dim).cuda()

        # phi embeddings for the entities
        self.m_phi_h = nn.Embedding(self.dataset.numEnt(), self.params.t_emb_dim).cuda()
        self.m_phi_t = nn.Embedding(self.dataset.numEnt(), self.params.t_emb_dim).cuda()
        self.d_phi_h = nn.Embedding(self.dataset.numEnt(), self.params.t_emb_dim).cuda()
        self.d_phi_t = nn.Embedding(self.dataset.numEnt(), self.params.t_emb_dim).cuda()
        self.y_phi_h = nn.Embedding(self.dataset.numEnt(), self.params.t_emb_dim).cuda()
        self.y_phi_t = nn.Embedding(self.dataset.numEnt(), self.params.t_emb_dim).cuda()

        # frequency embeddings for the entities
        self.m_amps_h = nn.Embedding(self.dataset.numEnt(), self.params.t_emb_dim).cuda()
        self.m_amps_t = nn.Embedding(self.dataset.numEnt(), self.params.t_emb_dim).cuda()
        self.d_amps_h = nn.Embedding(self.dataset.numEnt(), self.params.t_emb_dim).cuda()
        self.d_amps_t = nn.Embedding(self.dataset.numEnt(), self.params.t_emb_dim).cuda()
        self.y_amps_h = nn.Embedding(self.dataset.numEnt(), self.params.t_emb_dim).cuda()
        self.y_amps_t = nn.Embedding(self.dataset.numEnt(), self.params.t_emb_dim).cuda()

        nn.init.xavier_uniform_(self.m_freq_h.weight)
        nn.init.xavier_uniform_(self.d_freq_h.weight)
        nn.init.xavier_uniform_(self.y_freq_h.weight)
        nn.init.xavier_uniform_(self.m_freq_t.weight)
        nn.init.xavier_uniform_(self.d_freq_t.weight)
        nn.init.xavier_uniform_(self.y_freq_t.weight)

        nn.init.xavier_uniform_(self.m_phi_h.weight)
        nn.init.xavier_uniform_(self.d_phi_h.weight)
        nn.init.xavier_uniform_(self.y_phi_h.weight)
        nn.init.xavier_uniform_(self.m_phi_t.weight)
        nn.init.xavier_uniform_(self.d_phi_t.weight)
        nn.init.xavier_uniform_(self.y_phi_t.weight)

        nn.init.xavier_uniform_(self.m_amps_h.weight)
        nn.init.xavier_uniform_(self.d_amps_h.weight)
        nn.init.xavier_uniform_(self.y_amps_h.weight)
        nn.init.xavier_uniform_(self.m_amps_t.weight)
        nn.init.xavier_uniform_(self.d_amps_t.weight)
        nn.init.xavier_uniform_(self.y_amps_t.weight)

        self.Rel_emb = Rel_time_emb(self.dataset, self.params)

    def get_time_embedd(self, entities, years, months, days, h_or_t):
        if h_or_t == "head":
            emb  = self.y_amps_h(entities) * self.time_nl(self.y_freq_h(entities) * years  + self.y_phi_h(entities))
            emb += self.m_amps_h(entities) * self.time_nl(self.m_freq_h(entities) * months + self.m_phi_h(entities))
            emb += self.d_amps_h(entities) * self.time_nl(self.d_freq_h(entities) * days   + self.d_phi_h(entities))
        else:
            emb  = self.y_amps_t(entities) * self.time_nl(self.y_freq_t(entities) * years  + self.y_phi_t(entities))
            emb += self.m_amps_t(entities) * self.time_nl(self.m_freq_t(entities) * months + self.m_phi_t(entities))
            emb += self.d_amps_t(entities) * self.time_nl(self.d_freq_t(entities) * days   + self.d_phi_t(entities))
            
        return emb

    def getEmbeddings(self, batch, ent_type, train_or_test):
        heads, rels, tails, years, months, days, yearsid, monthsid, daysid, hiss = batch
        years = years.view(-1,1)
        months = months.view(-1,1)
        days = days.view(-1,1)
        h_embs1 = self.ent_embs_h(heads)
        r_embs1 = self.rel_embs_f(rels)
        t_embs1 = self.ent_embs_t(tails)
        h_embs2 = self.ent_embs_h(tails)
        r_embs2 = self.rel_embs_i(rels)
        t_embs2 = self.ent_embs_t(heads)
        
        h_embs1 = torch.cat((h_embs1, self.get_time_embedd(heads, years, months, days, "head")), 1)
        t_embs1 = torch.cat((t_embs1, self.get_time_embedd(tails, years, months, days, "tail")), 1)
        h_embs2 = torch.cat((h_embs2, self.get_time_embedd(tails, years, months, days, "head")), 1)
        t_embs2 = torch.cat((t_embs2, self.get_time_embedd(heads, years, months, days, "tail")), 1)

        pre_rel_emb = self.Rel_emb.getRelEmbeddings(batch, ent_type, train_or_test)
        #r_embs1 = torch.cat((r_embs1, pre_rel_emb), 1)
        #r_embs2 = torch.cat((r_embs2, pre_rel_emb), 1)

        r_embs1 = (1-self.params.alp)*r_embs1 + self.params.alp*pre_rel_emb
        r_embs2 = (1-self.params.alp)*r_embs2 + self.params.alp*pre_rel_emb
        
        return h_embs1, r_embs1, t_embs1, h_embs2, r_embs2, t_embs2
    
    def forward(self, batch1=None, batch2=None, train_or_test="train", ent_type="subs"):
        if batch2 == None:
            h_embs1, r_embs1, t_embs1, h_embs2, r_embs2, t_embs2 = self.getEmbeddings(batch1, ent_type, train_or_test)
            scores = ((h_embs1 * r_embs1) * t_embs1 + (h_embs2 * r_embs2) * t_embs2) / 2.0
        else:

            h_embs1, r_embs1, t_embs1, h_embs2, r_embs2, t_embs2 = self.getEmbeddings(batch1, "objs", train_or_test)
            scores1 = ((h_embs1 * r_embs1) * t_embs1 + (h_embs2 * r_embs2) * t_embs2) / 2.0

            h_embs3, r_embs3, t_embs3, h_embs4, r_embs4, t_embs4 = self.getEmbeddings(batch2, "subs", train_or_test)
            scores2 = ((h_embs3 * r_embs3) * t_embs3 + (h_embs4 * r_embs4) * t_embs4) / 2.0
        
            scores = torch.cat((scores1, scores2), 0)

        scores = F.dropout(scores, p=self.params.dropout, training=self.training)
        scores = torch.sum(scores, dim=1)
        return scores
