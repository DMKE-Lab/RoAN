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
from transformer import Encoder

class Rel_time_emb(torch.nn.Module):
    def __init__(self, dataset, params):
        super(Rel_time_emb, self).__init__()
        self.dataset = dataset
        self.params = params
        
        self.create_time_embedds()

        self.time_nl = torch.sin
        self.his_encoder = Encoder(self.params)
    
    def create_time_embedds(self):
        #Embeddings of relations
        self.h_map_emb = nn.Embedding(self.dataset.numEnt(), self.params.emb_dim).cuda()
        self.t_map_emb = nn.Embedding(self.dataset.numEnt(), self.params.emb_dim).cuda()
        self.rel_emb_h = nn.Embedding(self.dataset.numRel()+1, self.params.emb_dim).cuda()
        self.rel_emb_t = nn.Embedding(self.dataset.numRel()+1, self.params.emb_dim).cuda()
        self.rel_emb_q = nn.Embedding(self.dataset.numRel(), self.params.emb_dim).cuda()

        self.year_emb = nn.Embedding(self.dataset.numYear(), self.params.emb_dim).cuda()
        self.month_emb = nn.Embedding(self.dataset.numMonth(), self.params.emb_dim).cuda()
        self.day_emb = nn.Embedding(self.dataset.numDay(), self.params.emb_dim).cuda()

        nn.init.xavier_uniform_(self.h_map_emb.weight)
        nn.init.xavier_uniform_(self.t_map_emb.weight)
        nn.init.xavier_uniform_(self.rel_emb_h.weight)
        nn.init.xavier_uniform_(self.rel_emb_t.weight)
        nn.init.xavier_uniform_(self.rel_emb_q.weight)
        nn.init.xavier_uniform_(self.year_emb.weight)
        nn.init.xavier_uniform_(self.month_emb.weight)
        nn.init.xavier_uniform_(self.day_emb.weight)

    def getRelEmbeddings(self, batch, ent_type, train_or_test):
        if ent_type == "subs":
            heads, rels, tails, _, _, _, yearsid, monthsid, daysid, hiss = batch
            
            if train_or_test == "train":
                ents_for_rel, rels_his = heads.view(-1, (1 + self.params.neg_ratio))[:, 0], rels.view(-1, (1 + self.params.neg_ratio))[:, 0]
                ents_emb, rels_emb, rel_set_emb = self.h_map_emb(ents_for_rel), self.rel_emb_q(rels_his), self.rel_emb_h(hiss)
                pre_rel_emb = self.get_rel_embedds(ents_emb, rels_emb, rel_set_emb, yearsid, monthsid, daysid, hiss)
                pre_rel_emb = pre_rel_emb.unsqueeze(1)
                pre_rel_emb = pre_rel_emb.repeat(1, (1 + self.params.neg_ratio), 1).contiguous().view(-1, self.params.emb_dim)
            else:
                ents_for_rel, rels_his = heads.view(-1,1)[0], rels.view(-1,1)[0]
                if hiss.size(1) != 0:
                    ents_emb, rels_emb, rel_set_emb = self.h_map_emb(ents_for_rel), self.rel_emb_q(rels_his), self.rel_emb_h(hiss)
                    pre_rel_emb = self.get_rel_embedds(ents_emb, rels_emb, rel_set_emb, yearsid, monthsid, daysid, hiss)
                    pre_rel_emb = pre_rel_emb.repeat(rels.size(0), 1).contiguous().view(-1, self.params.emb_dim)
                else:
                    pre_rel_emb = torch.zeros([rels.size(0), self.params.emb_dim]).float().cuda()
        else:
            heads, rels, tails, _, _, _, yearsid, monthsid, daysid, hiss = batch
            
            if train_or_test == "train": 
                ents_for_rel, rels_his = tails.view(-1, (1 + self.params.neg_ratio))[:,0], rels.view(-1, (1 + self.params.neg_ratio))[:,0]
                ents_emb, rels_emb, rel_set_emb = self.t_map_emb(ents_for_rel), self.rel_emb_q(rels_his), self.rel_emb_t(hiss)
                pre_rel_emb = self.get_rel_embedds(ents_emb, rels_emb, rel_set_emb, yearsid, monthsid, daysid, hiss)
                pre_rel_emb = pre_rel_emb.unsqueeze(1)
                pre_rel_emb = pre_rel_emb.repeat(1, (1 + self.params.neg_ratio), 1).contiguous().view(-1, self.params.emb_dim)
            else:
                ents_for_rel, rels_his = tails.view(-1,1)[0], rels.view(-1,1)[0]
                if hiss.size(1) != 0:
                    ents_emb, rels_emb, rel_set_emb = self.t_map_emb(ents_for_rel), self.rel_emb_q(rels_his), self.rel_emb_t(hiss)
                    pre_rel_emb = self.get_rel_embedds(ents_emb, rels_emb, rel_set_emb, yearsid, monthsid, daysid, hiss)
                    pre_rel_emb = pre_rel_emb.repeat(rels.size(0), 1).contiguous().view(-1, self.params.emb_dim)
                else:
                    pre_rel_emb = torch.zeros([rels.size(0), self.params.emb_dim]).float().cuda()
        return pre_rel_emb

    def get_rel_embedds(self, ents_emb, rels_emb, rel_set_emb, yearsid, monthsid, daysid, historys):
        yearsid = yearsid.view(-1,1)
        monthsid = monthsid.view(-1,1)
        daysid = daysid.view(-1,1)

        pre_rel_emb, pre_attn = self.get_pre_embedd(ents_emb, rels_emb, rel_set_emb, yearsid, monthsid, daysid, historys)
        return pre_rel_emb

    def get_pre_embedd(self, ents_emb, rels_emb, rel_set_emb, years, months, days, historys):
        times_emb = self.get_timestamp_emb(years, months, days)
        mask_attn = self.get_mask(historys)
        pre_his_emb, pre_rels = self.make_his_emb(ents_emb, times_emb, rels_emb, rel_set_emb)
        pre_rel_emb, pre_attn = self.his_encoder(pre_rels, pre_his_emb, mask_attn)
        return pre_rel_emb, pre_attn

    def get_timestamp_emb(self, years, months, days):
        times_emb = self.year_emb(years) + self.month_emb(months) + self.day_emb(days)
        return times_emb

    def get_mask(self, his):
        assert isinstance(his, torch.Tensor)
        idx, len_q = self.dataset.numRel(), 1
        batch_size, len_k = his.size()
        mask_attn = his.data.eq(idx).unsqueeze(1)
        return mask_attn.expand(batch_size, len_q, len_k)
    
    def make_his_emb(self, ents, times, rels_emb, rel_set_emb):
        batch_size, size, dim = rel_set_emb.size()
        rels_emb = rels_emb.unsqueeze(1)
        ents = ents.unsqueeze(1)

        ents_new = ents.expand(batch_size, size, dim)
        times_new = times.expand(batch_size, size, dim)
        his_set_emb = torch.sum(rel_set_emb * ents_new, dim=2, keepdim=True) * times_new
        
        que_rels = torch.sum(rels_emb * ents, dim=2, keepdim=True) * times
        return his_set_emb, que_rels.view(batch_size, -1, self.params.emb_dim)
        
