# Copyright (c) 2018-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import torch

def shredFacts(facts, hiss, raw_facts=None): #takes a batch of facts and shreds it into its columns
        
    heads      = torch.tensor(facts[:,0]).long().cuda()
    rels       = torch.tensor(facts[:,1]).long().cuda()
    tails      = torch.tensor(facts[:,2]).long().cuda()
    years = torch.tensor(facts[:,3]).float().cuda()
    months = torch.tensor(facts[:,4]).float().cuda()
    days = torch.tensor(facts[:,5]).float().cuda()
    yearsid = torch.tensor(raw_facts[:,6]).long().cuda()
    monthsid = torch.tensor(raw_facts[:,7]).long().cuda()
    daysid = torch.tensor(raw_facts[:,8]).long().cuda()
    hiss = torch.tensor(hiss).long().cuda()
    return heads, rels, tails, years, months, days, yearsid, monthsid, daysid, hiss
