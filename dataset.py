# Copyright (c) 2018-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import random
import math
import copy
import time
import numpy as np
from random import shuffle
from scripts import shredFacts

class Dataset:
    def __init__(self, 
                 ds_name, batch_size):
        """
        Params:
                ds_name : name of the dataset 
        """
        self.name = ds_name
        self.ds_path = "datasets/" + ds_name.lower() + "/"
        self.ent2id, self.rel2id, self.year2id, self.month2id, self.day2id = {}, {}, {}, {}, {}
        self.batch_size = batch_size

        self.data = {"train": self.readFile(self.ds_path + "train.txt"),
                     "valid": self.readFile(self.ds_path + "valid.txt"),
                     "test":  self.readFile(self.ds_path + "test.txt")}
        
        self.start_batch = 0
        self.all_facts_as_tuples = None
        
        self.convertTimes()
        
        self.all_facts_as_tuples = set([tuple(d) for d in self.data["train"] + self.data["valid"] + self.data["test"]])

        self.his = {"train": self.get_his(self.data["train"], self.data["valid"], self.data["test"])[0],
                     "valid": self.get_his(self.data["train"], self.data["valid"], self.data["test"])[1],
                     "test":  self.get_his(self.data["train"], self.data["valid"], self.data["test"])[2]}
        
        
        for spl in ["train", "valid", "test"]:
            self.data[spl] = np.array(self.data[spl])
            self.his[spl] = np.array(self.his[spl])

        self.shiss, self.ohiss = self.get_pad(self.his["train"][0]), self.get_pad(self.his["train"][1])
    
    def get_sort_data(self, t=0):
        self.train_data, self.train_his = self.data["train"], [self.his["train"][0], self.his["train"][1]] #self.make_drop(t)
        idxs = self.get_sorted_idx()
        self.train_data, self.train_his = self.train_data[idxs], [self.train_his[0][idxs], self.train_his[1][idxs]]

    def get_pad(self, his_data):
        start_batch = 0
        his_pad = []
        while start_batch + self.batch_size < len(his_data):   
            hiss = self.padding(his_data[start_batch : start_batch + self.batch_size])
            start_batch += self.batch_size
        hiss = self.padding(his_data[self.start_batch:])
        his_pad.extend(hiss)
        return his_pad

    def padding(self, hiss):
        batch_size, maxlen = len(hiss), max(map(len, hiss))
        hiss_new = []
        for i, his in enumerate(hiss):
            curlen = len(his)
            padding = [self.numRel() for _ in range(maxlen-curlen)]
            hiss_new.append(his+padding)
        return np.array(hiss_new)

    def readFile(self, 
                 filename):

        with open(filename, "r", encoding="utf-8") as f:
            data = f.readlines()
        
        facts = []
        for line in data:
            elements = line.strip().split("\t")
            
            head_id =  self.getEntID(elements[0])
            rel_id  =  self.getRelID(elements[1])
            tail_id =  self.getEntID(elements[2])
            timestamp = elements[3]
            date = list(map(float, timestamp.split("-")))
            year_id = self.getYearID(date[0])
            month_id = self.getMonthID(date[1])
            day_id = self.getDayID(date[2])
            
            facts.append([head_id, rel_id, tail_id, timestamp])
            
        return facts
    
    
    def convertTimes(self):      
        """
        This function spits the timestamp in the day,date and time.
        """  
        for split in ["train", "valid", "test"]:
            for i, fact in enumerate(self.data[split]):
                fact_date = fact[-1]
                self.data[split][i] = self.data[split][i][:-1]
                date = list(map(float, fact_date.split("-")))
                self.data[split][i] += date
                self.data[split][i] += [self.year2id[date[0]], self.month2id[date[1]], self.day2id[date[2]]]
                
    
    def numEnt(self):
    
        return len(self.ent2id)

    def numRel(self):
    
        return len(self.rel2id)

    def numYear(self):
        return len(self.year2id)

    def numMonth(self):
        return len(self.month2id)   

    def numDay(self):
        return len(self.day2id)
    
    def getEntID(self,
                 ent_name):

        if ent_name in self.ent2id:
            return self.ent2id[ent_name] 
        self.ent2id[ent_name] = len(self.ent2id)
        return self.ent2id[ent_name]
    
    def getRelID(self, rel_name):
        if rel_name in self.rel2id:
            return self.rel2id[rel_name] 
        self.rel2id[rel_name] = len(self.rel2id)
        return self.rel2id[rel_name]

    def getYearID(self, year_name):
        if year_name in self.year2id:
            return self.year2id[year_name] 
        self.year2id[year_name] = len(self.year2id)
        return self.year2id[year_name]

    def getMonthID(self, month_name):
        if month_name in self.month2id:
            return self.month2id[month_name] 
        self.month2id[month_name] = len(self.month2id)
        return self.month2id[month_name]

    def getDayID(self, day_name):
        if day_name in self.day2id:
            return self.day2id[day_name] 
        self.day2id[day_name] = len(self.day2id)
        return self.day2id[day_name]


    def get_his(self, train_quas, valid_quas, test_quas):
        quas = train_quas+valid_quas
        subs_dict = {ent:set() for ent in self.ent2id.values()}
        objs_dict = {ent:set() for ent in self.ent2id.values()}

        for qua in quas:
            subs_dict[qua[0]].add(qua[1])
            objs_dict[qua[2]].add(qua[1])
    
        train_subs_his, train_objs_his = [], []
        for qua in train_quas:
            train_subs_his.append(list(subs_dict[qua[0]]))
            train_objs_his.append(list(objs_dict[qua[2]]))
    
        valid_subs_his, valid_objs_his = [], []
        for qua in valid_quas:
            valid_subs_his.append(list(subs_dict[qua[0]]))
            valid_objs_his.append(list(objs_dict[qua[2]]))
        
        test_subs_his, test_objs_his = [], []

        for qua in test_quas:
            if qua[0] in subs_dict.keys():
                test_subs_his.append(list(subs_dict[qua[0]]))
            else:
                test_subs_his.append([])
            if qua[2] in objs_dict.keys():
                test_objs_his.append(list(objs_dict[qua[2]]))
            else:
                test_objs_his.append([])
    
        return [train_subs_his, train_objs_his], [valid_subs_his, valid_objs_his], [test_subs_his, test_objs_his]

    def get_sorted_idx(self):
        assert len(self.train_his[0])==len(self.train_his[1])
        sub_his, obj_his = self.train_his[0], self.train_his[1]
        sub_len, obj_len = -np.asarray(list(map(len, sub_his))), -np.asarray(list(map(len, obj_his)))
        length = np.add(sub_len, obj_len)
        idxs = np.lexsort((obj_len, sub_len, length), )
        
        return idxs   
    
    def nextPosBatch(self, batch_size):
        if self.start_batch + batch_size > len(self.data["train"]):
            ret_facts = self.data["train"][self.start_batch : ]
            shiss, ohiss = self.shiss[self.start_batch : ], self.ohiss[self.start_batch : ]
            if len(ret_facts)%2 != 0:
                ret_facts = np.append(ret_facts, ret_facts[-1].reshape(1,-1), axis=0)
                shiss = np.append(shiss, shiss[-1].reshape(1,-1), axis=0)
                ohiss = np.append(ohiss, ohiss[-1].reshape(1,-1), axis=0)

            self.start_batch = 0
        else:
            ret_facts = self.data["train"][self.start_batch : self.start_batch + batch_size]
            shiss, ohiss = self.shiss[self.start_batch : self.start_batch + batch_size], self.ohiss[self.start_batch : self.start_batch + batch_size]
            self.start_batch += batch_size
        return ret_facts, shiss, ohiss
    
    def addNegFacts2(self, bp_facts, neg_ratio):
        pos_neg_group_size = 1 + neg_ratio
        facts1 = np.repeat(np.copy(bp_facts), pos_neg_group_size, axis=0)
        facts2 = np.copy(facts1)
        rand_nums1 = np.random.randint(low=1, high=self.numEnt(), size=facts1.shape[0])
        rand_nums2 = np.random.randint(low=1, high=self.numEnt(), size=facts2.shape[0])
        
        for i in range(facts1.shape[0] // pos_neg_group_size):
            rand_nums1[i * pos_neg_group_size] = 0
            rand_nums2[i * pos_neg_group_size] = 0
        
        facts1[:,0] = (facts1[:,0] + rand_nums1) % self.numEnt()
        facts2[:,2] = (facts2[:,2] + rand_nums2) % self.numEnt()

        return facts1, facts2
    
    def nextBatch(self, batch_size, neg_ratio=1):
        bp_facts, shiss, ohiss = self.nextPosBatch(batch_size)
        batch1, batch2 = self.addNegFacts2(bp_facts, neg_ratio)
        return shredFacts(batch1, ohiss, bp_facts), shredFacts(batch2, shiss, bp_facts)
    
    def wasLastBatch(self):
        return (self.start_batch == 0)
            
