##  RoAN: A Relation-oriented Attention Network for Temporal Knowledge Graph Completion
Paper: RoAN: A Relation-oriented Attention Network for Temporal Knowledge Graph Completion

This repository contains the implementation of the RoAN architectures described in the paper.

## Installation

Install PyTorch (>= 1.1.0)  following the instuctions on the [PyTorch](https://pytorch.org/) .
Our code is written in Python3.

## How to use?
After installing the requirements, run the following command to reproduce results for RoAN-DES:
```
$ python main.py -dropout 0.4 -se_prop 0.36 -beta 0.5 -neg_ratio 5 -model RoAN—DES
```
To reproduce the results for RoAN-DED and RoAN-DET, specify **model** as RoAN-DED/RoAN-DET as following.
```
$ python main.py -dropout 0.4 -se_prop 0.36 -beta 0.5 -model RoAN—DED
$ python main.py -dropout 0.4 -se_prop 0.36 -beta 0.5 -model RoAN—DET
```
## Baselines

We use the following public codes for baselines and hyperparameters. 

| Baselines                           | Code                                                         |
| ----------------------------------- | ------------------------------------------------------------ |
| TransE                              | [link](https://github.com/jimmywangheng/knowledge_representation_pytorch) |
| TTransE                             | [link](https://github.com/INK-USC/RE-Net)                    |
| HyTE                                | [link](https://github.com/malllabiisc/HyTE)                  |
| DE-TransE / DE-DistMult / DE-SimplE | [link](https://github.com/BorealisAI/DE-SimplE)              |
| TA-TransE / TA-DistMult             | [link](https://github.com/INK-USC/RE-Net)                    |
