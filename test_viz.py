# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 22:24:42 2021

@author: EE348PC1
"""

import unittest
import torch
from torch import nn
from torchviz import make_dot, make_dot_from_trace


def make_mlp_and_input():
    model = nn.Sequential()
    model.add_module('W0', nn.Linear(8, 16))
    model.add_module('tanh', nn.Tanh())
    model.add_module('W1', nn.Linear(16, 1))
    x = torch.randn(1, 8)
    return model, x


class TestTorchviz(unittest.TestCase):

    def test_mlp_make_dot(self):
        model, x = make_mlp_and_input()
        y = model(x)
        dot = make_dot(y.mean(), params=dict(model.named_parameters())).render("attached", format="png")

  

if __name__ == '__main__':
    unittest.main()