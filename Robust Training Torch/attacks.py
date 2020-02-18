"""
Adversarial attacks.
Author: Zhen Xiang
Date: 12/18/2019
"""

from __future__ import absolute_import
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import argparse

class PGDAttack:

    def __init__(self, model, epsilon, num_steps, step_size, random_start, loss_func, device):
        self.model = model
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.step_size = step_size
        self.rand = random_start
        self.device = device

        if loss_func == 'xent':
            self.criterion = nn.CrossEntropyLoss()
        
    def perturb(self, x, y):
        if self.rand:
            x_adv = x + torch.randint(low=int(-self.epsilon), high=int(self.epsilon), size=x.size(),
                                      dtype=torch.float).to(self.device)/255
            x_adv = torch.clamp(x_adv, min=0, max=1)

        for i in range(self.num_steps):
            x_temp = x_adv.clone()
            x_temp.requires_grad = True
            outputs, _ = self.model(x_temp)
            # outputs = self.model(x_temp) # If penultimate layer features are not needed
            loss = self.criterion(outputs, y)
            self.model.zero_grad()
            loss.backward(retain_graph=True)
            x_temp.detach()
            grad = x_temp.grad.data
            x_adv = x_adv + torch.sign(grad) * self.step_size / 255
            x_adv = torch.clamp(x_adv, min=0, max=1)
            del x_temp

        return x_adv
