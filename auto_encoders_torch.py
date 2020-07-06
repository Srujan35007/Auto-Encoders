import time 
bef = time.time()
import torch
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim 
import numpy as np 
import math 
aft = time.time()
print(f'Imports complete in {aft-bef} seconds')
del(aft, bef)

N_bottleNeck = 10
flattened_conv2lin = 20
class AutoEncoder(nn.Module):
     def __init__(self):
          super(AutoEncoder, self).__init__()
          self.enc_conv1 = nn.Conv2d(1, 32, kernel_size=(2,2))
          self.enc_conv2 = nn.Conv2d(32, 64, kernel_size=(2,2))
          self.enc_lin1 = nn.Linear(flattened_conv2lin, 100)
          self.enc_lin2 = nn.Linear(100, 100)
          self.bottle_neck_enc = nn.Linear(100, N_bottleNeck)
          self.bottle_neck_dec = nn.Linear(N_bottleNeck,100)
          self.dec_lin2 = nn.Linear(100, 100)
          self.dec_lin1 = nn.Linear(100,flattened_conv2lin)
          self.dec_conv2 = nn.Conv2d(64, 32, kernel_size=(2,2))
          self.dec_conv1 = nn.Conv2d(32, 1, kernel_size=(2,2))
          self.init_weights()

     def full_pass(self, x):
          pass

     def enc_pass(self, x):
          pass

     def dec_pass(self, x):
          pass

     def init_weights(self):
          for module in self.modules():
               conditions = bool(isinstance(module, nn.Linear) 
               or isinstance(module, nn.Conv2d))
               if conditions:
                    nn.init.kaiming_uniform_(module.weight)
          print('Network initialized with kaiming uniform weights.')
               
auto = AutoEncoder()
print(auto)