import time
bef = time.time()
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
aft = time.time()
print(f'Imports complete in {aft-bef} seconds')


N_bottleNeck = 10 # N Dimensional compressed sequence
flattened_conv2lin = 2592


def get_flat_shape(shape_):
    prod = 1
    for elem in shape_:
        prod = prod * elem
    return prod

class AutoEncoder(nn.Module):
    # The autoencoder network and its functions
    def __init__(self):
        # init model architecture
        super(AutoEncoder, self).__init__()
        self.enc_conv1 = nn.Conv2d(1, 32, kernel_size=(2, 2), padding=(1,1))
        self.enc_conv2 = nn.Conv2d(32, 32, kernel_size=(2, 2))
        self.enc_lin1 = nn.Linear(flattened_conv2lin, 1000)
        self.enc_lin2 = nn.Linear(1000, 100)
        self.bottle_neck_enc = nn.Linear(100, N_bottleNeck)
        self.bottle_neck_dec = nn.Linear(N_bottleNeck, 100)
        self.dec_lin2 = nn.Linear(100, 1000)
        self.dec_lin1 = nn.Linear(1000, flattened_conv2lin)
        self.dec_conv2 = nn.Conv2d(32, 32, kernel_size=(2, 2))
        self.dec_conv1 = nn.ConvTranspose2d(32, 1, kernel_size=(2, 2))
        self.init_weights()

    def full_pass(self, x):
        # pass end to end for training autoencoder
        x = self.enc_pass(x)
        x = self.dec_pass(x)
        return x

    def enc_pass(self, x):
        # pass end to bottle neck for compressing data
        x = x.view(-1, 1, 28, 28)
        x = F.relu(self.enc_conv1(x))
        x = F.relu(F.max_pool2d(self.enc_conv2(x), kernel_size=(3, 3)))
        x = x.view(-1, flattened_conv2lin)
        x = F.relu(self.enc_lin1(x))
        x = F.relu(self.enc_lin2(x))
        x = F.relu(self.bottle_neck_enc(x))
        return x

    def dec_pass(self, x):
        # pass bottle neck to end for recreating data
        pass

    def init_weights(self):
        for module in self.modules():
            conditions = bool(isinstance(module, nn.Linear)
                              or isinstance(module, nn.Conv2d))
            if conditions:
                nn.init.kaiming_uniform_(module.weight)
        print('Network initialized with kaiming uniform weights.')

    def train(self, train_loader, optimizer, loss_fn, epochs=1):
        # trains the network on the train_loader data
        # returns train_loss and train_accuracy
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')
        self.cpu = torch.device('cpu')
        for _ in range(epochs):
            temp_loss_list = []
            for data in tqdm(train_loader, disable=not(VERBOSE)):
                X, y = data
                X, y = X.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                out = self.full_pass(X)
                loss = self.loss_fn(out, y)
                temp_loss_list.append(loss)
                loss.backward()
                self.optimizer.step()
        loss = torch.mean(torch.tensor(
            temp_loss_list, dtype=torch.float)).item()
        return (loss, accuracy)

    def test(self, test_loader):
        # tests the network on the test_loader data
        # returns test_loss and test_accuracy
        with torch.no_grad():
            for data in tqdm(test_loader, disable=not(VERBOSE)):
                X, y = data
                X, y = X.to(self.device), y.to(self.device)

        return (loss, accuracy)

t = torch.rand(28, 28)
net = AutoEncoder()