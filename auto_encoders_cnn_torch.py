import time
bef = time.time()
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchsummary import summary
import matplotlib.pyplot as plt
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

def plot_metrics(test_loss_list, train_loss_list):
    epoch_list = [i+1 for i in range(len(train_loss_list))]
    plt.style.use('fivethirtyeight')
    plt.plot(epoch_list, test_loss_list,
             color='blue', linewidth=0.8,
             label='test_loss'
            )
    plt.plot(epoch_list, train_loss_list,
             color='red', linewidth=0.8,
             label='train_loss'
            )
    plt.title('Loss metrics')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def make_batches(train_data, batch_size):
    batch = []
    return batch

def rename_trained_model(save_file_name):
    if save_file_name.endswith('.pt') and ' ' not in save_file_name:
        split_ = save_file_name.split('.')
        file_name = split_[0]
        min_loss = str(min(test_loss_list)).replace('.', '_')
        renamed_file_name = file_name + f'__loss__{min_loss}.pt'
        os.rename(f'./{save_file_name}', f'./{renamed_file_name}')
        print(f'Model saved as <{renamed_file_name}>')
    else:
        raise TypeError('Use a .pt save file\
             and make sure there are no spaces in the save file name')



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

        # The train step
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

PATIENCE = 5
PLT_SHOW = 5
VERBOSE = False
SAVE_FILE_NAME = 'model.pt' # None if you don't wanna save the model (.pt file)
BATCH_SIZE = 30
train_data =            # Can be a numpy list
test_data =             # Can be a numpy list

# Empty initializations
train_flag = True
current_patience = PATIENCE
test_loss_list = []
train_loss_list = []
test_acc_list = []
train_acc_list = []
epoch_count = 0

# Start training 
while train_flag:
    epoch_count += 1
    print(f'\nEpoch {epoch_count}')
    bef = time.time()
    train_batch = make_batches(train_data, BATCH_SIZE)
    train_loss = net.train(train_batch, verbose=VERBOSE)
    aft = time.time()
    test_loss = net.test(test_data, verbose=VERBOSE)
    if not VERBOSE:
        print(f'Training done in {round((aft-bef)/60, 1)} mins.')
        print(f'Testing done in {round((time.time()-aft)/60, 1)} mins.')
    print(f'train_loss : {train_loss} \t test_loss : {test_loss}')
    test_loss_list.append(test_loss)
    train_loss_list.append(train_loss)
    
    if SAVE_FILE_NAME != None:
    # If the user wants to save the model
        if test_loss == min(test_loss_list):
        # If the current loss is the loss minima
            if Path(f'./{SAVE_FILE_NAME}').is_file():
                os.remove(f'./{SAVE_FILE_NAME}')
                net.to(cpu)
                torch.save(net, f'./{SAVE_FILE_NAME}')
                net.to(device)
                if VERBOSE:
                    print(f'New version saved at epoch : {epoch_count} as <{SAVE_FILE_NAME}>')
            else:
                net.to(cpu)
                torch.save(net, f'./{SAVE_FILE_NAME}')
                net.to(device)
                if VERBOSE:
                    print(f'Model saved at epoch : {epoch_count} as <{SAVE_FILE_NAME}>')
    else:
        pass

    if test_loss == min(test_loss_list):
    # Early stopping
        current_patience = PATIENCE
    else:
        current_patience -= 1
    
    if current_patience <= 0:
        train_flag = False
        print(f'Training terminated.')
    
    if epoch_count % PLT_SHOW == 0:
    # Show metrics
        plot_metrics(test_loss_list, train_loss_list)

min_loss = min(test_loss_list)
min_loss_at = test_loss_list.index(min_loss) + 1
plot_metrics(test_loss_list, train_loss_list)
print(f'Loss minima at {min_loss_at}')
rename_trained_model(SAVE_FILE_NAME)