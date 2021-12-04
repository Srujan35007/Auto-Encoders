import time 
import os 
from datetime import datetime
import numpy as np 
import torch
import torch.nn as nn 
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from dataset import ArtData
from models import Encoder, Decoder


# Hyperparameters and config
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cpu = torch.device("cpu")
LEARNING_RATE = 0.0003
loss_fn = nn.BCELoss()

N_EPOCHS = 300
BATCH_SIZE = 8
TRAINING_SAMPLES_LIMIT = 2048
RESOLUTION = 256
IMG_CHANNELS = 3


# Some helper functions
get_timestamp = lambda : datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

def append_to_logfile(file_path, content):
    with open(file_path, 'a') as write_file:
        write_file.write(content)


# Load dataset
DATASET_PATH = f"/home/attacktitan/Projects/Reddit_Metrics/Subreddit_Metrics/PushShift_method/Download_and_Preprocess_Posts/Clean_Pictures"
train_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5]*IMG_CHANNELS, [0.5]*IMG_CHANNELS),
    ])
train_data = ArtData(DATASET_PATH, resolution=RESOLUTION, 
                    limit=TRAINING_SAMPLES_LIMIT, transform=train_transforms)
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
print(f"Data loaded")


# Initialize models and optimizers
load_enc_weigths_path = None
load_dec_weights_path = None

if load_gen_weigths_path:
    enc = Encoder()
    enc = torch.load(load_enc_weigths_path)
    enc = enc.to(device)
    print(f"Encoder weights loaded")
else:
    netG = Generator(z_dims=Z_DIMS).to(device)

if load_dec_weights_path:
    dec = Decoder()
    dec = torch.load(load_dec_weights_path)
    dec = dec.to(device)
    print(f"Decoder weights loaded")
else:
    netD = Discriminator().to(device)

optim_enc = optim.Adam(enc.parameters(), lr=LEARNING_RATE)
optim_dec = optim.Adam(dec.parameters(), lr=LEARNING_RATE)


# metrics
enc_losses_per_epoch = []
dec_losses_per_epoch = []
logs_path = f'./logs/AE_RUN_{get_timestamp()}'
os.system(f'mkdir -p {logs_path}')
test_images_dir = f"{logs_path}/generated_images"
os.system(f"mkdir -p {test_images_dir}")
save_models_dir = f"{logs_path}/saved_models"
os.system(f"mkdir -p {save_models_dir}")
metrics_logfile_path = f'{logs_path}/metrics.csv'
logfile_column_headers = "EPOCHS,BATCHES,LOSSES\n"
append_to_logfile(metrics_logfile_path, logfile_column_headers)


# Start training
print(f"Started training on device: {device}\n\n")
for epoch_idx in range(N_EPOCHS):
    start_clock = time.perf_counter()
    # For batch metrics
    per_batch_log = ''
    g_losses_per_batch = []
    d_losses_per_batch = []
    for batch_idx, real in enumerate(train_loader):
        _batch_size = real.shape[0]
        enc.train()
        dec.train()

        # Batch metrics
        per_batch_log = per_batch_log + \
                        f"{epoch_idx+1},{batch_idx+1},{lossG.item()},{lossD.item()}\n"
        g_losses_per_batch.append(lossG.item())
        d_losses_per_batch.append(lossD.item())
        # Display batch metrics
        if batch_idx % (len(train_loader)//10) == 0:
            print(f"\tEpoch({epoch_idx+1}/{N_EPOCHS}) |", end=' ')
            print(f"Batch: ({batch_idx+1}/{len(train_loader)}) |", end=' ')
            print(f"LossG: {np.average(g_losses_per_batch):.6f} |", end=' ')
            print(f"LossD: {np.average(d_losses_per_batch):.6f}")

    # Epoch metrics
    g_losses_per_epoch.append(np.average(g_losses_per_batch))
    d_losses_per_epoch.append(np.average(d_losses_per_batch))
    append_to_logfile(metrics_logfile_path, per_batch_log)
    # Generate and save test images
    # Save model weights
    enc = enc.to(cpu)
    dec = dec.to(cpu)
    torch.save(enc, f"{save_models_dir}/epoch_{epoch_idx+1}_enc.pt")
    torch.save(dec, f"{save_models_dir}/epoch_{epoch_idx+1}_dec.pt")
    enc = enc.to(device)
    dec = dec.to(device)
    # Display epoch metrics
    end_epoch_clock = time.perf_counter()
    elapsed = end_epoch_clock-start_clock
    print(f"Epoch: {epoch_idx+1} | Elapsed: {elapsed/60:.2f} Min. |", end=' ')
    print(f"LossG: {g_losses_per_epoch[-1]:.8f} | LossD: {d_losses_per_epoch[-1]:.8f}\n")
