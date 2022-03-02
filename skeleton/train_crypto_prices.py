import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

import pickle

with open("./data/btc_usd_spot_data.pkl", 'rb') as f:
    prices_data = pickle.load(f)

trainset = torch.utils.data.TensorDataset(torch.tensor(prices_data))
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4)
