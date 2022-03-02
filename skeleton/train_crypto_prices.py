import torch
import numpy
import pickle
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

with open("./data/btc_usd_spot_data.pkl", 'rb') as f:
    prices_data = pickle.load(f)

prices_data = numpy.array(prices_data)
prices_data = torch.from_numpy(prices_data).unsqueeze(1)
print(prices_data)

trainset = torch.utils.data.TensorDataset(torch.tensor(prices_data))
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4)
