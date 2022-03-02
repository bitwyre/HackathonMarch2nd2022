import torch
import numpy
import pickle
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

with open("./data/btc_usd_spot_data.pkl", 'rb') as f:
    prices_data = pickle.load(f)

# autoregressive dataset

def build_lag_dataset(prices_data: list, lag: int):
    """ Build a timeseries dataset from 
        all the previous lags
    """
    train_data = []
    print(f"Total length of dataset is {len(prices_data)}")
    train_data = prices_data[lag:-lag]
    print(f"New timeseries is length is {len(train_data)}")
    return train_data

lag = 100

train_data = build_lag_dataset(prices_data, lag)
train_data = numpy.array(train_data)
prices_data = numpy.array(prices_data)
# print(prices_data)

test_data_size = 200

label_data = prices_data[2*lag:]

# convert to autoregressive input
# an autoregressive input is the next sequence is predicted from
# the previous sequence
test_data = prices_data[-test_data_size:]

print(f"Length of training set is {len(train_data)}")
print(f"Length of label set is {len(label_data)}")
print(f"Length of test set is {len(test_data)}")

train_data = torch.Tensor(train_data).view(-1)
label_data = torch.Tensor(label_data).view(-1)
test_data = torch.Tensor(test_data).view(-1)

batch_size = 64

trainset = torch.utils.data.TensorDataset(train_data, label_data)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size)
testset = torch.utils.data.TensorDataset(test_data)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size)

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
model = nn.LSTM(1, 1)
model = model.to(device)
loss_function = nn.MSELoss()
loss_function = loss_function.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
# torch.autograd.set_detect_anomaly(True)

print(model)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

epochs = 10
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(trainloader, model, loss_function, optimizer)
    torch.save(model.state_dict(), "model_lstm.pth")
    print("Saved PyTorch Model State to model.pth")
print("Done!")
