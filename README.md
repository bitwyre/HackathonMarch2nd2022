# Bitwyre Hackathon 2nd of March 2022

This hackathon is a trial hackathon for the public. We will be using hackathon for modes of hiring starting from March 2022.

# Dataset

Download the dataset here https://drive.google.com/file/d/1FmTzKa_DLOmIg6bZqAKGjJcuxOWQ3bHw/view

# Metadata

1 is base asset
2 is quote asset (for example for BTC/USDT the base asset is BTC and the quote asset)
3 is the price of the base asset in the quote asset
4 is the datasource
5 is a boolean value if it's a stablecoin
6 is the nanosecond timestamp
7 is the datetime in iso format

# Task

## Trial

Try running train.py on your computer. Don't forget to have a Python interpreter and also Pytorch installed. To start follow this link https://pytorch.org/get-started/locally/

## Dataloader

Adjust train.py so that you can load the dataset given above.

## Best Predictive Model for Prices

After adjusting the train.py dataloader, adjust the NeuralNetwork class so that it can train from the price dataset.

Your task is then to find the best predictive model for prices for any assets given the dataset of past asset trajectory.

For example the input should be an array of N x 1 or N x m assets, where N is the number of steps (rows) of datapoints taken.

Fetch the array into the model and spit out a number that would be the output of the next day's price. This is called backtesting.

# Tutorial

Follow some tutorials on Pytorch here https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html

# Copyright

2022, Bitwyre Technologies Holdings Incorporated (BTHC) Panama
