import pickle
from tqdm import tqdm
from decimal import Decimal


btc_usd_data = []

with open("./data/btc_usd_spot_data.pkl", "rb") as f:
    data = pickle.load(f)
    for row in data:
        print(row)
