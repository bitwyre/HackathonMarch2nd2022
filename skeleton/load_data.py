import pickle
from tqdm import tqdm
from decimal import Decimal


btc_usd_data = []

with open("./data/cleaned_data.pkl", "rb") as f:
    data = pickle.load(f)
    for row in tqdm(data):
        if row[1] == "btc_usd_spot":
            btc_usd_data.append(float(Decimal(row[2])))

    # dump to pickle file
    with open("./data/btc_usd_spot_data.pkl", 'wb') as f:
        pickle.dump(btc_usd_data, f)    
