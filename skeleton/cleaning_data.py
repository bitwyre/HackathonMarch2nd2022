import csv
import pickle
from tqdm import tqdm

data = []

def clean_data():
    # read big CSV file and remove stablecoin prices
    # then if it's not a stablecoin dump the data to a pickle file
    # the data is in the format nanosecond, instrument, prices
    with open("./data/external_crypto_asset_prices.csv", newline='') as csvfile:
        external_crypto_asset_prices = csv.reader(csvfile, delimiter=",")
        for row in tqdm(external_crypto_asset_prices):
            is_stablecoin = row[5]
            is_stablecoin = bool(is_stablecoin)
            if is_stablecoin:
                continue
            base_asset = row[1]
            quote_asset = row[2]
            instrument= f"{base_asset}_{quote_asset}_spot"
            price_of_asset = row[3]
            nanosecond_timestamp = row[6]
            nanosecond_timestamp = int(nanosecond_timestamp)
            data.append([nanosecond_timestamp, instrument, price_of_asset])
   
    # dump to pickle file
    with open("./data/cleaned_data.pkl", 'wb') as f:
        pickle.dump(data, f)

if __name__ == "__main__":
    clean_data()
