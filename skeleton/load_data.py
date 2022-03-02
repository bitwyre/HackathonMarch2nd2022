import pickle

with open("./data/cleaned_data.pkl", "rb") as f:
    data = pickle.load(f)
    for row in data:
        print(row)
    