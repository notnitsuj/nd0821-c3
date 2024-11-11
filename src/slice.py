import logging
import pickle

import pandas as pd
from sklearn.model_selection import train_test_split

from ml.data import process_data
from ml.model import compute_model_metrics, inference


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s: %(name)s - %(levelname)s - %(message)s")


cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

# Load data
logging.info("Load and process data")
data = pd.read_csv("../data/cleaned_census.csv")
_, test = train_test_split(data, test_size=0.20)

# Load model
logging.info("Load model")
with open("../model/model.pkl", "rb") as f:
    model = pickle.load(f)

with open("../model/encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

with open("../model/lb.pkl", "rb") as f:
    lb = pickle.load(f)


slice_metrics = []

for feature in cat_features:
    for item in test[feature].unique():
        df_temp = test[test[feature] == item]

        X_test, y_test, _, _ = process_data(
            df_temp,
            cat_features,
            label="salary",
            encoder=encoder,
            lb=lb,
            training=False)

        y_pred = inference(model, X_test)

        precision, recall, fbeta = compute_model_metrics(y_test, y_pred)
        row = f"{feature} - {item} - Precision: {precision} - Recall: {recall} - Fbeta: {fbeta}"
        slice_metrics.append(row)
        logging.info(f"{row}")

        with open("slice_output.txt", "w") as f:
            for row in slice_metrics:
                f.write(row + "\n")
