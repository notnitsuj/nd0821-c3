import pickle
import logging

import uvicorn
import pandas as pd
from fastapi import FastAPI

from src.ml.data import process_data
from src.ml.model import inference
from data.model import Request


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s: %(name)s - %(levelname)s - %(message)s")

model_dict = {}
with open("model/model.pkl", "rb") as f:
    model_dict["model"] = pickle.load(f)

with open("model/encoder.pkl", "rb") as f:
    model_dict["encoder"] = pickle.load(f)

with open("model/lb.pkl", "rb") as f:
    model_dict["lb"] = pickle.load(f)


app = FastAPI()


@app.get("/")
async def welcome():
    return "Welcome to Udacity Machine Learning DevOps Engineer Nanodegree!"


@app.post("/predict")
async def predict(request: Request):
    request = {key.replace('_', '-'): [value]
               for key, value in request.__dict__.items()}
    data = pd.DataFrame.from_dict(request)

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

    X, _, _, _ = process_data(data, categorical_features=cat_features, label=None,
                              training=False, encoder=model_dict["encoder"], lb=model_dict["lb"])

    pred = inference(model_dict["model"], X)[0]

    return {'prediction': '<= 50K'} if pred == 0 else {'prediction': '> 50K'}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
