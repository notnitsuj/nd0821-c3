import json
import requests
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s: %(name)s - %(levelname)s - %(message)s")

ENDPOINT = "https://nd0821-c3.onrender.com/predict"

data = {
    "age": 29,
    "workclass": "State-gov",
    "fnlgt": 267989,
    "education": "Bachelors",
    "education_num": 13,
    "marital_status": "Married-civ-spouse",
    "occupation": "Prof-specialty",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital_gain": 0,
    "capital_loss": 0,
    "hours_per_week": 50,
    "native_country": "United-States"
}

response = requests.post(ENDPOINT, data=json.dumps(data))

assert response.status_code == 200

logging.info(f"Response status code: {response.status_code}")
logging.info(f"Response: {response.json()}")
