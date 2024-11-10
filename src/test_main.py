import json

from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_get_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == "Welcome to Udacity Machine Learning DevOps Engineer Nanodegree!"


def test_post_less_and_equal_than_50K():
    data = {
        "age": 49,
        "workclass": "Private",
        "fnlgt": 160187,
        "education": "9th",
        "education_num": 5,
        "marital_status": "Married-spouse-absent",
        "occupation": "Other-service",
        "relationship": "Not-in-family",
        "race": "Black",
        "sex": "Female",
        "capital_gain": 0,
        "capital_loss": 0,
        "hours_per_week": 16,
        "native_country": "Jamaica"
    }

    response = client.post("/predict", data=json.dumps(data))

    assert response.status_code == 200
    assert response.json() == {"prediction": "<= 50K"}


def test_post_greater_than_50K():
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

    response = client.post("/predict", data=json.dumps(data))

    assert response.status_code == 200
    assert response.json() == {"prediction": "> 50K"}
