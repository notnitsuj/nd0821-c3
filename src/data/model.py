from typing import Literal

from pydantic import BaseModel


class Request(BaseModel):
    age: int
    workclass: Literal["State-gov",
                       "Self-emp-not-inc",
                       "Private",
                       "Federal-gov",
                       "Local-gov",
                       "Self-emp-inc",
                       "Without-pay"]
    fnlgt: int
    education: Literal[
        "Bachelors", "HS-grad", "11th", "Masters", "9th",
        "Some-college",
        "Assoc-acdm", "7th-8th", "Doctorate", "Assoc-voc", "Prof-school",
        "5th-6th", "10th", "Preschool", "12th", "1st-4th"]
    education_num: int
    marital_status: Literal["Never-married",
                            "Married-civ-spouse",
                            "Divorced",
                            "Married-spouse-absent",
                            "Separated",
                            "Married-AF-spouse",
                            "Widowed"]
    occupation: Literal["Tech-support",
                        "Craft-repair",
                        "Other-service",
                        "Sales",
                        "Exec-managerial",
                        "Prof-specialty",
                        "Handlers-cleaners",
                        "Machine-op-inspct",
                        "Adm-clerical",
                        "Farming-fishing",
                        "Transport-moving",
                        "Priv-house-serv",
                        "Protective-serv",
                        "Armed-Forces"]
    relationship: Literal["Wife", "Own-child", "Husband",
                          "Not-in-family", "Other-relative", "Unmarried"]
    race: Literal["White", "Asian-Pac-Islander",
                  "Amer-Indian-Eskimo", "Other", "Black"]
    sex: Literal["Female", "Male"]
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: Literal[
        "United-States", "Cuba", "Jamaica", "India", "Mexico",
        "Puerto-Rico", "Honduras", "England", "Canada", "Germany", "Iran",
        "Philippines", "Poland", "Columbia", "Cambodia", "Thailand",
        "Ecuador", "Laos", "Taiwan", "Haiti", "Portugal",
        "Dominican-Republic", "El-Salvador", "France", "Guatemala",
        "Italy", "China", "South", "Japan", "Yugoslavia", "Peru",
        "Outlying-US(Guam-USVI-etc)", "Scotland", "Trinadad&Tobago",
        "Greece", "Nicaragua", "Vietnam", "Hong", "Ireland", "Hungary",
        "Holand-Netherlands"]

    class Config:
        schema_extra = {
            "example": {
                "age": 39,
                "workclass": 'State-gov',
                "fnlgt": 77516,
                "education": 'Bachelors',
                "education_num": 13,
                "marital_status": "Never-married",
                "occupation": "Adm-clerical",
                "relationship": "Adm-clerical",
                "race": "White",
                "sex": "Female",
                "capital_gain": 2174,
                "capital_loss": 0,
                "hours_per_week": 40,
                "native_country": "United-States"
            }
        }
