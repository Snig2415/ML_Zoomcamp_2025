import joblib
import numpy as np
from sklearn.feature_extraction import DictVectorizer

model = joblib.load("model.joblib")
dv: DictVectorizer = joblib.load("dictvectorizer.joblib")

def predict_single(animal_features: dict):
    """
    Example input:
    {
      "intake_type": "Stray",
      "intake_condition": "Normal",
      "animal_type": "Dog",
      "sex_upon_intake": "Neutered Male",
      "breed": "Labrador",
      "color": "Black",
      "month": 3,
      "year": 2022
    }
    """
    X = dv.transform([animal_features])
    pred = model.predict(X)[0]
    return int(pred)


if __name__ == "__main__":
    # example manual test
    example = {
        "intake_type": "Stray",
        "intake_condition": "Normal",
        "animal_type": "Dog",
        "sex_upon_intake": "Neutered Male",
        "breed": "Labrador Mix",
        "color": "Black",
        "month": 5,
        "year": 2024
    }

    print("Prediction:", predict_single(example))
