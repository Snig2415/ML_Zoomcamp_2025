import sys
import os
import pandas as pd
import joblib

sys.path.append(os.path.dirname(__file__))
from src.old_utils import preprocess_features  # optional if needed

# Load model and encoders
model = joblib.load("src/model.joblib")
le_dict = joblib.load("src/label_encoders.joblib")

# Example new pet
new_pet = pd.DataFrame([{
    "intake_type": "Stray",
    "intake_condition": "Normal",
    "animal_type": "Dog",
    "sex_upon_intake": "Neutered Male",
    "breed": "Labrador Retriever",
    "color": "Black"
}])

# Encode categorical features
for col, le in le_dict.items():
    new_pet[col] = le.transform(new_pet[col])

# Predict
pred = model.predict(new_pet)
print("Predicted adopted:", pred[0])
