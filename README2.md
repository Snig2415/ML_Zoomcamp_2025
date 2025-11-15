ML_Zoomcamp_2025/
│
├── src/
│   ├── train.py
│   ├── predict.py
│   ├── serve.py
│   ├── model.joblib
│   ├── dictvectorizer.joblib
│
├── notebook.ipynb
├── requirements.txt
├── Dockerfile
└── README.md

1. Dataset

This project uses Austin Animal Center public datasets:

Intakes:
https://data.austintexas.gov/api/views/wter-evkm/rows.csv?accessType=DOWNLOAD

Outcomes:
https://data.austintexas.gov/api/views/9t4d-g238/rows.csv?accessType=DOWNLOAD

These are automatically downloaded inside the notebook and train.py.

2. Data Cleaning Pipeline

The notebook performs:

✔ Column standardization
✔ Filtering animal_id present in both datasets
✔ Creation of target label:
adopted = 1 if outcome_type == "Adoption" else 0


3. Model

The model is a RandomForestClassifier trained on:

Intake attributes

Intake condition

Animal type

Sex

Breed, color

Month & year features

Output: Probability of adoption (0/1)

Saved models:
model.joblib
dictvectorizer.joblib

4. Train the Model

To train the model, run:
python src/train.py
This will:

Download data

Clean & process

Train RandomForest model

Save the model + dictvectorizer

5. Make Predictions

Use the prediction script:
python src/predict.py

Example input and predictions will display in terminal.

6. FastAPI Service

Run the API locally:
uvicorn serve:app --host 0.0.0.0 --port 8000

Open interactive API docs: http://localhost:8000/docs

Example request body:
{
  "intake_type": "Stray",
  "intake_condition": "Normal",
  "animal_type": "Dog",
  "sex_upon_intake": "Neutered Male",
  "breed": "Labrador Retriever Mix",
  "color": "Black",
  "month": 7,
  "year": 2020
}


7. Docker Deployment

Build the image
docker build -t pet-adoption-api .


Run the container
docker run -p 8000:8000 pet-adoption-api
API available at: http://localhost:8000/docs