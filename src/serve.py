from fastapi import FastAPI
import joblib
from pydantic import BaseModel
import uvicorn

app = FastAPI()

model = joblib.load("model.joblib")
dv = joblib.load("dictvectorizer.joblib")


class Animal(BaseModel):
    intake_type: str
    intake_condition: str
    animal_type: str
    sex_upon_intake: str
    breed: str
    color: str
    month: int
    year: int


@app.post("/predict")
def predict(animal: Animal):
    X = dv.transform([animal.dict()])
    pred = int(model.predict(X)[0])
    return {"adopted": pred}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
