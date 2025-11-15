
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.feature_extraction import DictVectorizer
import joblib


def load_data():
    intakes_url = "https://data.austintexas.gov/api/views/wter-evkm/rows.csv?accessType=DOWNLOAD"
    outcomes_url = "https://data.austintexas.gov/api/views/9t4d-g238/rows.csv?accessType=DOWNLOAD"

    df_intakes = pd.read_csv(intakes_url)
    df_outcomes = pd.read_csv(outcomes_url)

    df_intakes.columns = df_intakes.columns.str.lower().str.replace(" ", "_")
    df_outcomes.columns = df_outcomes.columns.str.lower().str.replace(" ", "_")

    df_outcomes = df_outcomes[["animal_id", "datetime", "outcome_type"]]

    df = pd.merge(df_intakes, df_outcomes, on="animal_id", how="inner")

    df["adopted"] = (df["outcome_type"] == "Adoption").astype(int)

    df = df.fillna({
        "intake_type": "Unknown",
        "intake_condition": "Unknown",
        "sex_upon_intake": "Unknown",
        "monthyear": "Unknown"
    })

    df["monthyear"] = pd.to_datetime(df["monthyear"], errors="coerce")
    df["month"] = df["monthyear"].dt.month.fillna(0).astype(int)
    df["year"] = df["monthyear"].dt.year.fillna(0).astype(int)

    return df


def preprocess(df):
    categorical = ["intake_type", "intake_condition", "animal_type",
                   "sex_upon_intake", "breed", "color"]

    numeric = ["month", "year"]

    dv = DictVectorizer(sparse=False)

    cat_dicts = df[categorical].to_dict(orient="records")
    X_cat = dv.fit_transform(cat_dicts)
    X_num = df[numeric].values

    X = np.hstack([X_num, X_cat])
    y = df["adopted"].values
    return X, y, dv


def train_model():
    df = load_data()
    X, y, dv = preprocess(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("F1 Score:", f1_score(y_test, y_pred))

    joblib.dump(model, "model.joblib")
    joblib.dump(dv, "dictvectorizer.joblib")
    print("Model saved: model.joblib")
    print("Vectorizer saved: dictvectorizer.joblib")


if __name__ == "__main__":
    train_model()
