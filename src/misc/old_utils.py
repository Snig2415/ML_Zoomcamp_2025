import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_data(intake_path, outcome_path):
    # Load CSVs
    df_intakes = pd.read_csv(intake_path)
    df_outcomes = pd.read_csv(outcome_path)

    # Rename columns to snake_case
    df_intakes = df_intakes.rename(columns={
        "Animal ID": "animal_id",
        "DateTime": "intake_datetime",
        "Intake Type": "intake_type",
        "Intake Condition": "intake_condition",
        "Animal Type": "animal_type",
        "Sex upon Intake": "sex_upon_intake",
        "Age upon Intake": "age_upon_intake",
        "Breed": "breed",
        "Color": "color"
    })

    df_outcomes = df_outcomes.rename(columns={
        "Animal ID": "animal_id",
        "DateTime": "outcome_datetime",
        "Outcome Type": "outcome_type",
        "Outcome Subtype": "outcome_subtype",
        "Animal Type": "animal_type",
        "Sex upon Outcome": "sex_upon_outcome",
        "Age upon Outcome": "age_upon_outcome",
        "Breed": "breed",
        "Color": "color"
    })

    # Merge intakes and outcomes
    df = df_intakes.merge(
        df_outcomes[["animal_id", "outcome_datetime", "outcome_type"]],
        on="animal_id",
        how="left"
    )

    # Target: adopted = 1 if adoption, else 0
    df["adopted"] = (df["outcome_type"] == "Adoption").astype(int)

    # Fill missing values
    df = df.fillna({
        "intake_type": "Unknown",
        "intake_condition": "Unknown",
        "sex_upon_intake": "Unknown"
    })

    return df

def preprocess_features(df, categorical_features):
    df_encoded = df.copy()
    le_dict = {}
    for col in categorical_features:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col])
        le_dict[col] = le
    return df_encoded, le_dict

def split_data(df, target="adopted", test_size=0.2, random_state=42):
    X = df.drop(columns=[target, "animal_id", "intake_datetime", "outcome_datetime", "outcome_type"])
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test
