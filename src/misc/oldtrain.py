import sys
import os

# Ensure src folder is in path
sys.path.append(os.path.dirname(__file__))

from src.old_utils import load_data, preprocess_features, split_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import joblib

# Load data (update paths if needed)
df = load_data("data/aac_intakes_outcomes.csv", "data/aac_intakes_outcomes.csv")

# Categorical features
categorical_features = ["intake_type", "intake_condition", "animal_type", "sex_upon_intake", "breed", "color"]

# Preprocess
df_encoded, le_dict = preprocess_features(df, categorical_features)

# Split
X_train, X_test, y_train, y_test = split_data(df_encoded)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("F1-score:", f1_score(y_test, y_pred))

# Save model and encoders
joblib.dump(model, "src/model.joblib")
joblib.dump(le_dict, "src/label_encoders.joblib")
print("Model saved successfully.")
