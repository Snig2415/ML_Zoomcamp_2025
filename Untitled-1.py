# %%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import joblib

# %%
intakes  = "https://data.austintexas.gov/api/views/wter-evkm/rows.csv?accessType=DOWNLOAD"
outcomes = "https://data.austintexas.gov/api/views/9t4d-g238/rows.csv?accessType=DOWNLOAD"
df_intakes  = pd.read_csv(intakes)
df_outcomes = pd.read_csv(outcomes)

# %%
df_intakes.columns = df_intakes.columns.str.lower().str.replace(" ", "_")
df_intakes.columns

# %%
df_outcomes.columns = df_outcomes.columns.str.lower().str.replace(" ", "_")
df_outcomes.columns

# %%
# from outcomes selecting only the important columns 'animal_id', 'datetime','outcome_type' 
df_outcomes = df_outcomes[['animal_id', 'datetime', 'outcome_type']]

# %%
# merging the two dataframes on 'animal_id', just using the animals that are present in both dataframes as for the rest we dont have the outcome
df = pd.merge(df_intakes, df_outcomes, on='animal_id', how='inner', suffixes=('_intake', '_outcome'))
df.head()

# %%
df.shape

# %%
animal_id_counts = df['animal_id'].value_counts()
animal_id_counts

# %%
# Create target column
df["adopted"] = (df["outcome_type"] == "Adoption").astype(int)

# %%
# animals by outcome_type in graph 
df['outcome_type'].value_counts().plot.bar()

# %%
# Fill missing values
df = df.fillna({
    "intake_type": "Unknown",
    "intake_condition": "Unknown",
    "sex_upon_intake": "Unknown",
    "monthyear": "Unknown"
})

# %%
df.dtypes


# %%
# Convert monthyear to numeric features
# ----------------------------
df["monthyear"] = pd.to_datetime(df["monthyear"], errors='coerce')
df["month"] = df["monthyear"].dt.month.fillna(0).astype(int)
df["year"] = df["monthyear"].dt.year.fillna(0).astype(int)

# %%
# Encode categorical features
# ----------------------------
categorical_features = ["intake_type", "intake_condition", "animal_type",
                        "sex_upon_intake", "breed", "color"]

# %%
from sklearn.feature_extraction import DictVectorizer
# Copy dataframe
df_features = df.copy()

# %%
# Convert categorical columns to dict
cat_dicts = df_features[categorical_features].to_dict(orient='records')

# %%
# Initialize DictVectorizer
dv = DictVectorizer(sparse=False)

# %%
# Fit and transform
X_cat = dv.fit_transform(cat_dicts)

# %%
# X_cat is now a numeric array for categorical features
print(X_cat.shape)

# %%
# Optionally, combine with numeric columns
numeric_features = ["month", "year"]  # if you extracted month/year
X_numeric = df_features[numeric_features].values

# %%
import numpy as np
X = np.hstack([X_numeric, X_cat])
y = df["adopted"].values

# %%
# Train/test split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# %%
# Train RandomForest
# ----------------------------
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# %%
# Evaluate
# ----------------------------
y_pred = model.predict(X_test)
print("F1-score:", f1_score(y_test, y_pred))

# %%
# Save model and DictVectorizer
# ----------------------------
joblib.dump(model, "model.joblib")
joblib.dump(dv, "dictvectorizer.joblib")
print("Model and DictVectorizer saved successfully!")


