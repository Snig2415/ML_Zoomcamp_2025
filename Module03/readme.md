**Typical Steps in Module 3**

**1. Data Preparation:**

  Clean the data; handle missing values.

  Split into train/validation/test sets.

  Separate features (categorical + numerical) and target (churn).

**2. Exploratory Data Analysis (EDA):**

  Look at distributions of features.

  Calculate churn rates by category.

  Compute risk ratios, mutual information, correlation with the target.

**3. Feature Engineering & Encoding:**

  One-hot encode categorical features.

  Scale/transform numerical features if needed.

  Combine into a matrix ready for model training.

**4. Model Training:**

  Instantiate and train logistic regression (or another classifier).

  Fit on training data, validate on separate split.

**5. Model Interpretation & Evaluation:**

  Inspect model intercept and coefficients to see which features push up/down the churn probability.

  Evaluate performance: accuracy, and check how many correct vs incorrect predictions.

**6. Making Predictions & Using the Model:**

  Use the trained model to predict on new/unseen data.

  Use probability outputs and decide on a threshold for classification (e.g., churn if p ≥ 0.5).

  Understand business implications of predictions: false positives vs false negatives.

**-> Key Takeaway**

  Even a simple model like logistic regression becomes powerful when you:

  Prepare your data well (cleaning + features)

Encode categories correctly

Interpret what the model is doing (which features matter), and

Validate it properly (using separate data).
