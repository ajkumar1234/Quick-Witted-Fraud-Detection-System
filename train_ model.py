import joblib
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from data_preprocessing import load_data, preprocess_data, split_data
from feature_engineering import add_behavioral_features

# Load data
df = load_data("data/raw/creditcard.csv")

# Preprocess
X, y = preprocess_data(df)
X = add_behavioral_features(X)

# Train-test split
X_train, X_test, y_train, y_test = split_data(X, y)

# Handle imbalance
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Model
model = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    eval_metric="logloss",
    random_state=42
)

model.fit(X_train_res, y_train_res)

# Evaluation
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(model, "models/fraud_model.pkl")
print("Model saved successfully.")
