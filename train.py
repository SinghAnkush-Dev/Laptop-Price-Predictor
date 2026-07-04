import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error


# Load Dataset

df = pd.read_csv("data/Laptop_price.csv")


# Features and Target

X = df.drop("Price", axis=1)
y = df["Price"]


# Train Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Preprocessing

preprocessor = ColumnTransformer(
    transformers=[
        ("brand", OneHotEncoder(handle_unknown="ignore"), ["Brand"])
    ],
    remainder="passthrough"
)

# Model

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])


# Train

pipeline.fit(X_train, y_train)

# Prediction

y_pred = pipeline.predict(X_test)


# Evaluation

print("="*50)
print("Model Evaluation")
print("="*50)

print("R2 Score :", r2_score(y_test, y_pred))
print("MAE      :", mean_absolute_error(y_test, y_pred))


# Save Model

joblib.dump(pipeline, "model/model.pkl")

print("\n✅ Model Saved Successfully!")