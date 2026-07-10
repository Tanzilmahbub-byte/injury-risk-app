"""
train_models.py
================
Trains all injury-risk models on the REAL data.csv file and saves every
artifact the Flask API (app.py) needs to serve the dashboard:

  models/scaler.joblib          - StandardScaler fitted on training features
  models/label_encoder.joblib   - LabelEncoder fitted on the Position column
  models/logreg.joblib          - Logistic Regression model
  models/rf.joblib              - Random Forest model
  models/mlp.keras              - Keras MLP model
  models/metrics.json           - accuracy / AUC / F1 / confusion matrix / feature importance
  models/feature_defaults.json  - mean value of every feature (used to fill in
                                   fields the "Predict my risk" form doesn't ask for)
  models/test_players.json      - the held-out test set, with each player's
                                   real feature values and real predicted risk,
                                   used to populate the Player Explorer table

Run this once before starting app.py:
    python train_models.py

Re-run it any time data.csv changes.
"""
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, confusion_matrix
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "data.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

RANDOM_STATE = 42
TARGET_COL = "Injury_Next_Season"

# ---------------------------------------------------------------------
# 1. Load + preprocess (mirrors preprocess_general() from the notebook,
#    but every step is saved to disk so the API can re-apply it later)
# ---------------------------------------------------------------------
print("Loading data.csv ...")
df = pd.read_csv(DATA_PATH)

le_pos = LabelEncoder()
df["Position"] = le_pos.fit_transform(df["Position"])

feature_cols = [c for c in df.columns if c != TARGET_COL]
X = df[feature_cols]
y = df[TARGET_COL]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Train rows: {len(X_train)}  Test rows: {len(X_test)}  Features: {len(feature_cols)}")

# ---------------------------------------------------------------------
# 2. Train models
# ---------------------------------------------------------------------
print("Training Logistic Regression ...")
logreg = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
logreg.fit(X_train_scaled, y_train)

print("Training Random Forest ...")
rf = RandomForestClassifier(
    n_estimators=300, max_depth=8, random_state=RANDOM_STATE, n_jobs=-1
)
rf.fit(X_train_scaled, y_train)

print("Training MLP (Keras) ...")
import tensorflow as tf
from tensorflow.keras import layers, models

mlp = models.Sequential([
    layers.Input(shape=(X_train_scaled.shape[1],)),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(64, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(1, activation="sigmoid"),
])
mlp.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
)
mlp_history = mlp.fit(
    X_train_scaled, y_train,
    validation_data=(X_test_scaled, y_test),
    epochs=30, batch_size=32, verbose=0,
)
print("MLP final val accuracy:", mlp_history.history["val_accuracy"][-1])

# ---------------------------------------------------------------------
# 3. Evaluate every model the same way
# ---------------------------------------------------------------------
def evaluate(name, y_true, y_pred, y_proba):
    cm = confusion_matrix(y_true, y_pred).tolist()  # [[TN, FP], [FN, TP]]
    return {
        "name": name,
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "auc": round(float(roc_auc_score(y_true, y_proba)), 4),
        "f1": round(float(f1_score(y_true, y_pred)), 4),
        "confusion_matrix": {
            "tn": cm[0][0], "fp": cm[0][1], "fn": cm[1][0], "tp": cm[1][1],
        },
    }

results = {}

proba = logreg.predict_proba(X_test_scaled)[:, 1]
results["logreg"] = evaluate("Logistic Regression", y_test, logreg.predict(X_test_scaled), proba)

proba = rf.predict_proba(X_test_scaled)[:, 1]
results["rf"] = evaluate("Random Forest", y_test, rf.predict(X_test_scaled), proba)

proba = mlp.predict(X_test_scaled, verbose=0).ravel()
results["mlp"] = evaluate("Neural Network (MLP)", y_test, (proba >= 0.5).astype(int), proba)

# pick the model with the best AUC as the "default" model used for
# predictions shown in the Player Explorer and the live predict form
best_key = max(results, key=lambda k: results[k]["auc"])
print(f"Best model by AUC: {best_key} ({results[best_key]['auc']})")

# ---------------------------------------------------------------------
# 4. Feature importance (from Random Forest — the most interpretable model)
# ---------------------------------------------------------------------
importances = sorted(
    zip(feature_cols, rf.feature_importances_.tolist()),
    key=lambda kv: kv[1], reverse=True
)

# ---------------------------------------------------------------------
# 5. Save everything
# ---------------------------------------------------------------------
joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.joblib"))
joblib.dump(le_pos, os.path.join(MODELS_DIR, "label_encoder.joblib"))
joblib.dump(logreg, os.path.join(MODELS_DIR, "logreg.joblib"))
joblib.dump(rf, os.path.join(MODELS_DIR, "rf.joblib"))
mlp.save(os.path.join(MODELS_DIR, "mlp.keras"))

metrics = {
    "best_model": best_key,
    "feature_cols": feature_cols,
    "models": results,
    "feature_importance": [{"feature": f, "importance": round(v, 4)} for f, v in importances],
    "mlp_training_history": {
        "loss": [round(v, 4) for v in mlp_history.history["loss"]],
        "val_accuracy": [round(v, 4) for v in mlp_history.history["val_accuracy"]],
    },
    "position_classes": le_pos.classes_.tolist(),
}
with open(os.path.join(MODELS_DIR, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)

feature_defaults = {c: round(float(X_train[c].mean()), 3) for c in feature_cols}
with open(os.path.join(MODELS_DIR, "feature_defaults.json"), "w") as f:
    json.dump(feature_defaults, f, indent=2)

# Held-out test set players, with real feature values + real predicted risk
# from the best model — this is what powers the Player Explorer table.
best_model = {"logreg": logreg, "rf": rf, "mlp": mlp}[best_key]
if best_key == "mlp":
    risk_scores = best_model.predict(X_test_scaled, verbose=0).ravel()
else:
    risk_scores = best_model.predict_proba(X_test_scaled)[:, 1]

test_players = []
X_test_reset = X_test.reset_index(drop=True)
y_test_reset = y_test.reset_index(drop=True)
for i in range(len(X_test_reset)):
    row = X_test_reset.iloc[i].to_dict()
    row["position_label"] = le_pos.classes_[int(row["Position"])]
    row["actual_injury_next_season"] = int(y_test_reset.iloc[i])
    row["predicted_risk"] = round(float(risk_scores[i]) * 100, 1)
    row["id"] = i
    test_players.append(row)

with open(os.path.join(MODELS_DIR, "test_players.json"), "w") as f:
    json.dump(test_players, f, indent=2)

print("\nAll artifacts saved to:", MODELS_DIR)
print("Done. You can now run: python app.py")
