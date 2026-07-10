"""
app.py
======
Flask API + static file server for the Injury Risk Dashboard.

Endpoints
---------
GET  /                     -> serves the frontend (frontend/index.html)
GET  /api/overview         -> squad-level stats from data.csv
GET  /api/players          -> held-out test-set players + real predicted risk
GET  /api/player/<id>      -> one player's real feature values + top factors
GET  /api/models           -> accuracy/AUC/F1/confusion matrix/importance
GET  /api/injuries         -> real insights from player_injuries_impact.csv
POST /api/predict          -> real prediction for a hypothetical player
POST /api/train            -> retrains MLP or Random Forest live and streams
                               back the real per-epoch/per-tree history

Run:
    pip install -r requirements.txt
    python train_models.py      # only needed once, or after data.csv changes
    python app.py
Then open http://localhost:5000 in a browser.
"""
import json
import os

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

from injury_insights import get_injury_insights

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_PATH = os.path.join(BASE_DIR, "data", "data.csv")
FRONTEND_DIR = os.path.join(os.path.dirname(BASE_DIR), "frontend")

app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path="")
CORS(app)

# ---------------------------------------------------------------------
# Load all trained artifacts once at startup
# ---------------------------------------------------------------------
def require_trained_models():
    if not os.path.exists(os.path.join(MODELS_DIR, "metrics.json")):
        raise RuntimeError(
            "No trained models found. Run `python train_models.py` first."
        )

require_trained_models()

scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.joblib"))
label_encoder = joblib.load(os.path.join(MODELS_DIR, "label_encoder.joblib"))
logreg = joblib.load(os.path.join(MODELS_DIR, "logreg.joblib"))
rf = joblib.load(os.path.join(MODELS_DIR, "rf.joblib"))

import tensorflow as tf
mlp = tf.keras.models.load_model(os.path.join(MODELS_DIR, "mlp.keras"))

with open(os.path.join(MODELS_DIR, "metrics.json")) as f:
    metrics = json.load(f)
with open(os.path.join(MODELS_DIR, "feature_defaults.json")) as f:
    feature_defaults = json.load(f)
with open(os.path.join(MODELS_DIR, "test_players.json")) as f:
    test_players = json.load(f)

FEATURE_COLS = metrics["feature_cols"]
BEST_MODEL_KEY = metrics["best_model"]
MODEL_LOOKUP = {"logreg": logreg, "rf": rf, "mlp": mlp}

raw_df = pd.read_csv(DATA_PATH)


def predict_risk(feature_dict, model_key=None):
    """feature_dict must already have numeric Position (label-encoded)."""
    model_key = model_key or BEST_MODEL_KEY
    row = np.array([[feature_dict[c] for c in FEATURE_COLS]])
    row_scaled = scaler.transform(row)
    model = MODEL_LOOKUP[model_key]
    if model_key == "mlp":
        proba = float(model.predict(row_scaled, verbose=0).ravel()[0])
    else:
        proba = float(model.predict_proba(row_scaled)[:, 1][0])
    return proba, row_scaled[0]


def top_factors(row_scaled, n=4):
    """Approximate per-prediction factor contribution: RF feature importance
    weighted by how far this player's (scaled) value sits from the training
    mean of 0. This is a simple, honest approximation — not true SHAP —
    and is labelled as such in the UI."""
    importances = dict(
        (f["feature"], f["importance"]) for f in metrics["feature_importance"]
    )
    contributions = []
    for i, feat in enumerate(FEATURE_COLS):
        weight = importances.get(feat, 0)
        contribution = abs(row_scaled[i]) * weight
        contributions.append({"feature": feat, "contribution": round(float(contribution), 4)})
    contributions.sort(key=lambda c: c["contribution"], reverse=True)
    return contributions[:n]


# ---------------------------------------------------------------------
# Static frontend
# ---------------------------------------------------------------------
@app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


# ---------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------
@app.route("/api/overview")
def overview():
    injury_rate = float(raw_df["Injury_Next_Season"].mean()) * 100
    position_counts = raw_df["Position"].value_counts().to_dict()
    return jsonify({
        "n_players": int(len(raw_df)),
        "injury_rate_pct": round(injury_rate, 1),
        "avg_previous_injuries": round(float(raw_df["Previous_Injury_Count"].mean()), 2),
        "avg_training_hours": round(float(raw_df["Training_Hours_Per_Week"].mean()), 1),
        "position_counts": position_counts,
        "n_features": len(FEATURE_COLS),
        "best_model": BEST_MODEL_KEY,
    })


@app.route("/api/players")
def players():
    return jsonify(test_players)


@app.route("/api/player/<int:player_id>")
def player_detail(player_id):
    match = next((p for p in test_players if p["id"] == player_id), None)
    if not match:
        return jsonify({"error": "player not found"}), 404

    row = np.array([[match[c] for c in FEATURE_COLS]])
    row_scaled = scaler.transform(row)[0]
    factors = top_factors(row_scaled)
    return jsonify({**match, "top_factors": factors})


@app.route("/api/models")
def models_comparison():
    return jsonify(metrics)


@app.route("/api/injuries")
def injuries():
    return jsonify(get_injury_insights())


@app.route("/api/predict", methods=["POST"])
def predict():
    body = request.get_json(force=True)

    # Start from squad averages, then overwrite with whatever the
    # frontend form actually sent (age, training load, etc.)
    feature_dict = dict(feature_defaults)
    for key, value in body.items():
        if key == "position":
            try:
                feature_dict["Position"] = int(
                    np.where(label_encoder.classes_ == value)[0][0]
                )
            except IndexError:
                pass  # unknown position label, keep the default
        elif key in feature_dict:
            feature_dict[key] = float(value)

    model_key = body.get("model", BEST_MODEL_KEY)
    proba, row_scaled = predict_risk(feature_dict, model_key)
    factors = top_factors(row_scaled)

    return jsonify({
        "risk_pct": round(proba * 100, 1),
        "model_used": model_key,
        "top_factors": factors,
    })


@app.route("/api/train", methods=["POST"])
def train_live():
    """Retrains a model live, in-request, on the real training split, and
    returns the REAL per-epoch (MLP) or per-tree-batch (Random Forest)
    history so the frontend can animate genuine training curves."""
    body = request.get_json(force=True)
    architecture = body.get("architecture", "mlp")
    epochs = int(body.get("epochs", 20))
    batch_size = int(body.get("batch_size", 32))

    from sklearn.model_selection import train_test_split
    X = raw_df[FEATURE_COLS].copy()
    X["Position"] = label_encoder.transform(raw_df["Position"])
    y = raw_df["Injury_Next_Season"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if architecture == "rf":
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score

        step = max(1, epochs // 20)  # up to ~20 points on the chart
        val_accuracy = []
        model = RandomForestClassifier(
            n_estimators=step, warm_start=True, random_state=42, n_jobs=-1
        )
        n_estimators = 0
        for _ in range(min(epochs, 20)):
            n_estimators += step
            model.n_estimators = n_estimators
            model.fit(X_train_scaled, y_train)
            acc = accuracy_score(y_test, model.predict(X_test_scaled))
            val_accuracy.append(round(float(acc), 4))
        return jsonify({
            "architecture": "rf",
            "loss": None,
            "val_accuracy": val_accuracy,
        })

    # default: MLP, real epoch-by-epoch training
    from tensorflow.keras import layers, models as keras_models
    import tensorflow as tf

    live_model = keras_models.Sequential([
        layers.Input(shape=(X_train_scaled.shape[1],)),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(1, activation="sigmoid"),
    ])
    live_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    history = live_model.fit(
        X_train_scaled, y_train,
        validation_data=(X_test_scaled, y_test),
        epochs=epochs, batch_size=batch_size, verbose=0,
    )
    return jsonify({
        "architecture": "mlp",
        "loss": [round(v, 4) for v in history.history["loss"]],
        "val_accuracy": [round(v, 4) for v in history.history["val_accuracy"]],
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
