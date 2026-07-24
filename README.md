# Pitchside — Player Injury Risk Analytics

A small full-stack project: a Flask backend trains real Logistic Regression,
Random Forest, and Keras MLP models on `data.csv`, and a single-page HTML
dashboard (`frontend/index.html`) calls that backend for every number it
shows — nothing on the page is hard-coded or simulated.

```
injury-risk-app/
├── backend/
│   ├── app.py                 # Flask API + serves the frontend
│   ├── train_models.py        # trains & saves all models (run this first)
│   ├── injury_insights.py     # parses player_injuries_impact.csv
│   ├── requirements.txt
│   ├── data/
│   │   ├── data.csv
│   │   └── player_injuries_impact.csv
│   └── models/                # created by train_models.py (not committed)
├── frontend/
│   └── index.html
└── README.md
```

## 1. Set up a virtual environment

```bash
cd injury-risk-app/backend
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

## 2. Install dependencies

```bash
pip install -r requirements.txt
```

If `tensorflow` fails to install (common on very old machines or Apple
Silicon), swap the line in `requirements.txt` for `tensorflow-macos` or
`tensorflow-cpu` — the training code doesn't use anything GPU-specific.

## 3. Train the models (run once)

```bash
python train_models.py
```

This reads `data/data.csv`, trains Logistic Regression / Random Forest /
MLP, evaluates all three on a held-out test split, and writes everything
the API needs into `backend/models/`:

- `scaler.joblib`, `label_encoder.joblib` — the exact preprocessing used at
  training time, so predictions later use identical scaling
- `logreg.joblib`, `rf.joblib`, `mlp.keras` — the trained models themselves
- `metrics.json` — accuracy / AUC / F1 / confusion matrix / feature
  importance for the Model Comparison page
- `feature_defaults.json` — squad-average values used to fill in any field
  the "Predict my risk" form doesn't ask for
- `test_players.json` — the held-out test set with real feature values and
  real predicted risk, used for the Player Explorer table

Re-run this script any time `data.csv` changes.

## 4. Start the server

```bash
python app.py
```

Open **http://localhost:5000** in a browser. Flask serves the dashboard
and the API from the same origin, so there's nothing else to configure.

## 5. What each page actually calls

| Page | Backend endpoint | What it does |
|---|---|---|
| Overview | `GET /api/overview` | Squad stats computed live from `data.csv` |
| Player Explorer | `GET /api/players`, `GET /api/player/<id>` | Real test-set players + real risk score from the best model |
| Model Comparison | `GET /api/models` | Accuracy / AUC / F1 / confusion matrix / feature importance |
| Injury Insights | `GET /api/injuries` | Real stats from `player_injuries_impact.csv` |
| Live Demo — Train | `POST /api/train` | Actually retrains a model with the chosen architecture/epochs/batch size and returns the real training curve |
| Live Demo — Predict | `POST /api/predict` | Runs a real forward pass through the saved model for a hypothetical player |

## 6. Pushing this to GitHub

```bash
cd injury-risk-app
git init
git add .
git commit -m "Injury risk dashboard: Flask backend + dashboard frontend"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

Add a `.gitignore` (see below) before your first commit so you don't check
in your virtual environment or the trained model binaries — anyone who
clones the repo just runs `train_models.py` once to regenerate them.

```gitignore
venv/
__pycache__/
*.pyc
backend/models/
.DS_Store
```

If you *do* want the trained models available immediately after cloning
(useful for a coursework submission where the marker won't run
`train_models.py`), remove `backend/models/` from `.gitignore` and commit
that folder — the `.joblib` and `.keras` files here are small (a few MB at
most).

## 7. Known, deliberately-disclosed limitations

- `data.csv` and `player_injuries_impact.csv` don't share a player ID, so
  the risk-prediction models are trained only on `data.csv`. The Injury
  Insights page reads the injuries file on its own terms rather than
  pretending the two are linked.
- The "top contributing factors" shown per player are Random Forest
  feature importance weighted by how far that player's value sits from the
  training mean — a simple, honest approximation, not true SHAP.
- The dataset's target looks cleanly separable, which is why all three
  models score highly. Treat this as a demonstration of a correct pipeline
  rather than a validated real-world injury-risk model.

## 8. Extending this further

- Swap `feature_defaults.json` filling with a fuller form once you're
  ready to expose all 18 features in the UI.
- Add `SHAP` (`pip install shap`) for a proper per-prediction explanation
  instead of the current importance-weighted approximation.
- Deploy: `app.py` runs fine behind `gunicorn app:app` for a real deployment
  (Render, Railway, a university server, etc.) — just don't leave
  `debug=True` on in production.

## 9. Commands For Github
cd backend
python3 --version
pip install -r requirements.txt
python3 train_models.py
python3 app.py
