# MLOPS Assignment — Housing Price Model

Predict house prices from the Housing dataset. The project has two parts:

| File | Purpose |
|------|---------|
| `src/housing_eda.ipynb` | Exploratory data analysis **and** preprocessing. Self-contained. Writes `data/Housing_processed.csv`. |
| `src/train.py` | Loads the data, trains a log-target **Ridge + ElasticNet + Gradient Boosting** ensemble, saves the model to `model/`. |

## Project layout

```
.
├── data/            # Housing.csv (raw), Housing_processed.csv (generated)
├── model/           # trained models (generated, timestamped .joblib files)
├── src/
│   ├── housing_eda.ipynb
│   └── train.py
│   └── train_23L8072.py
├── requirements.txt
└── README.md
```

## Setup

Run these from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate          
pip install -r requirements.txt
```

## Run the training script

```bash
python src/train.py
```

This loads `data/Housing_processed.csv` if it exists (better features), otherwise
falls back to the raw `data/Housing.csv` (which it engineers the same features on,
so it always runs on a fresh clone). It prints a training summary — a repeated
5-fold cross-validated R² plus hold-out R²/MAE/RMSE — and saves the fitted
pipeline as `model/housing_model_<timestamp>.joblib`.



## (Optional) Regenerate the preprocessed data

Only needed if you change `data/Housing.csv` or the feature engineering:

```bash
jupyter nbconvert --to notebook --execute --inplace src/housing_eda.ipynb
```

or open `src/housing_eda.ipynb` in Jupyter and run all cells.