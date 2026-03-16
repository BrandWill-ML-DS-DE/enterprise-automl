from data_pipeline import load_and_validate, feature_engineering
from automl_engine import run_automl

DATA_PATH = "data.csv"
TARGET_COLUMN = "target"

df = load_and_validate(DATA_PATH, TARGET_COLUMN)
X, y = feature_engineering(df, TARGET_COLUMN)

model = run_automl(X, y, n_trials=30)

print("Training complete. Model saved.")