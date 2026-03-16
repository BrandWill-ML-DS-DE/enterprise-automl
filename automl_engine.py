import optuna
import lightgbm as lgb
import mlflow
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os

MODEL_PATH = "models/best_model.pkl"

def objective(trial, X_train, X_val, y_train, y_val):

    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "num_leaves": trial.suggest_int("num_leaves", 20, 150),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
    }

    model = lgb.LGBMClassifier(**params)
    model.fit(X_train, y_train)

    preds = model.predict(X_val)
    acc = accuracy_score(y_val, preds)

    return acc


def run_automl(X, y, n_trials=20):

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    mlflow.set_experiment("Enterprise_AutoML")

    study = optuna.create_study(direction="maximize")

    study.optimize(
        lambda trial: objective(trial, X_train, X_val, y_train, y_val),
        n_trials=n_trials,
    )

    best_params = study.best_params

    best_model = lgb.LGBMClassifier(**best_params)
    best_model.fit(X_train, y_train)

    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)

    mlflow.start_run()
    mlflow.log_params(best_params)
    mlflow.log_metric("best_accuracy", study.best_value)
    mlflow.end_run()

    return best_model