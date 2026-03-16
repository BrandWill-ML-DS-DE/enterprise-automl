import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_and_validate(path, target_column):
    df = pd.read_csv(path)

    if target_column not in df.columns:
        raise ValueError("Target column not found")

    df = df.dropna()

    return df


def feature_engineering(df, target_column):
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    for col in X.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])

    return X, y