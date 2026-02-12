import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split

def load_data(path):
    return pd.read_csv(path)

def preprocess_data(df):
    X = df.drop("Class", axis=1)
    y = df["Class"]

    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y

def split_data(X, y):
    return train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )
