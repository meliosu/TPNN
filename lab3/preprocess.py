import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple


def data() -> pd.DataFrame:
    dataset = pd.read_csv(
        '../individual_electric/household_power_consumption.csv',
        sep=';',
        na_values='?',
    )

    dataset['Date'] = pd.to_datetime(
        dataset['Date'] + ' ' + dataset['Time'],
        format='%d/%m/%Y %H:%M:%S'
    )

    dataset.drop(columns='Time', inplace=True)

    for column in dataset.columns:
        dataset[column] = dataset[column].ffill()

    return dataset


def prepare_data(
    df: pd.DataFrame, 
    target_col: str,
    sequence_length: int = 60,
    test_size: float = 0.2
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, MinMaxScaler]:
    # Extract the target column
    data = df[target_col].values.reshape(-1, 1)
    
    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)

    # Create sequences
    X, y = [], []
    for i in range(len(data_scaled) - sequence_length):
        X.append(data_scaled[i:i+sequence_length])
        y.append(data_scaled[i+sequence_length])
    
    X, y = np.array(X), np.array(y)

    train_size = int(len(X) * (1 - test_size))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    return X_train, X_test, y_train, y_test, scaler
