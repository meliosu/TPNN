import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple


def data(data_fraction: float = 1.0) -> pd.DataFrame:
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

    if data_fraction < 1.0:
        dataset = dataset[:int(len(dataset) * data_fraction)]
    
    return dataset


def prepare_data(
    df: pd.DataFrame, 
    target_col: str,
    sequence_length: int = 60,
    test_size: float = 0.2
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, MinMaxScaler]:
    target = df[target_col].values.reshape(-1, 1)

    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(target)

    x, y = [], []
    for i in range(len(data_scaled) - sequence_length):
        x.append(data_scaled[i:i+sequence_length])
        y.append(data_scaled[i+sequence_length])
    
    x, y = np.array(x), np.array(y)

    train_size = int(len(x) * (1 - test_size))
    x_train, x_test = x[:train_size], x[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    return x_train, x_test, y_train, y_test, scaler
