import pandas as pd


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
