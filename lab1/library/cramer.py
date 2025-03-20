import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn
from scipy.stats import chi2_contingency


def cramer_v(x, y):
    confusion_matrix = pd.crosstab(x, y)
    chi2, _, _, _ = chi2_contingency(confusion_matrix)
    n = confusion_matrix.sum().sum()
    r, k = confusion_matrix.shape
    return np.sqrt(chi2 / n / min((k-1), (r-1)))


def main():
    df = pd.read_csv("../../dataset/students.csv", sep=";")

    for col in ['G1', 'G2', 'G3']:
        df[col] = pd.cut(
            df[col],
            bins=[0, 9, 11, 13, 15, 20],
            include_lowest=True
        )

    columns = df.columns

    matrix = pd.DataFrame(
        data=np.zeros((len(columns), len(columns))),
        index=columns,
        columns=columns
    )

    for column1 in columns:
        for column2 in columns:
            matrix.loc[column1, column2] = cramer_v(df[column1], df[column2])

    plt.figure(figsize=(20, 20))
    seaborn.heatmap(matrix, annot=True, fmt=".2f", vmin=0, vmax=1)
    plt.show()


if __name__ == '__main__':
    main()
