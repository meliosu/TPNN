import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn


def cramer_v(xs, ys):
    n = len(xs)

    xy_freq = pd.crosstab(xs, ys)
    rows, cols = xy_freq.index, xy_freq.columns

    x_freq = xs.value_counts()
    y_freq = ys.value_counts()

    chi2 = 0

    for row in rows:
        for col in cols:
            chi2 += (
                (xy_freq.loc[row, col] - (x_freq[row] * y_freq[col]) / n) ** 2
                / ((x_freq[row] * y_freq[col]) / n)
            )

    phi2 = chi2 / n

    return math.sqrt(phi2 / min(len(rows) - 1, len(cols) - 1))


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
