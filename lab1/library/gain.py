import math

import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from scipy.stats import entropy


def main():
    df = pd.read_csv('../../dataset/students.csv', sep=';')
    X = df.drop('G3', axis=1)
    y = pd.cut(df['G3'], bins=5, labels=False)

    X_processed = pd.DataFrame()
    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            X_processed[col] = pd.cut(X[col], bins=5, labels=False)
        else:
            X_processed[col] = X[col].astype('category').cat.codes

    info_gain = mutual_info_classif(X_processed, y, discrete_features=True)

    gain_ratios = {}
    for info_gain, col in zip(info_gain, X_processed.columns):
        value_counts = X_processed[col].value_counts(normalize=True).values
        intrinsic_info = entropy(value_counts, base=2)
        gain_ratios[col] = math.log2(math.e) * info_gain / intrinsic_info

    for feature, ratio in sorted(gain_ratios.items(), key=lambda x: x[1], reverse=True):
        print(f"Gain Ratio ({feature}) = {ratio:.5f}")


if __name__ == '__main__':
    main()
    