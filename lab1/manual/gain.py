import pandas as pd
import numpy as np


def is_numeric(column):
    return pd.api.types.is_numeric_dtype(column)


def entropy(target):
    _, counts = np.unique(target, return_counts=True)
    probabilities = counts / len(target)
    return -np.sum(probabilities * np.log2(probabilities))


def information_gain(feature, target, target_entropy):
    if is_numeric(feature):
        feature_binned = pd.cut(feature, bins=5, labels=False)
    else:
        feature_binned = feature

    weighted_entropy = 0

    for bin in np.unique(feature_binned):
        subset = target[feature_binned == bin]
        subset_entropy = entropy(subset)
        weighted_entropy += (len(subset) / len(target)) * subset_entropy

    return target_entropy - weighted_entropy


def intrinsic_information(feature):
    if is_numeric(feature):
        feature_binned = pd.cut(feature, bins=5, labels=False)
    else:
        feature_binned = feature

    return entropy(feature_binned)


def gain_ratio(feature, target, target_entropy):
    return information_gain(feature, target, target_entropy) / intrinsic_information(feature)


def main():
    df = pd.read_csv("../../dataset/students.csv", sep=";")
    df['score'] = pd.cut(df['G3'], bins=5, labels=False)

    target_entropy = entropy(df['score'])

    ratios = [
        (column, gain_ratio(df[column], df['score'], target_entropy))
        for column in df.columns[:-2]
    ]

    for column, ratio in sorted(ratios, key=lambda x: x[1], reverse=True):
        print(f"Gain Ratio ({column}) = {ratio:.5f}")


if __name__ == '__main__':
    main()

