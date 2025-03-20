import pandas as pd

df = pd.read_csv("../dataset/students.csv", sep=";")

print(df.isna().sum())
