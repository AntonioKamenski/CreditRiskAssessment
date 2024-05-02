import pandas as pd

from sklearn.preprocessing import MinMaxScaler

def australian():
    df = pd.read_csv('datasets/australian.dat', header=None, delimiter='\\s+')

    df.columns = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'A11', 'A12', 'A13', 'A14', 'A15']

    categorical_columns = ['A1', 'A4', 'A5', 'A6', 'A8', 'A9', 'A11', 'A12', 'A15']
    df[categorical_columns] = df[categorical_columns].astype('category')

    scaler = MinMaxScaler()
    numerical_columns = ['A2', 'A3', 'A7', 'A10', 'A13', 'A14']
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

    x = df.copy()
    x = x.drop('A15', axis=1)
    y = df['A15']

    return x, y