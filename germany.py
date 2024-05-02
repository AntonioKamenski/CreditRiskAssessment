import pandas as pd

from sklearn.preprocessing import MinMaxScaler

def german():
    scaler = MinMaxScaler()

    df = pd.read_csv('datasets/german.data', header=None, delimiter='\\s+')

    df.columns = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'A11', 'A12', 'A13', 'A14', 'A15', 'A16', 'A17', 'A18', 'A19', 'A20', 'A21']

    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype('category')

    df['A21'] = df['A21'].astype('category')

    col = df.select_dtypes(include=['int64']).columns
    df[col] = scaler.fit_transform(df[col])

    x=df
    x = x.drop('A21', axis=1)
    y=df['A21']

    print(df.info())

    return x, y