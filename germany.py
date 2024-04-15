import pandas as pd

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

df = pd.read_csv('datasets/german.data', header=None, delimiter='\\s+')

df.columns = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'A11', 'A12', 'A13', 'A14', 'A15', 'A16', 'A17', 'A18', 'A19', 'A20', 'A21']

for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].astype('category')

col = df.select_dtypes(include=['int64']).columns
df[col] = scaler.fit_transform(df[col])

print(df)
print(df.info())

x=df
x = x.drop('A1', axis=1)
y=df['A1']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)