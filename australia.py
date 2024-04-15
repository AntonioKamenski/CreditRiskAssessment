import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV

df = pd.read_csv('datasets/australian.dat', header=None, delimiter='\\s+')

df.columns = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'A11', 'A12', 'A13', 'A14', 'A15']

categorical_columns = ['A1', 'A4', 'A5', 'A6', 'A8', 'A9', 'A11', 'A12', 'A15']
df[categorical_columns] = df[categorical_columns].astype('category')

scaler = StandardScaler()

numerical_columns = ['A2', 'A3', 'A7', 'A10', 'A13', 'A14']
df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

print(df['A1'].info())  
print(df.info())

x=df
x = x.drop('A1', axis=1)
y=df['A1']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)