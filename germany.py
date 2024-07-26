import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from ucimlrepo import fetch_ucirepo 

def german():

    statlog_german_credit_data = fetch_ucirepo(id=144) 
    
    X = statlog_german_credit_data.data.features 
    y = statlog_german_credit_data.data.targets 

    X = pd.DataFrame(X)
    y = pd.DataFrame(y)

    X = pd.get_dummies(X, columns=X.select_dtypes(include=['object']).columns)

    y['class'] = y['class'].replace({2: 0, 1: 1})

    y = y.values.ravel()

    scaler = MinMaxScaler()
    int_columns = X.select_dtypes(include=['int64']).columns
    X[int_columns] = scaler.fit_transform(X[int_columns])

    return X, y