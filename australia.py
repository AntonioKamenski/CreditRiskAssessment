import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from ucimlrepo import fetch_ucirepo 

def australian():

    statlog_australian_credit_approval = fetch_ucirepo(id=143) 
    
    X = statlog_australian_credit_approval.data.features 
    y = statlog_australian_credit_approval.data.targets 

    X = pd.DataFrame(X)
    y = pd.DataFrame(y)
    
    int_columns = X.select_dtypes(include=['int64']).columns
    X[int_columns] = X[int_columns].astype('object')

    X = pd.get_dummies(X, columns=X.select_dtypes(include=['object']).columns)

    y['A15'] = y['A15'].astype('category')

    y = y.values.ravel()

    scaler = MinMaxScaler()
    int_columns = X.select_dtypes(include=['float64']).columns
    X[int_columns] = scaler.fit_transform(X[int_columns])

    return X, y