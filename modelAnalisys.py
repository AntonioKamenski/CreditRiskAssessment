from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix
from sklearn.model_selection import train_test_split

import pandas as pd

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB

def modelanalisys(name, randValues, X, y):
    with open(f'parameters/{name}.txt', 'r') as file:
        text = file.read()

    # Strip the outer brackets and split the entries
    text = text.strip()[2:-2]  # Removes the outer [[ ]]
    entries = text.split('}, ')  # Splits the dictionary entries

    # Fix the last entry
    entries[-1] = entries[-1].rstrip("}]")

    # Add back the closing brace for each dictionary entry
    entries = [entry + '}' if not entry.endswith('}') else entry for entry in entries]

    # Parse each dictionary or string
    parsed_list = []
    for entry in entries:
        if entry.startswith('{'):
            parsed_list.append(eval(entry))
        else:
            parsed_list.append(entry.strip("'"))

    SVMparams = parsed_list[0]
    KNNparams = parsed_list[1]
    RFparams = parsed_list[2]
    MLPparams = parsed_list[3]
    DTparams = parsed_list[4]
    LRparams = parsed_list[5]
    XGBparams = parsed_list[6]

    collumns = ["SVM", "KNN", "RF", "MLP", "DT", "LR", "XGB", "GNB"]
    excel = pd.DataFrame(columns=collumns)

    for i in range(len(randValues)):
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=randValues[i])
        
        f1_scores = []

        model = SVC(**SVMparams)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        f1_scores.append(f1_score(y_test, y_pred))

        model = KNeighborsClassifier(**KNNparams)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        f1_scores.append(f1_score(y_test, y_pred))

        model = RandomForestClassifier(**RFparams)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        f1_scores.append(f1_score(y_test, y_pred))

        model = MLPClassifier(**MLPparams)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        f1_scores.append(f1_score(y_test, y_pred))

        model = DecisionTreeClassifier(**DTparams)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)  
        f1_scores.append(f1_score(y_test, y_pred))

        model = LogisticRegression(**LRparams)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)  
        f1_scores.append(f1_score(y_test, y_pred))

        model = XGBClassifier(**XGBparams)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)  
        f1_scores.append(f1_score(y_test, y_pred))

        model = GaussianNB()
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)  
        f1_scores.append(f1_score(y_test, y_pred))

        #print(f1_scores)

        excel.loc[i] = f1_scores

    excel.to_excel(f'Excel/{name}.xlsx')