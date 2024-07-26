from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix
import time

import pandas as pd
import numpy as np

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB

def bestParamsToTxt(name, X, y, randValues):
    algorythms = ["SVM", "KNN", "RF", "MLP", "DT", "LR", "XGBoost", "GNB"]
    svmGrid = {'kernel': ['linear','rbf','poly'], 'degree': range(1,5), 'C': [0.01,0.1,1,10,100]}
    knnGrid = {'n_neighbors': range(1,21,2)}
    rfGrid = {'bootstrap': [True, False], 'max_depth': [10, 30, 50, None], 'n_estimators': [20, 50, 100]}
    mlpGrid = {'hidden_layer_sizes': [(10), (10,10), (10,10,10), (50), (50,50), (50,50,50)], 'activation': ['tanh', 'relu', 'identity', 'logistic'], 'alpha': [0.0001, 0.05], 'max_iter': [500]}
    dtGrid = {'criterion': ['gini','entropy','log_loss'], 'max_depth': [10, 30, 50, None]}
    lrGrid = {'C': [100, 10, 1.0, 0.1, 0.01], 'max_iter': [500]}
    xgbGrid = {'n_estimators': [50, 100, 150], 'learning_rate': [0.01, 0.1,0.2, 0.3], 'max_depth': [3, 5, 7, 9, 11]}

    best_params = []

    columns = ["Round", "SVM", "KNN", "RF", "MLP", "DT", "LR", "XGBoost", "GNB"]

    f1_excel = []

    for i in range(10):
        start_time = time.time()

        print(f"Starting analysis for round {i+1}...")

        params = []
        y_pred = []

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=randValues[i])

        model = SVC()
        SVC_grid_search = GridSearchCV(model, svmGrid, cv=5, scoring='accuracy')
        SVC_grid_search.fit(X_train, y_train)
        params.append(SVC_grid_search.best_params_)
        print("SVC Analysis Complete...")
        y_pred.append(SVC_grid_search.predict(X_test))

        model = KNeighborsClassifier()
        KNN_grid_search = GridSearchCV(model, knnGrid, cv=5, scoring='accuracy')
        KNN_grid_search.fit(X_train, y_train)
        params.append(KNN_grid_search.best_params_)
        print("KNN Analysis Complete...")
        y_pred.append(KNN_grid_search.predict(X_test))

        model = RandomForestClassifier()
        RF_grid_search = GridSearchCV(model, rfGrid, cv=5, scoring='accuracy')
        RF_grid_search.fit(X_train, y_train)
        params.append(RF_grid_search.best_params_)
        print("RF Analysis Complete...")
        y_pred.append(RF_grid_search.predict(X_test))

        model = MLPClassifier()
        MLP_grid_search = GridSearchCV(model, mlpGrid, cv=5, scoring='accuracy')
        MLP_grid_search.fit(X_train, y_train)
        params.append(MLP_grid_search.best_params_)
        print("MLP Analysis Complete...")
        y_pred.append(MLP_grid_search.predict(X_test))

        model = DecisionTreeClassifier()
        DT_grid_search = GridSearchCV(model, dtGrid, cv=5, scoring='accuracy')
        DT_grid_search.fit(X_train, y_train)
        params.append(DT_grid_search.best_params_)
        print("DT Analysis Complete...")
        y_pred.append(DT_grid_search.predict(X_test))

        model = LogisticRegression()
        LR_grid_search = GridSearchCV(model, lrGrid, cv=5, scoring='accuracy')
        LR_grid_search.fit(X_train, y_train)
        params.append(LR_grid_search.best_params_)
        print("LR Analysis Complete...")
        y_pred.append(LR_grid_search.predict(X_test))

        model = XGBClassifier(enable_categorical=True)
        XGB_grid_search = GridSearchCV(model, xgbGrid, cv=5, scoring='accuracy')
        XGB_grid_search.fit(X_train, y_train)
        params.append(XGB_grid_search.best_params_)
        print("XGB Analysis Complete...")
        y_pred.append(XGB_grid_search.predict(X_test))

        GNBmodel = GaussianNB()
        GNBmodel.fit(X_train, y_train)
        params.append("GaussianNB has no params to search...")
        print("GaussianNB Analysis Complete...")
        y_pred.append(GNBmodel.predict(X_test))

        print("========================================================================================")

        best_params.append(params)

        f1_array = []
        f1_array.append(i+1)

        for j in range(len(algorythms)):
            print(algorythms[j])

            conf_matrix = confusion_matrix(y_test, y_pred[j])

            accuracy = accuracy_score(y_test, y_pred[j])
            f1 = f1_score(y_test, y_pred[j])
            f1_array.append(f1)
            recall = recall_score(y_test, y_pred[j])
            precision = precision_score(y_test, y_pred[j])

            print("Confusion Matrix:")
            print(conf_matrix)
            print("Accuracy:", accuracy)
            print("F1 Score:", f1)
            print("Recall (True Positive Rate):", recall)
            print("Precision:", precision)
            print("========================================================================================")

        f1_excel.append(f1_array)

        end_time = time.time()
        execution_time = end_time - start_time

        print(f"Execution time: {execution_time} seconds")
        print(f"Round {i+1} Complete...")
        print("========================================================================================")

    with open(f'{name}.txt', 'w') as file:
        file.write(str(best_params))

    f1_excel = pd.DataFrame(f1_excel, columns=columns)

    excel_file = f'{name}.xlsx'
    f1_excel.to_excel(excel_file, index=False)

    print(f"Excel file '{excel_file}' has been created with the data in a table format.")