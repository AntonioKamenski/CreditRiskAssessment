from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB

def bestParamsToTxt(name, X, y):
    svmGrid = {'kernel': ['linear','rbf','poly'], 'degree': range(1,5), 'C': [0.01,0.1,1,10,100]}
    knnGrid = {'n_neighbors': range(1,21,2)}
    rfGrid = {'bootstrap': [True, False], 'max_depth': [10, 30, 50, None], 'n_estimators': [20, 50, 100]}
    mlpGrid = {'hidden_layer_sizes': [(10), (10,10), (10,10,10), (50), (50,50), (50,50,50)], 'activation': ['tanh', 'relu', 'identity', 'logistic'], 'alpha': [0.0001, 0.05], 'max_iter': [500]}
    dtGrid = {'criterion': ['gini','entropy','log_loss'], 'max_depth': [10, 30, 50, None]}
    lrGrid = {'C': [100, 10, 1.0, 0.1, 0.01], 'max_iter': [500]}
    xgbGrid = {'n_estimators': [50, 100, 150], 'learning_rate': [0.01, 0.1,0.2, 0.3], 'max_depth': [3, 5, 7, 9, 11]}

    best_params = []
    params = []

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"Starting Analysis for {name}")

    model = SVC()
    SVC_grid_search = GridSearchCV(
        model,
        svmGrid,
        cv=5,
        scoring='accuracy'
    )
    SVC_grid_search.fit(X_train, y_train)
    params.append(SVC_grid_search.best_params_)
    print("SVC Analysis Complete...")
    
    model = KNeighborsClassifier()
    KNN_grid_search = GridSearchCV(
        model,
        knnGrid,
        cv=5,
        scoring='accuracy'
    )
    KNN_grid_search.fit(X_train, y_train)
    params.append(KNN_grid_search.best_params_)
    print("KNN Analysis Complete...")

    model = RandomForestClassifier()
    RF_grid_search = GridSearchCV(
        model,
        rfGrid,
        cv=5,
        scoring='accuracy'
    )
    RF_grid_search.fit(X_train, y_train)
    params.append(RF_grid_search.best_params_)
    print("RF Analysis Complete...")

    model = MLPClassifier()
    MLP_grid_search = GridSearchCV(
        model,
        mlpGrid,
        cv=5,
        scoring='accuracy'
    )
    MLP_grid_search.fit(X_train, y_train)
    params.append(MLP_grid_search.best_params_)
    print("MLP Analysis Complete...")

    model = DecisionTreeClassifier()
    DT_grid_search = GridSearchCV(
        model,
        dtGrid,
        cv=5,
        scoring='accuracy'
    )
    DT_grid_search.fit(X_train, y_train)
    params.append(DT_grid_search.best_params_)
    print("DT Analysis Complete...")

    model = LogisticRegression()
    LR_grid_search = GridSearchCV(
        model,
        lrGrid,
        cv=5,
        scoring='accuracy'
    )
    LR_grid_search.fit(X_train, y_train)
    params.append(LR_grid_search.best_params_)
    print("LR Analysis Complete...")

    model = XGBClassifier(enable_categorical=True)
    XGB_grid_search = GridSearchCV(
        model,
        xgbGrid,
        cv=5,
        scoring='accuracy'
    )
    XGB_grid_search.fit(X_train, y_train)
    params.append(XGB_grid_search.best_params_)
    print("XGB Analysis Complete...")

    GNBmodel = GaussianNB()
    GNBmodel.fit(X_train, y_train)
    params.append("GaussianNB has no params to search...")
    print("GaussianNB Analysis Complete...")

    print("========================================================================================")

    best_params.append(params)

    with open(f'parameters/{name}.txt', 'w') as file:
        file.write(str(best_params))