from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import os 
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.inspection import PartialDependenceDisplay
import matplotlib.pyplot as plt

def PDP(name, X, y, params, features):
    old_names = X.columns.tolist()

    rename_dict = dict(zip(old_names, features))

    X = X.rename(columns=rename_dict)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = SVC(**params[0])
    model.fit(X_train, y_train)

    perm_importance = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)

    sorted_idx = perm_importance.importances_mean.argsort()

    for i in sorted_idx:
        print(f"{X.columns[i]}: {perm_importance.importances_mean[i]:.4f}")

    plt.figure(figsize=(10, 6))
    plt.barh(X.columns[sorted_idx], perm_importance.importances_mean[sorted_idx])
    plt.xlabel("Permutation Importance")
    plt.title("Feature Importance using Permutation Importance")
    plt.savefig(f'Permutation Importance/PI for {name} using SVM model.png')
    plt.close()

    if not os.path.exists(f'Partial Dependence/PDP for {name} using SVM model'):
        os.makedirs(f'Partial Dependence/PDP for {name} using SVM model')

    for feature in X.columns:
        display = PartialDependenceDisplay.from_estimator(
            model, X_train, features=[feature], grid_resolution=100
        )
        plt.title(f'Partial Dependence Plot for {feature}')
        plt.savefig(f'Partial Dependence/PDP for {name} using SVM model/{feature}.png')
        plt.close()



    model = RandomForestClassifier(**params[1])
    model.fit(X_train, y_train)

    perm_importance = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)

    sorted_idx = perm_importance.importances_mean.argsort()

    plt.figure(figsize=(10, 6))
    plt.barh(X.columns[sorted_idx], perm_importance.importances_mean[sorted_idx])
    plt.xlabel("Permutation Importance")
    plt.title("Feature Importance using Permutation Importance")
    plt.savefig(f'Permutation Importance/PI for {name} using Random Forest model.png')
    plt.close()

    if not os.path.exists(f'Partial Dependence/PDP for {name} using Random Forest model'):
        os.makedirs(f'Partial Dependence/PDP for {name} using Random Forest model')

    for feature in X.columns:
        display = PartialDependenceDisplay.from_estimator(
            model, X_train, features=[feature], grid_resolution=100
        )
        plt.title(f'Partial Dependence Plot for {feature}')
        plt.savefig(f'Partial Dependence/PDP for {name} using Random Forest model/{feature}.png')
        plt.close()

    model = LogisticRegression(**params[2])
    model.fit(X_train, y_train)

    perm_importance = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)

    sorted_idx = perm_importance.importances_mean.argsort()

    plt.figure(figsize=(10, 6))
    plt.barh(X.columns[sorted_idx], perm_importance.importances_mean[sorted_idx])
    plt.xlabel("Permutation Importance")
    plt.title("Feature Importance using Permutation Importance")
    plt.savefig(f'Permutation Importance/PI for {name} using Logistic Regression model.png')
    plt.close()

    if not os.path.exists(f'Partial Dependence/PDP for {name} using Logistic Regression model'):
        os.makedirs(f'Partial Dependence/PDP for {name} using Logistic Regression model')

    for feature in X.columns:
        display = PartialDependenceDisplay.from_estimator(
            model, X_train, features=[feature], grid_resolution=100
        )
        plt.title(f'Partial Dependence Plot for {feature}')
        plt.savefig(f'Partial Dependence/PDP for {name} using Logistic Regression model/{feature}.png')
        plt.close()