from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.inspection import PartialDependenceDisplay
import matplotlib.pyplot as plt

def PDP(name, X, y, params):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    model = SVC(**params[0])
    model.fit(X_train, y_train)

    perm_importance = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)

    feature_importances = perm_importance.importances_mean

    sorted_idx = perm_importance.importances_mean.argsort()

    for i in sorted_idx:
        print(f"{X.columns[i]}: {perm_importance.importances_mean[i]:.4f}")

    plt.figure(figsize=(10, 6))
    plt.barh(X.columns[sorted_idx], perm_importance.importances_mean[sorted_idx])
    plt.xlabel("Permutation Importance")
    plt.title("Feature Importance using Permutation Importance")
    plt.show()


    PartialDependenceDisplay.from_estimator(model, X_train, features=X.columns, grid_resolution=100)
    plt.savefig(f'Partial Dependence/PDP {name} SVM.png')

    model = RandomForestClassifier(**params[1])
    model.fit(X_train, y_train)

    perm_importance = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)

    feature_importances = perm_importance.importances_mean
    feature_names = X.columns

    plt.figure(figsize=(10, 6))
    plt.barh(X.columns[sorted_idx], perm_importance.importances_mean[sorted_idx])
    plt.xlabel("Permutation Importance")
    plt.title("Feature Importance using Permutation Importance")
    plt.show()

    display = PartialDependenceDisplay.from_estimator(model, X_test, features=X.columns)
    plt.savefig(f'Partial Dependence/PDP {name} Random Forest.png')

    model = LogisticRegression(**params[2])
    model.fit(X_train, y_train)

    perm_importance = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)

    feature_importances = perm_importance.importances_mean
    feature_names = X.columns

    plt.figure(figsize=(10, 6))
    plt.barh(X.columns[sorted_idx], perm_importance.importances_mean[sorted_idx])
    plt.xlabel("Permutation Importance")
    plt.title("Feature Importance using Permutation Importance")
    plt.show()

    display = PartialDependenceDisplay.from_estimator(model, X_test, features=X.columns)
    plt.savefig(f'Partial Dependence/PDP {name} Logistic Regression.png')