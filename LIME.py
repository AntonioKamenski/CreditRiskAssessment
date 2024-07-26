from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import lime.lime_tabular
import lime
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

iris = load_iris()

def limeExplainer(name, X, y, params):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = SVC(**params[0], probability=True)
    model.fit(X_train, y_train)

    iris = load_iris()

    instance_idx = 0
    instance = X_test.iloc[instance_idx]

    feature_names = X.columns.tolist()

    explainer = lime.lime_tabular.LimeTabularExplainer(
        X_train.values,
        feature_names=feature_names,
        class_names=['setosa', 'versicolor', 'virginica'],
        discretize_continuous=True
    )

    explanation = explainer.explain_instance(instance.values, model.predict_proba, num_features=2)

    explanation.show_in_notebook(show_table=True, show_all=False)

    explanation.save_to_file('lime_explanation.html')