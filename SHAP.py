from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import PartialDependenceDisplay
from sklearn.model_selection import train_test_split
import shap

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

def shapValues(name, X, y, original_feature_prefixes, params):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = SVC(probability=True, random_state=42)
    model.fit(X_train, y_train)

    background = shap.sample(X_train, 100) #Smanjen broj uzoraka radi brze izvedbe programa

    explainer = shap.KernelExplainer(model.predict_proba, background)

    X_test_sample = shap.sample(X_test, 10) #Postavljen je mali broj u svrhe brzeg testiranja

    shap_values = explainer.shap_values(X_test_sample)

    print(f"Shape of shap_values[0]: {np.array(shap_values[0]).shape}")
    print(f"Shape of shap_values[1]: {np.array(shap_values[1]).shape}")
    print(f"Shape of X_test_sample: {X_test_sample.shape}")

    shap.summary_plot(shap_values)
    shap.summary_plot(shap_values[0], X_test_sample , feature_names=original_feature_prefixes)
    shap.summary_plot(shap_values[1], X_test_sample , feature_names=original_feature_prefixes)
    shap.power_plot(explainer.expected_value[0], shap_values[0], feature_names=original_feature_prefixes)
    #Zadnja 3 uopce ne rade, a prvi ne izgleda kako treba, svugdje na izvorima gdje sam trazio se rjesava na identican nacin...
    plt.close()