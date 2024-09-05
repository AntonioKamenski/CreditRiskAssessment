import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def LRExplainer(name, X, y, features, params):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = LogisticRegression(**params)
    model.fit(X_train, y_train)

    coefficients = model.coef_[0]

    if not os.path.exists('LRCoefficients'):
        os.makedirs('LRCoefficients')

    plt.figure(figsize=(10, 6))
    plt.barh(features, coefficients, color='blue')
    plt.xlabel('Coefficient Value')
    plt.title('Logistic Regression Coefficients')
    plt.axvline(0, color='black', linewidth=0.5)
    plt.savefig(f'LRCoefficients/{name} logistic_regression_coefficients.png', dpi=300, bbox_inches='tight')

    # Optional: Print the coefficients for reference
    print("Feature Coefficients:")
    for feature, coef in zip(features, coefficients):
        print(f"{feature}: {coef:.4f}")