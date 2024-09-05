import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from treeinterpreter import treeinterpreter as ti
import matplotlib.pyplot as plt

def RFContributions(name, X, y, features, params):  
    output_dir = 'RFContributions'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = RandomForestClassifier(**params)
    model.fit(pd.DataFrame(X_train, columns=features), y_train)

    explanations = []

    n = 20  # Number of instances to explain

    for i in range(n):
        prediction, bias, contributions = ti.predict(model, pd.DataFrame(X_test[i:i+1], columns=features))

        explanation = {
            'Instance': i,
            'Prediction': prediction[0][0],
            'Bias': bias[0][0],
        }

        for feature, contribution in zip(features, contributions[0]):
            explanation[feature] = contribution[0]
        
        explanations.append(explanation)

    explanations_df = pd.DataFrame(explanations)

    with pd.ExcelWriter(f'{output_dir}/{name} RF Explanations.xlsx') as writer:
        explanations_df.to_excel(writer, index=False, merge_cells=True)