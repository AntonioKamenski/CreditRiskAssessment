import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from treeinterpreter import treeinterpreter as ti
import matplotlib.pyplot as plt

def RFContributions(name, X, y, features, params, featureNames):  
    output_dir = 'RFContributions'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = RandomForestClassifier(**params)
    model.fit(pd.DataFrame(X_train, columns=features), y_train)

    explanations = []
    contributions_list = []

    n = 20  # Number of instances to explain

    for i in range(n):
        prediction, bias, contributions = ti.predict(model, pd.DataFrame(X_test[i:i+1], columns=features))

        explanation = {
            'Instance': i,
            'Prediction': prediction[0][0],
            'Bias': bias[0][0],
        }

        instance_contributions = []
        for feature, contribution in zip(featureNames, contributions[0]):
            explanation[feature] = contribution[0]
            instance_contributions.append(contribution[0])

        contributions_list.append(instance_contributions)
        explanations.append(explanation)


    explanations_df = pd.DataFrame(explanations)
    with pd.ExcelWriter(f'{output_dir}/{name} RF Explanations.xlsx') as writer:
        explanations_df.to_excel(writer, index=False, merge_cells=True)

    contributions_df = pd.DataFrame(contributions_list, columns=featureNames)
    avg_contributions = contributions_df.mean()

    print(contributions_df)

    plt.figure(figsize=(10, 6))
    avg_contributions.plot(kind='barh')
    plt.title(f'Average Feature Contributions for {name}')
    plt.ylabel('Average Contribution')
    plt.tight_layout()

    # Save the plot in the same output directory
    plot_path = f'{output_dir}/{name} RF Contributions.png'
    plt.savefig(plot_path)
    plt.close()