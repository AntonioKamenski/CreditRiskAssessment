import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn import svm
from sklearn.model_selection import train_test_split
from lime.lime_tabular import LimeTabularExplainer

def lime(name, X, y, features, params):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    if not os.path.exists('LIME'):
        os.makedirs('LIME')
    
    if isinstance(X_train, pd.DataFrame):
        X_train = X_train.to_numpy()
        X_test = X_test.to_numpy()

    model = svm.SVC(**params)
    model.fit(X_train, y_train)

    unique_classes = np.unique(y)
    class_names = [str(cls) for cls in unique_classes]

    explainer = LimeTabularExplainer(
        training_data=X_train,
        feature_names=features,
        class_names=class_names,
        mode='classification'
    )

    num_instances = 20

    feature_importances = defaultdict(float)

    for i, instance in enumerate(X_test[:num_instances]):
        explanation = explainer.explain_instance(
            data_row=instance,
            predict_fn=model.predict_proba
        )

        print(f"Explanation for instance {i + 1}:")
        for feature, weight in explanation.as_list():
            print(f"{feature}: {weight:.4f}")
            feature_importances[feature] += weight

    for feature in feature_importances:
        feature_importances[feature] /= num_instances

    sorted_features = sorted(feature_importances.items(), key=lambda x: abs(x[1]), reverse=True)
    features, importances = zip(*sorted_features)

    plt.figure(figsize=(10, 6))
    plt.barh(features, importances, color='skyblue')
    plt.xlabel('Average Feature Importance')
    plt.title(f'Average LIME Feature Importance for {num_instances} Instances')
    plt.gca().invert_yaxis()  # To display the most important features at the top
    plt.savefig(f'LIME/{name}_LIME_Average_Analysis.png', dpi=300, bbox_inches='tight')
    plt.show()