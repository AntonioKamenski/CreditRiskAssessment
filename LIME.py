import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split
from lime.lime_tabular import LimeTabularExplainer

def lime(name, X, y, features, params):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
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

    instance = X_test[0]

    explanation = explainer.explain_instance(
        data_row=instance,
        predict_fn=model.predict_proba
    )

    print("Explanation for the instance:")
    for feature, weight in explanation.as_list():
        print(f"{feature}: {weight:.4f}")

    fig = explanation.as_pyplot_figure()
    plt.title('LIME Explanation')
    plt.show()
