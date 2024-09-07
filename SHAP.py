from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import shap
import os
import numpy as np
import matplotlib.pyplot as plt

def shapValues(name, X, y, original_feature_prefixes, params):
    X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    model = SVC(**params[0])
    model.fit(X_train, y_train)
    
    background = shap.sample(X_train, 200)
    explainer = shap.KernelExplainer(model.predict_proba, background)

    X_test_sample = shap.sample(X_test, 200)

    shap_values = explainer.shap_values(X_test_sample)

    shap_values = np.transpose(shap_values, (2, 0, 1))

    means = np.mean(shap_values, axis=0)

    explanation = shap.Explanation(values=shap_values[0], base_values=explainer.expected_value[0], data=X_test_sample, feature_names=original_feature_prefixes)

    directories = ["SHAP/Summary", "SHAP/Summary Mean", "SHAP/Force", "SHAP/Waterfall", "SHAP/Decision", "SHAP/Summary Bar"]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

    shap.summary_plot(shap_values[0], X_test_sample, feature_names=original_feature_prefixes, show=False)
    plt.savefig(f'SHAP/Summary/{name} SVM.png')
    plt.close()

    shap.summary_plot(means, X_test_sample, feature_names=original_feature_prefixes, show=False)
    plt.savefig(f'SHAP/Summary Mean/{name} SVM.png')
    plt.close()

    shap.initjs()
        
    force_plot = shap.force_plot(explainer.expected_value[0], shap_values[0], X_test_sample, feature_names=original_feature_prefixes)
        
    html_file = f"SHAP/Force/{name} SVM.html"
    shap.save_html(html_file, force_plot)
    
    plt.subplots_adjust(left=0.25)
    shap.waterfall_plot(explanation[0], show = False)
    plt.savefig(f'SHAP/Waterfall/{name} SVM.png')
    plt.close()

    shap.decision_plot(explainer.expected_value[0], shap_values[0], X_test_sample, feature_names=original_feature_prefixes, show=False)
    plt.savefig(f'SHAP/Decision/{name} SVM.png')
    plt.close()

    shap.summary_plot(shap_values[0], X_test_sample, plot_type="bar", feature_names=original_feature_prefixes, show=False)
    plt.savefig(f'SHAP/Summary Bar/{name} SVM.png')
    plt.close()


    
    model = RandomForestClassifier(**params[1])
    model.fit(X_train, y_train)

    explainer = shap.TreeExplainer(model)

    X_test_sample = shap.sample(X_test, 200)

    shap_values = explainer.shap_values(X_test_sample)

    shap_values = np.transpose(shap_values, (2, 0, 1))

    shap_values_class1 = shap_values[1]

    shap.summary_plot(shap_values_class1, X_test_sample, feature_names=original_feature_prefixes, show=False)
    plt.savefig(f'SHAP/Summary/{name} RF.png')
    plt.close()


    shap.initjs()
    force_plot = shap.force_plot(explainer.expected_value[1], shap_values_class1[0], X_test_sample.iloc[0], feature_names=original_feature_prefixes)
    html_file = f"SHAP/Force/{name} RF.html"
    shap.save_html(html_file, force_plot)

    plt.subplots_adjust(left=0.25)
    explanation = shap.Explanation(values=shap_values_class1[0], base_values=explainer.expected_value[1], data=X_test_sample.iloc[0], feature_names=original_feature_prefixes)
    shap.waterfall_plot(explanation, show=False)
    plt.savefig(f'SHAP/Waterfall/{name} RF.png')
    plt.close()

    shap.decision_plot(explainer.expected_value[1], shap_values_class1, X_test_sample, feature_names=original_feature_prefixes, show=False)
    plt.savefig(f'SHAP/Decision/{name} RF.png')
    plt.close()

    shap.summary_plot(shap_values_class1, X_test_sample, plot_type="bar", feature_names=original_feature_prefixes, show=False)
    plt.savefig(f'SHAP/Summary Bar/{name}_RF_Bar.png')
    plt.close()
    


    model = LogisticRegression(**params[2])
    model.fit(X_train, y_train)

    background = shap.sample(X_train, 200)
    explainer = shap.LinearExplainer(model, background)

    X_test_sample = shap.sample(X_test, 200)

    shap_values = explainer.shap_values(X_test_sample)


    explanation = shap.Explanation(values=shap_values[0], base_values=explainer.expected_value, data=X_test_sample.iloc[0], feature_names=original_feature_prefixes)

    shap.summary_plot(shap_values, X_test_sample, feature_names=original_feature_prefixes, show=False)
    plt.savefig(f'SHAP/Summary/{name} LR.png')
    plt.close()

    shap.initjs()
        
    force_plot = shap.force_plot(explainer.expected_value, shap_values[0], X_test_sample.iloc[0,:], feature_names=original_feature_prefixes)
        
    html_file = f"SHAP/Force/{name} LR.html"
    shap.save_html(html_file, force_plot)
    

    plt.subplots_adjust(left=0.25)
    shap.waterfall_plot(explanation, show = False)
    plt.savefig(f'SHAP/Waterfall/{name} LR.png')
    plt.close()

    shap.decision_plot(explainer.expected_value, shap_values, X_test_sample, feature_names=original_feature_prefixes, show=False)
    plt.savefig(f'SHAP/Decision/{name} LR.png')
    plt.close()

    shap.summary_plot(shap_values, X_test_sample, plot_type="bar", feature_names=original_feature_prefixes, show=False)
    plt.savefig(f'SHAP/Summary Bar/{name} LR.png')
    plt.close()