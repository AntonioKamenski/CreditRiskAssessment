import australia
import germany

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import accuracy_score

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


australia.australian()
germany.german()

germanx, germany = germany.german()
australiax, australiay = australia.australian()

X_train_a, X_test_a, y_train_a, y_test_a = train_test_split(australiax, australiay, test_size=0.3)
X_train_g, X_test_g, y_train_g, y_test_g = train_test_split(germanx, germany, test_size=0.3)

svmGrid = {'kernel': ['linear','rbf','poly'], 'degree': range(1,5), 'C': [0.01,0.1,1,10,100]}
knnGrid = {'n_neighbors': range(1,21,2)}
rfGrid = {'bootstrap': [True, False], 'max_depth': [10, 30, 50, None], 'n_estimators': [20, 50, 100]}
mlpGrid = {'hidden_layer_sizes': [(10), (10,10), (10,10,10), (50), (50,50), (50,50,50)], 'activation': ['tanh', 'relu', 'identity', 'logistic'], 'alpha': [0.0001, 0.05], 'max_iter': [500]}
dtGrid = {'criterion': ['gini','entropy','log_loss'], 'max_depth': [10, 30, 50, None]}
lrGrid = {'C': [100, 10, 1.0, 0.1, 0.01], 'max_iter': [500]}
xgbGrid = {'n_estimators': [50, 100, 150], 'learning_rate': [0.01, 0.1,0.2, 0.3], 'max_depth': [3, 5, 7, 9, 11]}

best_params = []

model = SVC()
grid_search = GridSearchCV(model, svmGrid, cv=5, scoring='accuracy')
grid_search.fit(X_train_a, y_train_a)
best_params.append(grid_search.best_params_)
print("SVC Complete...")

model = KNeighborsClassifier()
grid_search = GridSearchCV(model, knnGrid, cv=5, scoring='accuracy')
grid_search.fit(X_train_a, y_train_a)
best_params.append(grid_search.best_params_)
print("KNN Complete...")

model = RandomForestClassifier()
grid_search = GridSearchCV(model, rfGrid, cv=5, scoring='accuracy')
grid_search.fit(X_train_a, y_train_a)
best_params.append(grid_search.best_params_)
print("RF Complete...")

model = MLPClassifier()
grid_search = GridSearchCV(model, mlpGrid, cv=5, scoring='accuracy')
grid_search.fit(X_train_a, y_train_a)
best_params.append(grid_search.best_params_)
print("MLP Complete...")

model = DecisionTreeClassifier()
grid_search = GridSearchCV(model, dtGrid, cv=5, scoring='accuracy')
grid_search.fit(X_train_a, y_train_a)
best_params.append(grid_search.best_params_)
print("DT Complete...")

model = LogisticRegression()
grid_search = GridSearchCV(model, lrGrid, cv=5, scoring='accuracy')
grid_search.fit(X_train_a, y_train_a)
best_params.append(grid_search.best_params_)
print("LR Complete...")

for params in best_params:
    print(params)

with open('best_params.txt', 'w') as file:
    file.write(str(best_params))