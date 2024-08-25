import germany
import warnings
import australia
from modelAnalisys import modelanalisys
from bestParams import bestParamsToTxt
from feature_importance import PDP
from SHAP import shapValues
from sklearn.exceptions import ConvergenceWarning


warnings.filterwarnings("ignore", category=ConvergenceWarning)

australia_x, australia_y = australia.australian()
germany_x, germany_y = germany.german()

randValues = [11, 20, 25, 39, 42, 50, 55, 60, 91, 100]


#bestParamsToTxt("australia", australia_x, australia_y)
#bestParamsToTxt("germany", germany_x, germany_y)

#modelanalisys("australia", randValues, australia_x, australia_y)
#modelanalisys("germany", randValues, germany_x, germany_y)


australianParams = [
    {'C': 0.1, 'degree': 4, 'kernel': 'poly', 'probability': True},
    {'bootstrap': True, 'max_depth': 30, 'n_estimators': 50},
    {'C': 0.01, 'max_iter': 500},
]

germanyParams = [
    {'C': 100, 'degree': 1, 'kernel': 'linear', 'probability': True},
    {'bootstrap': True, 'max_depth': 10, 'n_estimators': 100},
    {'C': 10, 'max_iter': 500},
]

featuresAustralia = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'A11', 'A12', 'A13', 'A14']
featuresGermany = ['Attribute1', 'Attribute2', 'Attribute3', 'Attribute4', 'Attribute5', 'Attribute6', 'Attribute7', 'Attribute8', 'Attribute9', 'Attribute10', 'Attribute11', 'Attribute12', 'Attribute13', 'Attribute14', 'Attribute15', 'Attribute16', 'Attribute17', 'Attribute18', 'Attribute19', 'Attribute20']

selectedAustralia = ['A2', 'A3', 'A7']
selectedGermany = ['Attribute2', 'Attribute5', 'Attribute8', 'Attribute11', 'Attribute13', 'Attribute16']

PDP("australia", australia_x, australia_y, australianParams)
PDP("germany", germany_x, germany_y, germanyParams)

shapValues("australia", australia_x, australia_y, featuresAustralia, australianParams)
shapValues("germany", germany_x, germany_y, featuresGermany, germanyParams)