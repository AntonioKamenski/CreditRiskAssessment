import germany
import warnings
import australia
from bestParams import bestParamsToTxt
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

australia_x, australia_y = australia.australian()
germany_x, germany_y = germany.german()

randValues = [11, 20, 25, 39, 42, 50, 55, 60, 91, 100]

bestParamsToTxt("australia", australia_x, australia_y, randValues)
bestParamsToTxt("germany", germany_x, germany_y, randValues)