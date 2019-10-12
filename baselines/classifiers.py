from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.svm import SVC as SVM
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


# logistic regression
lr = LogisticRegression(C = 0.01, dual = False, penalty='l1', verbose=100, max_iter=5000)
 # nearest neibors classification
knn = KNeighborsClassifier(n_neighbors=5, algorithm='auto')
# xgboost
xgb = XGBClassifier(min_child_weight=7, n_estimators =160, nthread=1, subsample=0.8)
# c-support vector classification
svm = SVM(probability=True, max_iter= 1000)
 # random forest
# rf = RandomForestClassifier(n_estimators = 160, criterion = 'entropy', max_features = 'sqrt')
rf = RandomForestClassifier(warm_start=True, n_estimators = 160, criterion = 'entropy', max_features = 'sqrt')
