import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from minepy import MINE
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler

data_train = pd.read_hdf("train.h5", key="train")
test = pd.read_hdf("test.h5", key="test")
X_train=data_train.iloc[:,1:]
X_train=np.array(X_train)
Y_train=data_train.iloc[:,0]
Y_train=np.array(Y_train)

selector=SelectKBest(mutual_info_classif, k=100)
X_train = selector.fit_transform(X_train, Y_train)
ss = StandardScaler()
#ss=MinMaxScaler()
X_train = ss.fit_transform(X_train)

accuracy=0
for i in range(1,21):
    seed=np.random.randint(0,100)
    test_size = 0.2
    X_tr, X_te, y_tr, y_te = train_test_split(X_train, Y_train, test_size=test_size, random_state=seed)
    #model=SVC(C=17.0, kernel='rbf', degree=5, gamma='auto', coef0=0.0, shrinking=True, probability=False, tol=0.0001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr', random_state=None)
    model = XGBClassifier(max_depth=8, learning_rate=0.1, n_estimators=1000,objective='multi:softmax')
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    predictions = [round(value) for value in y_pred]
    print(accuracy_score(y_te, predictions))
    accuracy =accuracy+ accuracy_score(y_te, predictions)

print("Accuracy: %.2f%%" % (accuracy * 5.0))
