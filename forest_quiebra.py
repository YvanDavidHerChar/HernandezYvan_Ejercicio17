import numpy as np
import matplotlib.pyplot as plt
import pandas as pd # para leer datos
import sklearn.ensemble # para el random forest
import sklearn.model_selection # para split train-test
import sklearn.metrics # para calcular el f1-score


df1 = pd.read_csv('1year.arff',header=None, index_col=False)
df2 = pd.read_csv('2year.arff',header=None, index_col=False)
df3 = pd.read_csv('3year.arff',header=None, index_col=False)
df4 = pd.read_csv('4year.arff',header=None, index_col=False)
df5 = pd.read_csv('5year.arff',header=None, index_col=False)
df = pd.concat([df1,df2,df3,df4,df5], axis=0, join='outer', ignore_index=True, keys=None,levels=None, names=None, verify_integrity=False, copy=True)
df =df.convert_objects(convert_numeric=True)
print(np.shape(df))
df = df.dropna()
print(np.shape(df))

aver = list(df.keys())[64]
Y=np.array(df[aver])
df = df.drop([aver],axis=1)
predictors =list(df.keys())

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(df[predictors], Y, test_size=0.5)
X_test, X_validation, y_test, y_validation = sklearn.model_selection.train_test_split(X_test, y_test, test_size=0.2)
print(np.shape(X_test))


n_trees = np.arange(1,400,25)
f1_train = []
f1_test = []
feature_importance = np.zeros((len(n_trees), len(predictors)))

for i, n_tree in enumerate(n_trees):
    clf = sklearn.ensemble.RandomForestClassifier(n_estimators=n_tree, max_features='sqrt')
    clf.fit(X_train, y_train)
    f1_train.append(sklearn.metrics.f1_score(y_train, clf.predict(X_train)))
    f1_test.append(sklearn.metrics.f1_score(y_test, clf.predict(X_test)))
    feature_importance[i, :] = clf.feature_importances_

plt.scatter(n_trees, f1_test)
plt.savefig('features.png')