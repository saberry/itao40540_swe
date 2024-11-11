# Columns needed:

# Index(['age', 'numberPriorJobs', 'proportion401K', 'startingSalary',
       #'currentSalary', 'performance', 'monthsToSeparate', 'workDistance',
       #'department_1', 'department_2', 'department_3']

from joblib import load, dump
import numpy as np
from sklearn import linear_model
from sklearn.metrics import accuracy_score, roc_auc_score, balanced_accuracy_score, matthews_corrcoef

X_train_scaled, X_test_scaled, y_train, y_test = load(
  
  )

lasso_model = linear_model.LogisticRegression(penalty='l1', solver='liblinear')

lasso_model.fit(X_train_scaled, y_train)

dump(lasso_model, 'C:/Users/sberry5/Documents/teaching/level-up-python-data-modeling-and-model-evaluation-metrics-2499737/data/lasso_model.joblib')

load('C:/Users/sberry5/Documents/teaching/level-up-python-data-modeling-and-model-evaluation-metrics-2499737/data/lasso_model.joblib')

y_pred = lasso_model.predict(X_test_scaled)

# Needs to be scaled:
X_train_scaler = StandardScaler().fit(X_train)

X_train_scaled = X_train_scaler.transform(X_train)