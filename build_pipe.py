from process import process_query_result, get_data_from_sql_file, categorical_columns, numeric_columns
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_curve, auc, roc_auc_score
import xgboost as xgb
from sklearn.externals import joblib
from sklearn.pipeline import Pipeline
import pickle


file_name = 'process_train_dataset.sql'
sql_result = get_data_from_sql_file(file_name)
pickle.dump(sql_result, open('sqlres.pkl', 'wb'))
# sql_result = pickle.load(open('sqlres.pkl', 'rb'))

X, Y = process_query_result(sql_result, categorical_columns, numeric_columns)
pickle.dump((X, Y), open('xy.pkl', 'wb'))
# X, Y = pickle.load(open('xy.pkl', 'rb'))


# xgboost classifier
xgb_model = xgb.XGBClassifier()
parameters = {
    'nthread': [-1],
    # 'objective': ['binary:logistic'],
    # 'max_delta_step': [0],
    'learning_rate': [0.05, 0.1, 0.5, 1],  # so called `eta` value
    'max_depth': [5, 10, 20],
    'min_child_weight': [3, 5, 10],
    'silent': [1],
    'subsample': [0.5, 0.8, 0.3],
    'reg_lambda': [0.5, 1, 2, 5],  # l1
    'reg_alpha': [0.1, 0.2, 0.5, 0.8],  # l2
    'colsample_bytree': [0.5, 0.8],
    'n_estimators': [50, 100, 1000],  # number of trees, change it to 1000 for better results
    'missing': [-999]
}
clf = GridSearchCV(xgb_model, parameters,  scoring='roc_auc', verbose=0, refit=True)
best_xgb = clf.fit(X, Y)
importance = pd.DataFrame(best_xgb.best_estimator_.get_booster().get_fscore().items(), columns=['feature', 'importance'])\
    .sort_values('importance', ascending=False)
joblib.dump(best_xgb, 'xgboost.pkl')
best_parameters, score, _ = max(clf.grid_scores_, key=lambda x: x[1])
print('Raw AUC score:', score)
for param_name in sorted(best_parameters.keys()):
    print("%s: %r" % (param_name, best_parameters[param_name]))


# logistic regression
pca = PCA()
logistic = LogisticRegression()
pipe = Pipeline(steps=[('pca', pca), ('logistic', logistic)])
penalty = ['l2', 'l1']
C = [0.001, 0.01, 1, 10]
class_weight = ['balanced']
n_components = [40]
params = dict(
    pca__n_components=n_components,
    logistic__penalty=penalty,
    logistic__C=C,
    logistic__class_weight=class_weight
)
clf = GridSearchCV(pipe,  params, cv=StratifiedKFold(n_splits=5), verbose=0, scoring='roc_auc')
best_logistic = clf.fit(X, Y)
joblib.dump(best_logistic, 'logisticRegression.pkl')
print('Best n component:', best_logistic.best_estimator_.get_params()['pca__n_components'])
print('Best Penalty:', best_logistic.best_estimator_.get_params()['logistic__penalty'])
print('Best C:', best_logistic.best_estimator_.get_params()['logistic__C'])
print('Best class weight:', best_logistic.best_estimator_.get_params()['logistic__class_weight'])
print(roc_auc_score(Y, best_logistic.predict_proba(X)[:, 1]))


# random forest
rf = RandomForestClassifier()
params = {
    'bootstrap': [True, False],
    'criterion': ['mse', 'gini'],
    "n_estimators": np.arange(2, 300, 2),
    "max_depth": np.arange(1, 28, 1),
    "min_samples_split": np.arange(1, 150, 1),
    "min_samples_leaf": np.arange(1, 60, 1),
    "max_leaf_nodes": np.arange(2, 60, 1),
    "min_weight_fraction_leaf": np.arange(0.1, 0.4, 0.1)
}
clf = GridSearchCV(rf,  params, cv=StratifiedKFold(n_splits=5), verbose=0, scoring='roc_auc')
best_rf = clf.fit(X, Y)
joblib.dump(best_rf, 'rf.pkl')
print(roc_auc_score(Y, best_rf.predict_proba(X)[:, 1]))
