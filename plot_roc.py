import sys
import xgboost as xgb
import matplotlib.pyplot as plt
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA


data_file_name = 'xy.pkl'
X, Y = pickle.load(open(data_file_name, 'rb'))
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33)
fpr = dict()
tpr = dict()
roc_auc = dict()


# xgboost model
parameters = {
    'learning_rate': 0.01,
    'max_depth': 1000,
    'min_child_weight': 3,
    'missing': -999,
    'n_estimators': 1000,
    'subsample': 0.8,
    'seed': 1337
}
xgb_model = xgb.XGBClassifier()
xgb_model.set_params(**parameters)
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict_proba(X_test)
fpr['xgb'], tpr['xgb'], _ = roc_curve(y_test, xgb_pred[:, 1])
roc_auc['xgb'] = auc(fpr['xgb'], tpr['xgb'])


# logistic regression model
pca = PCA()
logistic = LogisticRegression()
params = dict(
    pca__n_components=40,
    logistic__penalty='l1',
    logistic__C=1,
    logistic__class_weight='balanced'
)
logisticRegressionPipe = Pipeline(steps=[('pca', pca), ('logistic', logistic)])
logisticRegressionPipe.fit(X_train, y_train)
logistic_pred = logisticRegressionPipe.predict_proba(X_test)
fpr['logistic'], tpr['logistic'], _ = roc_curve(y_test, logistic_pred[:, 1])
roc_auc['logistic'] = auc(fpr['logistic'], tpr['logistic'])


# logistic regression model
params = dict(
    bootstrap=False,
    max_depth=100,
    max_leaf_nodes=3,
    min_samples_leaf=2,
    min_samples_split=10,
    min_weight_fraction_leaf=0.2,
    n_estimators=10
)
rf = RandomForestClassifier(**params)
rf.fit(X_train, y_train)
rf_pred = rf.predict_proba(X_test)
fpr['rf'], tpr['rf'], _ = roc_curve(y_test, rf_pred[:, 1])
roc_auc['rf'] = auc(fpr['rf'], tpr['rf'])


plt.figure()
plt.plot(fpr['xgb'], tpr['xgb'], color='darkorange', linestyle=':',
         lw=2, label='Xgboost (area = %0.2f)' % roc_auc['xgb'])
plt.plot(fpr['logistic'], tpr['logistic'], color='deeppink', linestyle=':',
         lw=2, label='Logistic Regression (area = %0.2f)' % roc_auc['logistic'])
plt.plot(fpr['rf'], tpr['rf'], color='cornflowerblue', linestyle=':',
         lw=2, label='Random Forest (area = %0.2f)' % roc_auc['rf'])
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic Plot')
plt.legend(loc="lower right")
plt.show()


