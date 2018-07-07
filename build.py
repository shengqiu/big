from process import process_query_result, get_data_from_sql_file
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_curve, auc, roc_auc_score
import xgboost as xgb
from sklearn.externals import joblib
import pickle


#file_name = 'process_train_dataset.sql'
#sql_result = get_data_from_sql_file(file_name)
#pickle.dump(sql_result, open('sqlres.pkl', 'wb'))
sql_result = pickle.load(open('sqlres.pkl', 'rb'))
categorical_columns = [
    'activity_new',
    'channel_sales',
    'contract_status',
    'return_cust',
    'is_mod',
    'has_gas',
    'campain_channel',
    'price_seg'
]
numeric_columns = [
    'cons',
    'gas_cons',
    'cons_last_month',
    'days_to_expire',
    'month_since_renewal',
    'contract_length',
    'tenure',
    'forecast_price_energy_p1',
    'forecast_price_energy_p2',
    'forecast_price_pow_p1',
    'forecast_bill_montly',
    'next_month_forecast',
    'forecast_con_monthly',
    'fore_cast_discount',
    'current_paid_cons',
    'invest',
    'num_prod',
    'power_net_margin',
    'total_net_margin',
    'num_years_antig',
    'power_subscribed',
    'p1_power',
    'p2_power',
    'p3_power',
    'p1_energy',
    'p2_energy',
    'p3_energy',
    'p1_power_variance',
    'p2_power_variance',
    'p3_power_variance',
    'p1_energy_variance',
    'p2_energy_variance',
    'p3_energy_variance',
    'price_p1_energy_increate_half_year',
    'price_p2_energy_increate_half_year',
    'price_p3_energy_increate_half_year',
    'price_p1_power_increate_half_year',
    'price_p2_power_increate_half_year',
    'price_p3_power_increate_half_year',
    'price_p1_energy_increate_last_q',
    'price_p2_energy_increate_last_q',
    'price_p3_energy_increate_last_q',
    'price_p1_power_increate_last_q',
    'price_p2_power_increate_last_q',
    'price_p3_power_increate_last_q',
    'price_p1_energy_increate_dec',
    'price_p2_energy_increate_dec',
    'price_p3_energy_increate_dec',
    'price_p1_power_increate_dec',
    'price_p2_power_increate_dec',
    'price_p3_power_increate_dec',
    'churn'
]
X, Y = process_query_result(sql_result, categorical_columns, numeric_columns)
pickle.dump((X, Y), open('xy.pkl', 'wb'))


xgb_model = xgb.XGBClassifier()
parameters = {
    # 'booster': ['gblinear', 'gbtree'],
    'objective': ['binary:logistic'],
    'learning_rate': [0.05],  # so called `eta` value
    'max_depth': [5,10],
    'min_child_weight': [3, 5, 10],
    'silent': [1],
    'subsample': [0.8, 0.5],
    'colsample_bytree': [0.7],
    'n_estimators': [5, 10, 100], #number of trees, change it to 1000 for better results
    'missing': [-999],
    'seed': [1337]
}
clf = GridSearchCV(xgb_model, parameters, n_jobs=1, scoring='roc_auc', verbose=0, refit=True)
best_xgb = clf.fit(X, Y)
joblib.dump(best_xgb, 'xgboost.pkl')
best_parameters, score, _ = max(clf.grid_scores_, key=lambda x: x[1])
print('Raw AUC score:', score)
for param_name in sorted(best_parameters.keys()):
    print("%s: %r" % (param_name, best_parameters[param_name]))
'''
('Raw AUC score:', 0.7000975013061959)
colsample_bytree: 0.7
learning_rate: 0.05
max_depth: 10
min_child_weight: 3
missing: -999
n_estimators: 100
objective: 'binary:logistic'
seed: 1337
silent: 1
subsample: 0.8
'''


logistic = linear_model.LogisticRegression()
penalty = ['l1', 'l2']
C = np.logspace(0, 4, 10)
class_weight = ['balanced', None]
params = dict(penalty=penalty, C=C, class_weight=class_weight)
clf = GridSearchCV(logistic,  params, cv=StratifiedKFold(n_splits=5), verbose=0, scoring='roc_auc')
best_logistic = clf.fit(X, Y)
joblib.dump(best_logistic, 'logisticRegression.pkl')
print('Best Penalty:', best_logistic.best_estimator_.get_params()['penalty'])
print('Best C:', best_logistic.best_estimator_.get_params()['C'])
print('Best class weight:', best_logistic.best_estimator_.get_params()['class_weight'])
print('Best solver:', best_logistic.best_estimator_.get_params()['solver'])
print(roc_auc_score(Y, best_logistic.predict_proba(X)[:, 1]))
'''
'Best Penalty:', 'l2')
('Best C:', 166.81005372000593)
('Best class weight:', 'balanced')
('Best solver:', 'liblinear')
0.6462144527243879
'''






# logreg = linear_model.LogisticRegression()
# cv = StratifiedKFold(n_splits=5, shuffle=True)
# i = 1
# tprs = []
# aucs = []
# mean_fpr = np.linspace(0, 1, 100)
# for train, test in cv.split(X, Y):
#     prediction = logreg.fit(X.iloc[train], Y.iloc[train]).predict_proba(X.iloc[test])
#     fpr, tpr, t = roc_curve(Y[test], prediction[:, 1])
#     tprs.append(interp(mean_fpr, fpr, tpr))
#     roc_auc = auc(fpr, tpr)
#     aucs.append(roc_auc)
#     print('ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
#     i = i + 1

