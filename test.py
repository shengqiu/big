import mysql.connector
import pandas as pd
import numpy as np
from scipy import interp
from sklearn import linear_model
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc

with open('process_train_dataset.sql', 'r') as sql_file:
    query = sql_file.read()
cnx = mysql.connector.connect(
    user='f',
    password='Last270731',
    host='127.0.0.1',
    database='bcg'
)
sql_result = pd.read_sql(query, cnx)
columns = sql_result.columns[1:] # the first column is id not useful for model
categorical_columns = ['activity_new', 'channel_sales', 'contract_status', 'is_mod', 'has_gas', 'campain_channel']
numeric_columns = [
    'cons', 'gas_cons', 'cons_last_month',  'days_to_expire', 'contract_length',
    'forecast_price_energy_p1', 'forecast_price_energy_p2', 'forecast_price_pow_p1', 'forecast_bill_montly',
    'next_month_forecast', 'forecast_con_monthly', 'discount','current_paid_cons', 'num_prod', 'power_net_margin',
    'total_net_margin', 'num_years_antig', 'power_subscribed'
]
categorical_data = pd.get_dummies(sql_result[categorical_columns])
categorical_columns = list(categorical_data.columns)
numeric_data = sql_result[numeric_columns]
processed_data = pd.concat([categorical_data, numeric_data], axis=1)
columns = list(categorical_columns + numeric_columns)
X = processed_data.fillna(.0)[columns[:-1]]
Y = sql_result['churn'].astype('int')
h = .02  # step size in the mesh
logreg = linear_model.LogisticRegression()
cv = StratifiedKFold(n_splits=5, shuffle=True)
i = 1
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)
for train, test in cv.split(X, Y):
    prediction = logreg.fit(X.iloc[train], Y.iloc[train]).predict_proba(X.iloc[test])
    fpr, tpr, t = roc_curve(Y[test], prediction[:, 1])
    tprs.append(interp(mean_fpr, fpr, tpr))
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    print('ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
    # plt.plot(fpr, tpr, lw=2, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
    i = i+1



# cursor = cnx.cursor()
# cursor.execute("SHOW columns FROM train")
# columns = [col[0] for col in cursor.fetchall()]
# print(columns)
# query = 'select * from test limit 100;'
# cursor.execute(query)
# for cur in cursor:
#     print(cur)