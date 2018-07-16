from process import get_data_from_sql_file, process_query_result, categorical_columns,new_campaign_channel, numeric_columns, new_activity_new, old_activity_new, old_campaign_channel
import pickle
import xgboost as xgb
from sklearn.externals import joblib


file_name = 'process_test_dataset.sql'
score_data = get_data_from_sql_file(file_name)
pickle.dump(score_data, open('score_data.pkl', 'wb'))
# sql_result = pickle.load(open('sqlres.pkl', 'rb'))
numeric_columns.remove('churn')
numeric_columns.remove('invest')
X, Y = process_query_result(score_data, categorical_columns, numeric_columns, 'test')

for col in new_activity_new:
    X[X[col] == 1]['activity_new_'] = 1
    X.drop([col], inplace=True, axis=1)
for col in new_campaign_channel:
    X[X[col] == 1]['campain_channel_'] = 1
    X.drop([col], inplace=True, axis=1)
for col in old_activity_new:
    X[col] = X['activity_new_'].apply(lambda x: 0)
for col in old_campaign_channel:
    X[col] = X['activity_new_'].apply(lambda x: 0)

X['invest'] = 0
X['price_p3_power_increate_dec']=0
pickle.dump((X, Y), open('score_data_as_np.pkl', 'wb'))

Xtrain, Ytrain = pickle.load(open('xy.pkl', 'rb'))
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
xgb_model.fit(Xtrain, Ytrain)
joblib.dump('finalModel.pkl', xgb_model)
xgb_pred = xgb_model.predict(X[Xtrain.columns])
score_data['pred'] = xgb_pred
score_data[['id', 'pred']].to_csv('output.csv', index=False, header=False)
