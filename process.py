import mysql.connector
import pandas as pd
from sklearn.preprocessing import scale


def get_data_from_sql_file(file_name):
    with open(file_name, 'r') as sql_file:
        query = sql_file.read()
    cnx = mysql.connector.connect(
        user='f',
        password='Last270731',
        host='127.0.0.1',
        database='bcg'
    )
    sql_result = pd.read_sql(query, cnx)
    # columns = sql_result.columns[1:] # the first column is id not useful for model
    return sql_result


def process_query_result(sql_result, categorical_columns, numeric_columns):
    categorical_data = pd.get_dummies(sql_result[categorical_columns])
    categorical_columns = list(categorical_data.columns)
    numeric_data = sql_result[numeric_columns]
    numeric_data_scaled = scale(numeric_data)
    processed_data = pd.concat([categorical_data, numeric_data_scaled], axis=1)
    columns = list(categorical_columns + numeric_columns)
    X = processed_data.fillna(.0)[columns[:-1]]
    Y = sql_result['churn'].astype('int')
    return X, Y



if __name__ == '__main__':
    file_name = 'process_train_dataset.sql'
    categorical_columns = ['activity_new', 'channel_sales', 'contract_status', 'is_mod', 'has_gas', 'campain_channel']
    numeric_columns = [
        'cons', 'gas_cons', 'cons_last_month',  'days_to_expire', 'contract_length',
        'forecast_price_energy_p1', 'forecast_price_energy_p2', 'forecast_price_pow_p1', 'forecast_bill_montly',
        'next_month_forecast', 'forecast_con_monthly', 'discount','current_paid_cons', 'num_prod', 'power_net_margin',
        'total_net_margin', 'num_years_antig', 'power_subscribed'
    ]
    sql_result = get_data_from_sql_file(file_name)
    X, Y = process_query_result(sql_result, categorical_columns, numeric_columns)





# cursor = cnx.cursor()
# cursor.execute("SHOW columns FROM train")
# columns = [col[0] for col in cursor.fetchall()]
# print(columns)
# query = 'select * from test limit 100;'
# cursor.execute(query)
# for cur in cursor:
#     print(cur)