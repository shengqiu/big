import pandas as pd
import pickle
from sklearn.externals import joblib


def get_quantile(data, column):
    column_churn_record = dict()
    column_churn_record['column'] = column
    low = .0
    for high in [.25, .5, .75, 1.]:
        subset = data.loc[(data[column] >= data.quantile(low)[column]) & (data[column] < data.quantile(high)[column])]
        churn_rate = subset['churn'].astype('float').fillna(0).mean()
        print(subset[column].astype('float').fillna(0).mean())
        low = high
        column_churn_record[high] = churn_rate
    return column_churn_record


if __name__ == '__main__':
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
    best_xgb = joblib.load('xgboost.pkl')
    importance = pd.DataFrame(
        best_xgb.best_estimator_.get_booster().get_fscore().items(), columns=['feature', 'importance']
    ).sort_values('importance', ascending=False)
    importance = importance[importance['importance'] > 10]
    data = pickle.load(open('sqlres.pkl', 'rb'))
    column_churn_rec_for_dict = list()
    for col in numeric_columns:
        col_rec = get_quantile(data, col)
        try:
            col_rec['importance'] = float(importance[importance['feature'] == col]['importance'])
        except TypeError:
            pass
        column_churn_rec_for_dict.append(col_rec)
    col_df = pd.DataFrame.from_records(
        column_churn_rec_for_dict, index='column'
    ).sort_values('importance', ascending=False)
    categorical_importance = {}
    for col in categorical_columns:
        score = 0
        for (ind, (col_temp, imp)) in importance.iterrows():
            if col in col_temp:
                score += imp
        categorical_importance[col] = score
