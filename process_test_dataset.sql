SELECT
    a.id,
	  a.activity_new,
    a.channel_sales,
    a.cons_12m/12 cons,
    a.cons_gas_12m/12 gas_cons,
    a.cons_last_month cons_last_month,
    case when a.date_end<'2016-03-31' then 'expired' else 'not expired' end contract_status,
    timestampdiff(day, a.date_end, '2016-03-31') days_to_expire,
    timestampdiff(month, case when a.date_renewal='' then '2016-03-01' else a.date_renewal end, '2016-03-01') month_since_renewal,
    timestampdiff(month, a.date_activ, a.date_end) contract_length,
    case when timestampdiff(month, a.date_activ, a.date_first_activ)<0 then'yes' else 'no' end as return_cust,
    CASE
      when timestampdiff(month, a.date_modif_prod, a.date_first_activ)<0 then 'yes'
      when timestampdiff(month, a.date_modif_prod, a.date_activ)<0 then 'yes'
    ELSE 'no' END  is_mod,
    case
    when timestampdiff(month, (a.date_activ), '2016-03-01') > coalesce( timestampdiff(month, (a.date_first_activ), '2016-03-01'), 0)
    then timestampdiff(month, (a.date_activ), '2016-03-01')
    else coalesce( timestampdiff(month, (a.date_first_activ), '2016-03-01'), 0)
    end tenure,
	  a.forecast_price_energy_p1,
    a.forecast_price_energy_p2,
    a.forecast_price_pow_p1,
    a.forecast_bill_12m/12 forecast_bill_montly,
    cast((case a.forecast_cons when '' then '0' else a.forecast_cons end)as decimal(10,6)) next_month_forecast,
	  a.forecast_cons_12m/12 forecast_con_monthly,
    a.forecast_discount_energy fore_cast_discount,
    a.has_gas,
    a.imp_cons current_paid_cons,
	  a.margin_gross_pow_ele ele_gross_margin,
	  a.margin_net_pow_ele power_net_margin,
    a.nb_prod_act num_prod,
    a.net_margin total_net_margin,
    a.num_years_antig,
    a.origin_up campain_channel,
    a.pow_max power_subscribed,
    b.price_seg,
    b.p1_power,
    b.p2_power,
    b.p3_power,
    b.p1_energy,
    b.p2_energy,
    b.p3_energy,
    b.p1_power_variance,
    b.p2_power_variance,
    b.p3_power_variance,
    b.p1_energy_variance,
    b.p2_energy_variance,
    b.p3_energy_variance,
    p.price_p1_energy_increate_half_year,
    p.price_p2_energy_increate_half_year,
    p.price_p3_energy_increate_half_year,
    p.price_p1_power_increate_half_year,
    p.price_p2_power_increate_half_year,
    p.price_p3_power_increate_half_year,
    p.price_p1_energy_increate_last_q,
    p.price_p2_energy_increate_last_q,
    p.price_p3_energy_increate_last_q,
    p.price_p1_power_increate_last_q,
    p.price_p2_power_increate_last_q,
    p.price_p3_power_increate_last_q,
    p.price_p1_energy_increate_dec,
    p.price_p2_energy_increate_dec,
    p.price_p3_energy_increate_dec,
    p.price_p1_power_increate_dec,
    p.price_p2_power_increate_dec,
    p.price_p3_power_increate_dec
FROM bcg.test a
left join (
select
  id,
  if (sum(price_p1_var) = 0 , 'no', if(sum(price_p2_var)=0, 'p1', if(sum(price_p3_var) =0,'p1p2', 'p1p2p3'))) price_seg,
  avg(price_p1_fix) p1_power,
  avg(price_p2_fix) p2_power,
  avg(price_p3_fix) p3_power,
  avg(price_p1_var) p1_energy,
  avg(price_p2_var) p2_energy,
  avg(price_p3_var) p3_energy,
  variance(price_p1_fix) p1_power_variance,
  variance(price_p2_fix) p2_power_variance,
  variance(price_p3_fix) p3_power_variance,
  variance(price_p1_var) p1_energy_variance,
  variance(price_p2_var) p2_energy_variance,
  variance(price_p3_var) p3_energy_variance
 from bcg.test_hist
 group by id
)b
  on a.id = b.id
left join (
 select
   h1.id,
   coalesce((h2.price_p1_energy-h1.price_p1_energy)/h1.price_p1_energy, 0) price_p1_energy_increate_half_year,
   coalesce((h2.price_p2_energy-h1.price_p2_energy)/h1.price_p2_energy, 0) price_p2_energy_increate_half_year,
   coalesce((h2.price_p3_energy-h1.price_p3_energy)/h1.price_p3_energy, 0) price_p3_energy_increate_half_year,
   coalesce((h2.price_p1_power-h1.price_p1_power)/h1.price_p1_power, 0) price_p1_power_increate_half_year,
   coalesce((h2.price_p2_power-h1.price_p2_power)/h1.price_p2_power, 0) price_p2_power_increate_half_year,
   coalesce((h2.price_p3_power-h1.price_p3_power)/h1.price_p3_power, 0) price_p3_power_increate_half_year,
   coalesce((q4.price_p1_energy-q3.price_p1_energy)/q3.price_p1_energy, 0) price_p1_energy_increate_last_q,
   coalesce((q4.price_p2_energy-q3.price_p2_energy)/q3.price_p2_energy, 0) price_p2_energy_increate_last_q,
   coalesce((q4.price_p3_energy-q3.price_p3_energy)/q3.price_p3_energy, 0) price_p3_energy_increate_last_q,
   coalesce((q4.price_p1_power-q3.price_p1_power)/q3.price_p1_power, 0) price_p1_power_increate_last_q,
   coalesce((q4.price_p2_power-q3.price_p2_power)/q3.price_p2_power, 0) price_p2_power_increate_last_q,
   coalesce((q4.price_p3_power-q3.price_p3_power)/q3.price_p3_power, 0) price_p3_power_increate_last_q,
   coalesce((de.price_p1_energy-nov.price_p1_energy)/nov.price_p1_energy, 0) price_p1_energy_increate_dec,
   coalesce((de.price_p2_energy-nov.price_p2_energy)/nov.price_p2_energy, 0) price_p2_energy_increate_dec,
   coalesce((de.price_p3_energy-nov.price_p3_energy)/nov.price_p3_energy, 0) price_p3_energy_increate_dec,
   coalesce((de.price_p1_power-nov.price_p1_power)/nov.price_p1_power, 0) price_p1_power_increate_dec,
   coalesce((de.price_p2_power-nov.price_p2_power)/nov.price_p2_power, 0) price_p2_power_increate_dec,
   coalesce((de.price_p3_power-nov.price_p3_power)/nov.price_p3_power, 0) price_p3_power_increate_dec
 from(
    select id, avg(price_p1_var) price_p1_energy, avg(price_p1_fix) price_p1_power, avg(price_p2_var) price_p2_energy, avg(price_p2_fix) price_p2_power, avg(price_p3_var) price_p3_energy, avg(price_p3_fix) price_p3_power from  bcg.test_hist where price_date <= '2015-06-01'group by id
  ) h1
  left join (
    select id, avg(price_p1_var) price_p1_energy, avg(price_p1_fix) price_p1_power, avg(price_p2_var) price_p2_energy, avg(price_p2_fix) price_p2_power, avg(price_p3_var) price_p3_energy, avg(price_p3_fix) price_p3_power from  bcg.test_hist where price_date > '2015-06-01' group by id
  )h2 on h1.id =h2.id
  left join (
    select id, avg(price_p1_var) price_p1_energy, avg(price_p1_fix) price_p1_power, avg(price_p2_var) price_p2_energy, avg(price_p2_fix) price_p2_power, avg(price_p3_var) price_p3_energy, avg(price_p3_fix) price_p3_power from  bcg.test_hist where price_date <= '2015-09-01' and price_date > '2015-06-01' group by id
  )q3 on h2.id=q3.id
  left join (
    select id, avg(price_p1_var) price_p1_energy, avg(price_p1_fix) price_p1_power, avg(price_p2_var) price_p2_energy, avg(price_p2_fix) price_p2_power, avg(price_p3_var) price_p3_energy, avg(price_p3_fix) price_p3_power from  bcg.test_hist where price_date > '2015-09-01' group by id
  )q4 on q3.id=q4.id
  left join (
    select id, avg(price_p1_var) price_p1_energy, avg(price_p1_fix) price_p1_power, avg(price_p2_var) price_p2_energy, avg(price_p2_fix) price_p2_power, avg(price_p3_var) price_p3_energy, avg(price_p3_fix) price_p3_power from  bcg.test_hist where price_date = '2015-11-01' group by id
  )nov on q4.id=nov.id
  left join (
    select id, avg(price_p1_var) price_p1_energy, avg(price_p1_fix) price_p1_power, avg(price_p2_var) price_p2_energy, avg(price_p2_fix) price_p2_power, avg(price_p3_var) price_p3_energy, avg(price_p3_fix) price_p3_power from  bcg.test_hist where price_date = '2015-12-01' group by id
  )de on nov.id=de.id
)p
  on a.id = p.id
;