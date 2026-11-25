import numpy as np
import pandas as pd

def add_net_contribution(df, cfg, correction_method='pareto'):
    """Return a copy with vectorized illustrative annual customer P&L components and net contribution."""
    p=cfg['pnl']; x=df.copy()
    spend=x['annual_spend_corrected'] if 'annual_spend_corrected' in x else x['annual_spend']
    bal=x['avg_revolving_balance_corrected'] if 'avg_revolving_balance_corrected' in x else x['avg_revolving_balance']
    score=x.credit_score.fillna(x.credit_score.median()); late=x.late_payment_count.fillna(0); util=x.utilization_ratio.fillna(0)
    pd_=np.clip(p['pd_base']+np.maximum(p['pd_score_reference']-score,0)*p['pd_score_penalty_per_point']+late*p['pd_late_payment_rate']+util*p['pd_utilization_rate'],0,.65)
    x['interchange_income']=spend*p['interchange_rate']
    x['interest_income']=bal*x.apr.fillna(x.apr.median())*(1-x.payment_ratio.fillna(x.payment_ratio.median())*p['payment_ratio_adjustment'])
    x['fee_income']=x.annual_fee.fillna(0)+late*p['late_fee']+x.cash_advance_amt.fillna(0)*p['cash_advance_rate']
    x['rewards_cost']=x.rewards_points_earned.fillna(0)*p['point_value']*(1+p['redemption_uplift']*x.rewards_redeemed_ratio.fillna(0))
    x['credit_loss']=bal*pd_*p['lgd']; x['cost_of_funds']=bal*p['funding_rate']
    x['servicing_cost']=p['base_servicing_cost']+x.customer_service_contacts.fillna(0)*p['cost_per_contact']
    x['acquisition_cost_amortized']=p['cac']/p['expected_lifetime_years']
    x['net_contribution']=x.interest_income+x.interchange_income+x.fee_income-x.rewards_cost-x.credit_loss-x.cost_of_funds-x.servicing_cost-x.acquisition_cost_amortized
    return x
