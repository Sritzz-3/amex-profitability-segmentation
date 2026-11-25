import numpy as np, pandas as pd
from .data_generator import FEATURES
KEY=['annual_spend','avg_revolving_balance','annual_income','credit_score','utilization_ratio','payment_ratio']
def profile(df,labels):
    x=df.copy(); x['cluster']=labels; q=x.net_contribution.quantile(.8); x['is_top20_profitable']=x.net_contribution>=q
    n=x.net_contribution.sum(); rows=[]
    for c,g in x.groupby('cluster'):
        row={'cluster':int(c),'size':len(g),'mean_net_contribution':g.net_contribution.mean(),'median_net_contribution':g.net_contribution.median(),'total_contribution':g.net_contribution.sum(),'share_portfolio_profit':g.net_contribution.sum()/n if n else 0,'top20_customers':int(g.is_top20_profitable.sum()),'top20_customer_share':g.is_top20_profitable.mean()}
        row.update({f'mean_{f}':g[f].mean() for f in KEY}); rows.append(row)
    prof=pd.DataFrame(rows); prof['cluster_name']=prof.apply(_name,axis=1)
    x=x.merge(prof[['cluster','cluster_name']],on='cluster',how='left')
    lift=prof[['cluster','cluster_name','top20_customers','top20_customer_share']].copy(); lift['share_flagged_customers']=lift.top20_customers/max(int(x.is_top20_profitable.sum()),1)
    return x,prof,lift,float(x.loc[x.is_top20_profitable,'net_contribution'].sum()/n if n else 0)
def _name(r):
    high_risk = r.mean_credit_score < 660 and r.mean_utilization_ratio > .45
    if r.mean_net_contribution < 0 and high_risk:
        return 'Loss-Making High-Risk'
    if r.mean_net_contribution < 0:
        return 'Loss-Making Customers'
    if high_risk:
        return 'High-Risk Revolvers'
    if r.mean_annual_spend >= 18000:
        return 'Premium Spenders'
    if r.mean_avg_revolving_balance >= 8000 or r.mean_utilization_ratio >= .4:
        return 'Profitable Revolvers'
    if r.mean_annual_spend < 2500 and r.mean_avg_revolving_balance >= 3000:
        return 'Dormant Balances'
    if r.mean_annual_spend < 8000:
        return 'Low-Engagement Transactors'
    return 'Steady Transactors'
