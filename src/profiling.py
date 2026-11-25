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
    if r.mean_credit_score<660 and r.mean_utilization_ratio>.45:return 'Loss-Making High-Risk'
    if r.mean_annual_spend>r.mean_annual_spend*0+18000 and r.mean_net_contribution>0:return 'Premium Spenders'
    if r.mean_avg_revolving_balance>5000 and r.mean_net_contribution>0:return 'Profitable Revolvers'
    if r.mean_net_contribution<0:return 'Low-Value / Loss-Making'
    return 'Steady Transactors'
