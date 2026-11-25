import numpy as np
import pandas as pd
from .config import load_config

FEATURES = ['tenure_months','age','annual_income','credit_limit','credit_score','annual_spend','txn_count','avg_txn_value','online_spend_share','travel_spend','dining_spend','intl_spend','cash_advance_amt','avg_revolving_balance','utilization_ratio','payment_ratio','late_payment_count','apr','annual_fee','rewards_points_earned','rewards_redeemed_ratio','customer_service_contacts','digital_engagement_score']

def generate_customers(n=None, seed=None, config=None):
    cfg=config or load_config(); n=n or cfg['n_customers']; seed=cfg['seed'] if seed is None else seed
    rng=np.random.default_rng(seed); arche=rng.choice(6,n,p=[.12,.24,.22,.20,.10,.12])
    prem,trans,rev,low,risk,dorm=[arche==i for i in range(6)]
    age=np.clip(rng.normal(np.select([prem,trans,rev,low,risk,dorm],[44,39,42,35,38,52]),11),21,79)
    income=np.exp(rng.normal(np.select([prem,trans,rev,low,risk,dorm],[11.35,10.95,10.55,10.1,10.25,10.5]),.48))
    tenure=np.clip(rng.gamma(3,18,n),1,240)
    spend=np.exp(rng.normal(np.select([prem,trans,rev,low,risk,dorm],[10.55,9.95,9.5,8.9,9.2,8.6]),.75))*(income/60000)**.25
    spend[dorm]*=.15; spend[low]*=.55
    limit=np.clip(income*np.exp(rng.normal(-.15,.45,n)),1000,None)
    revolving=np.clip(limit*rng.beta(np.select([rev,risk,dorm],[2.7,2.2,.8],default=.6),np.select([rev,risk,dorm],[2.2,1.5,5],default=4),n),0,None)
    score=np.clip(rng.normal(np.select([risk,rev,prem],[625,690,750],default=710),np.select([risk],[65],default=48)),300,850)
    txn=np.maximum(0,rng.poisson(np.maximum(spend/95,2))); avg=spend/np.maximum(txn,1)
    online=np.clip(rng.beta(2,2,n)+low*.15,0,1)
    travel=spend*np.clip(rng.beta(np.where(prem,3,1.2),np.where(prem,2,5),n),0,1)
    dining=spend*np.clip(rng.beta(1.5,6,n),0,1); intl=travel*np.clip(rng.beta(1,8,n),0,1)
    late=np.clip(rng.poisson(np.where(risk,1.5,.18),n),0,12); cash=np.where(risk|rev,rng.gamma(2,350,n),rng.gamma(1,80,n))
    util=np.clip(revolving/np.maximum(limit,1),0,1.5); pay=np.clip(rng.beta(np.where(rev,2,5),np.where(rev,4,1.5),n),0,1)
    apr=np.clip(rng.normal(np.where(risk, .245,.185),.025,n),.10,.32); fee=np.where(prem|dorm,195,np.where(trans,95,0))
    points=np.maximum(0,spend*rng.normal(1.1,.12,n)); redeem=np.clip(rng.beta(3,2,n),0,1)
    contacts=np.clip(rng.poisson(np.where(risk|dorm,2,.5),n),0,20); digital=np.clip(rng.beta(np.where(low|dorm,1.5,3),np.where(low|dorm,3,1.8),n),0,1)
    df=pd.DataFrame({'customer_id':np.arange(1,n+1),'tenure_months':tenure,'age':age,'annual_income':income,'credit_limit':limit,'credit_score':score,'annual_spend':spend,'txn_count':txn,'avg_txn_value':avg,'online_spend_share':online,'travel_spend':travel,'dining_spend':dining,'intl_spend':intl,'cash_advance_amt':cash,'avg_revolving_balance':revolving,'utilization_ratio':util,'payment_ratio':pay,'late_payment_count':late,'apr':apr,'annual_fee':fee,'rewards_points_earned':points,'rewards_redeemed_ratio':redeem,'customer_service_contacts':contacts,'digital_engagement_score':digital})
    # Feature-derived financial components with independent multiplicative noise.
    noise=lambda sd: rng.lognormal(-sd*sd/2,sd,n)
    pd_=(.004+np.maximum(680-score,0)*.00016+late*.012+util*.018).clip(0,.65)
    df['interest_income']=revolving*apr*(1-pay*.45)*noise(.08)
    df['interchange_income']=spend*cfg['pnl']['interchange_rate']*noise(.06)
    df['fee_income']=(fee+late*29+cash*.03)*noise(.08)
    df['rewards_cost']=points*.0008*noise(.08)
    df['credit_loss']=revolving*pd_*.85*noise(.18)
    df['cost_of_funds']=revolving*.035*noise(.05)
    df['servicing_cost']=(45+contacts*12)*noise(.08)
    df['acquisition_cost_amortized']=45*noise(.05)
    caps=cfg['censoring_caps']
    for col,cap in caps.items():
        df[col+'_censored']=df[col]>=cap; df[col]=df[col].clip(upper=cap)
    # 1.5% missingness in modeling features, excluding values needed for generated P&L.
    for col in FEATURES:
        mask=rng.random(n)<.015; df.loc[mask,col]=np.nan
    return df

def save_data(n=None,seed=None):
    from .config import ROOT
    df=generate_customers(n,seed); (ROOT/'data/raw').mkdir(parents=True,exist_ok=True); df.to_csv(ROOT/'data/raw/customers.csv',index=False); df.head(1000).to_csv(ROOT/'data/sample/customers_sample.csv',index=False); return df
