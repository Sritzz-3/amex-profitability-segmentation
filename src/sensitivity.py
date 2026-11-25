import numpy as np,pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score,normalized_mutual_info_score,silhouette_score
from scipy.stats import spearmanr
from .pnl import add_net_contribution

def stability(Z,labels,k,seed=42):
    rows=[]; rng=np.random.default_rng(seed)
    for s in range(10):
        km=KMeans(n_clusters=k,n_init=10,random_state=seed+100+s).fit(Z); rows.append({'seed':seed+100+s,'ari':adjusted_rand_score(labels,km.labels_)})
    seed_df=pd.DataFrame(rows)
    boot=[]
    for b in range(20):
        ix=rng.choice(len(Z),max(2,int(.8*len(Z))),replace=True); km=KMeans(n_clusters=k,n_init=5,random_state=seed+b+500).fit(Z[ix]); pred=km.predict(Z); boot.append({'bootstrap':b,'ari':adjusted_rand_score(labels,pred)})
    return seed_df,pd.DataFrame(boot)
def pnl_sensitivity(df,cfg,baseline,labels):
    rows=[]; base=baseline.is_top20_profitable.to_numpy(); basev=baseline.net_contribution.to_numpy()
    for param in ['funding_rate','point_value','lgd','interchange_rate']:
        for delta in [-.2,-.1,.1,.2]:
            cc={**cfg,'pnl':{**cfg['pnl'],param:cfg['pnl'][param]*(1+delta)}}
            # Keep baseline corrected inputs when applying the changed assumptions.
            scored=add_net_contribution(df,cc); vals=scored.net_contribution.to_numpy(); threshold=np.quantile(vals,.8); flag=vals>=threshold
            rows.append({'assumption':param,'change_pct':delta*100,'jaccard_top20':np.logical_and(base,flag).sum()/np.logical_or(base,flag).sum(),'spearman_net_contribution':spearmanr(basev,vals).statistic})
    # Compare the two censoring policies using identical P&L assumptions.
    naive = df.copy()
    for col in ['annual_spend','avg_revolving_balance','credit_limit']:
        if col+'_corrected' in naive: naive[col+'_corrected']=naive[col]
    vals=add_net_contribution(naive,cfg).net_contribution.to_numpy(); flag=vals>=np.quantile(vals,.8)
    rows.append({'assumption':'censoring_method','change_pct':0,'jaccard_top20':np.logical_and(base,flag).sum()/np.logical_or(base,flag).sum(),'spearman_net_contribution':spearmanr(basev,vals).statistic})
    return pd.DataFrame(rows)
