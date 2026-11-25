import argparse,logging,joblib
import numpy as np,pandas as pd
from .config import load_config,ROOT
from .data_generator import generate_customers,save_data,FEATURES
from .censoring import correct_censoring
from .pnl import add_net_contribution
from .modeling import evaluate_k,fit_model
from .profiling import profile
from .sensitivity import stability,pnl_sensitivity
from .visualization import figures
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score,normalized_mutual_info_score,silhouette_score
logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
def run(n_customers=None,seed=None):
    cfg=load_config(); seed=cfg['seed'] if seed is None else seed; n=n_customers or cfg['n_customers']
    df=generate_customers(n,seed,cfg); (ROOT/'data/raw').mkdir(parents=True,exist_ok=True); df.to_csv(ROOT/'data/raw/customers.csv',index=False); df.head(1000).to_csv(ROOT/'data/sample/customers_sample.csv',index=False)
    corr,stats=correct_censoring(df,cfg['censoring_caps']); scored=add_net_contribution(corr,cfg)
    model_features=corr[FEATURES].copy()
    for col in cfg['censoring_caps']:
        model_features[col]=corr[col+'_corrected']
    prep,pca,Z=fit_model(model_features,cfg,seed); ev=evaluate_k(Z,cfg['k_min'],cfg['k_max'],seed)
    choices=ev[ev.k.between(cfg['business_k_min'],cfg['business_k_max'])]; k=int(choices.sort_values('silhouette',ascending=False).iloc[0].k)
    km=KMeans(n_clusters=k,n_init=20,random_state=seed).fit(Z); labels=km.labels_
    scored,prof,lift,top_profit=profile(scored,labels)
    tabs=ROOT/'outputs/tables'; tabs.mkdir(parents=True,exist_ok=True); figs=ROOT/'outputs/figures'; figs.mkdir(parents=True,exist_ok=True); models=ROOT/'outputs/models'; models.mkdir(parents=True,exist_ok=True)
    ev.to_csv(tabs/'k_diagnostics.csv',index=False); prof.to_csv(tabs/'cluster_profiles.csv',index=False); lift.to_csv(tabs/'top20_cluster_lift.csv',index=False)
    scored[['customer_id','cluster','cluster_name','net_contribution','is_top20_profitable']].to_csv(tabs/'customer_scored.csv',index=False)
    loadings=pd.DataFrame(pca.components_.T,index=FEATURES,columns=[f'PC{i+1}' for i in range(pca.n_components_)]); loadings.to_csv(tabs/'pca_loadings.csv')
    pd.DataFrame([{'variable':v,**d} for v,d in stats.items()]).to_csv(tabs/'censoring_correction.csv',index=False)
    seed_df,boot_df=stability(Z,labels,k,seed); seed_df.to_csv(tabs/'seed_stability.csv',index=False); boot_df.to_csv(tabs/'bootstrap_stability.csv',index=False)
    # Bootstrap per-cluster Jaccard match to baseline cluster member sets.
    # With baseline labels on all rows, calculate the best matched cluster overlap per resample.
    rng=np.random.default_rng(seed+9); jr=[]
    for b in range(20):
        ix=rng.choice(len(Z),max(2,int(.8*len(Z))),replace=True); m=KMeans(n_clusters=k,n_init=5,random_state=seed+b+900).fit(Z[ix]); pred=m.predict(Z)
        for c in range(k):
            candidates=[np.logical_and(labels==c,pred==d).sum()/max(np.logical_or(labels==c,pred==d).sum(),1) for d in range(k)]
            jr.append({'bootstrap':b,'cluster':c,'jaccard':max(candidates)})
    pd.DataFrame(jr).to_csv(tabs/'bootstrap_cluster_jaccard.csv',index=False)
    # Hyperparameter sensitivity: k neighbors and requested PCA variance targets.
    hyper=[]
    for var in [.80,.85,.90,.95]:
        _,p2,z2=fit_model(model_features,cfg,seed,var)
        for kk in sorted(set(max(2,min(12,k+d)) for d in [-2,-1,0,1,2])):
            mm=KMeans(n_clusters=kk,n_init=10,random_state=seed).fit(z2); hyper.append({'variance_target':var,'k':kk,'ari_vs_baseline':adjusted_rand_score(labels,mm.labels_),'nmi_vs_baseline':normalized_mutual_info_score(labels,mm.labels_),'silhouette':silhouette_score(z2[np.random.default_rng(seed).choice(len(z2),min(5000,len(z2)),False)],mm.predict(z2[np.random.default_rng(seed).choice(len(z2),min(5000,len(z2)),False)]))})
    pd.DataFrame(hyper).to_csv(tabs/'hyperparameter_sensitivity.csv',index=False)
    sens=pnl_sensitivity(corr,cfg,scored,labels); sens.to_csv(tabs/'pnl_sensitivity.csv',index=False)
    figures(figs,ev,pca,loadings,Z,labels,prof,scored,seed_df,boot_df,sens,k)
    joblib.dump(prep,models/'preprocessor.joblib'); joblib.dump(pca,models/'pca.joblib'); joblib.dump(km,models/'kmeans.joblib')
    # Actual computed run summary for docs.
    summary={'n_customers':n,'features':len(FEATURES),'k':k,'pca_components':pca.n_components_,'silhouette':float(ev.loc[ev.k==k,'silhouette'].iloc[0]),'mean_seed_ari':float(seed_df.ari.mean()),'top20_profit_share':top_profit,'mean_bootstrap_ari':float(boot_df.ari.mean())}
    pd.DataFrame([summary]).to_csv(tabs/'run_summary.csv',index=False)
    (ROOT/'outputs/tables/sensitivity_summary.txt').write_text(f"Seed stability mean ARI {summary['mean_seed_ari']:.3f} (SD {seed_df.ari.std():.3f}); bootstrap mean ARI {summary['mean_bootstrap_ari']:.3f}; top-20% customers generate {top_profit:.1%} of net portfolio contribution. Results describe synthetic data only.",encoding='utf-8')
    logging.info('Completed: %s',summary); return summary
if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--n-customers',type=int); ap.add_argument('--seed',type=int); a=ap.parse_args(); run(a.n_customers,a.seed)
