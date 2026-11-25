from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt,seaborn as sns,numpy as np
sns.set_theme(style='whitegrid',context='notebook')
def figures(out,ev,pca,loadings,Z,labels,profiles,scored,seed_df,boot_df,sens,k):
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    plt.figure(figsize=(8,5)); plt.plot(np.arange(1,len(pca.explained_variance_ratio_)+1),pca.explained_variance_ratio_,marker='o',label='Component'); plt.plot(np.arange(1,len(pca.explained_variance_ratio_)+1),np.cumsum(pca.explained_variance_ratio_),marker='.',label='Cumulative'); plt.axhline(.9,ls='--',c='gray'); plt.xlabel('Principal component'); plt.ylabel('Explained variance'); plt.title('PCA explained variance'); plt.legend(); plt.tight_layout(); plt.savefig(out/'pca_scree.png',dpi=150); plt.close()
    fig,ax=plt.subplots(1,2,figsize=(12,4)); ax[0].plot(ev.k,ev.inertia,marker='o'); ax[0].set_title('K selection: inertia'); ax[0].set_xlabel('k'); ax[1].plot(ev.k,ev.silhouette,marker='o'); ax[1].set_title('Silhouette'); ax[1].set_xlabel('k'); fig.tight_layout(); fig.savefig(out/'k_diagnostics.png',dpi=150); plt.close(fig)
    # First two components shown (or one-dimensional fallback).
    ix=np.random.default_rng(42).choice(len(Z),min(20000,len(Z)),replace=False); plt.figure(figsize=(8,6)); plt.scatter(Z[ix,0],Z[ix,1] if Z.shape[1]>1 else np.zeros(len(ix)),c=labels[ix],s=5,cmap='tab10',alpha=.5); plt.xlabel('PC1'); plt.ylabel('PC2'); plt.title('Customer segments (PCA sample)'); plt.tight_layout(); plt.savefig(out/'pca_clusters.png',dpi=150); plt.close()
    means=profiles.set_index('cluster')[[c for c in profiles if c.startswith('mean_')]]; means=(means-means.mean())/means.std().replace(0,1); plt.figure(figsize=(11,5)); sns.heatmap(means,cmap='vlag',center=0); plt.title('Cluster profile heatmap'); plt.tight_layout(); plt.savefig(out/'cluster_heatmap.png',dpi=150); plt.close()
    plt.figure(figsize=(9,5)); sns.boxplot(data=scored,x='cluster',y='net_contribution',showfliers=False); plt.title('Net contribution by cluster'); plt.tight_layout(); plt.savefig(out/'contribution_boxplot.png',dpi=150); plt.close()
    vals=np.sort(scored.net_contribution.to_numpy())[::-1]; cum=np.cumsum(vals); cum=cum/cum[-1]; plt.figure(figsize=(8,5)); plt.plot(np.arange(1,len(vals)+1)/len(vals),cum); plt.axvline(.2,ls='--',c='r'); plt.xlabel('Share of customers, ranked by contribution'); plt.ylabel('Cumulative profit share'); plt.title('Cumulative profit curve'); plt.tight_layout(); plt.savefig(out/'profit_curve.png',dpi=150); plt.close()
    sens.groupby('assumption').jaccard_top20.min().sort_values().plot.barh(figsize=(8,5)); plt.xlabel('Worst top-20% overlap'); plt.title('Assumption sensitivity'); plt.tight_layout(); plt.savefig(out/'sensitivity_tornado.png',dpi=150); plt.close()
    fig,ax=plt.subplots(1,2,figsize=(10,4)); sns.boxplot(y=seed_df.ari,ax=ax[0]); ax[0].set_title('Seed stability ARI'); sns.boxplot(y=boot_df.ari,ax=ax[1]); ax[1].set_title('Bootstrap stability ARI'); fig.tight_layout(); fig.savefig(out/'stability_ari.png',dpi=150); plt.close(fig)
