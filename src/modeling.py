import numpy as np, pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from .preprocessing import build_preprocessor

def evaluate_k(X,kmin=2,kmax=12,seed=42):
    rows=[]; rng=np.random.default_rng(seed); ids=rng.choice(len(X),min(10000,len(X)),replace=False); sample=X[ids]; train=X if len(X)<=30000 else X[rng.choice(len(X),30000,replace=False)]
    for k in range(kmin,kmax+1):
        km=KMeans(n_clusters=k,n_init=10,random_state=seed).fit(train); lab=km.labels_; slab=km.predict(sample)
        rows.append({'k':k,'inertia':km.inertia_,'silhouette':silhouette_score(sample,slab),'calinski_harabasz':calinski_harabasz_score(train,lab),'davies_bouldin':davies_bouldin_score(train,lab)})
    return pd.DataFrame(rows)
def fit_model(df,cfg,seed=None,threshold=None):
    seed=cfg['seed'] if seed is None else seed
    prep=build_preprocessor(); X=prep.fit_transform(df)
    pca=PCA(n_components=threshold or cfg['pca_variance_target'],svd_solver='full'); Z=pca.fit_transform(X)
    return prep,pca,Z
