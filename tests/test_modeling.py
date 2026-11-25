import numpy as np
from sklearn.cluster import KMeans
from src.modeling import evaluate_k
def test_k_metrics():
 x=np.random.default_rng(1).normal(size=(100,3)); r=evaluate_k(x,2,3); assert len(r)==2 and {'silhouette','inertia','calinski_harabasz','davies_bouldin'}<=set(r.columns)
