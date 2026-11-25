import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from .data_generator import FEATURES
SKEWED={'annual_income','credit_limit','annual_spend','txn_count','avg_txn_value','travel_spend','dining_spend','intl_spend','cash_advance_amt','avg_revolving_balance','rewards_points_earned'}
def log_skewed(a):
    a=np.asarray(a,dtype=float).copy()
    for j,name in enumerate(FEATURES):
        if name in SKEWED: a[:,j]=np.log1p(np.maximum(a[:,j],0))
    return a
class Winsorizer(BaseEstimator,TransformerMixin):
    def __init__(self,lower=.005,upper=.995): self.lower=lower; self.upper=upper
    def fit(self,X,y=None): self.low_=np.quantile(X,self.lower,axis=0); self.high_=np.quantile(X,self.upper,axis=0); return self
    def transform(self,X): return np.clip(X,self.low_,self.high_)
def build_preprocessor():
    return Pipeline([('imputer',SimpleImputer(strategy='median')),('log1p',FunctionTransformer(log_skewed)),('winsorize',Winsorizer()),('scaler',StandardScaler())])
