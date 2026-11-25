import numpy as np,pandas as pd
from src.censoring import correct_censoring
def test_censoring():
 d=pd.DataFrame({'x':[1.,2.,3.,10.,10.],'x_censored':[False,False,False,True,True]})
 o,s=correct_censoring(d,{'x':10}); assert (o.loc[:2,'x_corrected']==d.loc[:2,'x']).all(); assert (o.loc[3:,'x_corrected']>=10).all()
def test_pareto_improves_mean():
 rng=np.random.default_rng(4); x=rng.pareto(4,20000)+1; cap=2.5; true=x.mean(); d=pd.DataFrame({'x':np.minimum(x,cap)}); o,_=correct_censoring(d,{'x':cap}); assert abs(o.x_corrected.mean()-true)<abs(d.x.mean()-true)
