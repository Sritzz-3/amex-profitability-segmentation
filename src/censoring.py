import logging
import numpy as np
from scipy.stats import genpareto
LOG=logging.getLogger(__name__)
def correct_censoring(df, caps, method='pareto'):
    out=df.copy(); stats={}
    for col,cap in caps.items():
        mask=out[col].ge(cap); corrected=col+'_corrected'; out[corrected]=out[col].astype(float).fillna(out.loc[~mask,col].median())
        count=int(mask.sum()); uplift=0.0
        if count and method=='pareto':
            vals=out.loc[~mask,col].dropna().to_numpy(); threshold=np.quantile(vals,.9) if len(vals)>20 else np.min(vals)
            excess=vals[vals>threshold]-threshold
            try:
                shape,loc,scale=genpareto.fit(excess,floc=0)
                threshold_cap=max(cap-threshold,0)
                if shape < 1:
                    conditional=threshold+threshold_cap+(scale+shape*threshold_cap)/(1-shape)
                else: conditional=threshold+scale/(1-min(shape,.999))
                ceiling=max(cap,threshold+genpareto.ppf(.999,shape,loc=loc,scale=scale))
                replacement=float(np.clip(max(cap,conditional),cap,ceiling))
            except Exception: replacement=cap*1.15
            out.loc[mask,corrected]=replacement; uplift=replacement-cap
        stats[col]={'corrected_count':count,'mean_uplift':uplift}
        LOG.info('%s: corrected %d records; mean uplift %.2f',col,count,uplift)
    return out,stats
