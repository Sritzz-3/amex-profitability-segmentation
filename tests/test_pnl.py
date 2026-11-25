import pandas as pd
from src.pnl import add_net_contribution
from src.config import load_config
def test_hand_calculation():
 d=pd.DataFrame([dict(annual_spend=1000,avg_revolving_balance=1000,apr=.2,payment_ratio=0,annual_fee=0,late_payment_count=0,cash_advance_amt=0,rewards_points_earned=0,rewards_redeemed_ratio=0,credit_score=800,utilization_ratio=0,customer_service_contacts=0)])
 c=load_config(); o=add_net_contribution(d,c); expected=200+18-0-1000*.004*.85-35-45-45; assert abs(o.net_contribution.iloc[0]-expected)<1e-8
