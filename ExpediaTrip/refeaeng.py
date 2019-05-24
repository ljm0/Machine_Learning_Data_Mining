import pandas as pd
import numpy as np
import re
import seaborn as sns
import matplotlib.pyplot as plt
import random

df_train = pd.read_csv("data/training_set_VU_DM.csv")
df_test = pd.read_csv("data/test_set_VU_DM.csv")

df_test['position']=0
df_test['click_bool']=0
df_test['booking_bool']=0
df_test['gross_bookings_usd']=0
combined_set=df_train.append(df_test)
pd.set_option('display.max_columns', None)

combined_set['prop_review_score'] = combined_set['prop_review_score'].fillna(combined_set['prop_review_score'].median())
combined_set['orig_destination_distance'] = combined_set['orig_destination_distance'].fillna(combined_set['orig_destination_distance'].median())
combined_set['prop_location_score2'] = combined_set['prop_location_score2'].fillna(combined_set['prop_location_score2'].mean())
combined_set['visitor_hist_adr_usd'].fillna(0, inplace=True)
combined_set['visitor_hist_starrating_bool'] = pd.notnull(combined_set['visitor_hist_starrating']) * 1
combined_set['date_time'] = pd.to_datetime(combined_set['date_time'])
for prop in ["month", "day", "hour", "minute", "dayofweek", "quarter"]:
    combined_set[prop] = getattr(combined_set["date_time"].dt, prop)

for i in range(1,9):
    combined_set['comp'+str(i)+'_rate'].fillna(0, inplace=True)
combined_set['comp_rate_sum'] = combined_set['comp1_rate']
for i in range(2,9):
    combined_set['comp_rate_sum'] += combined_set['comp'+str(i)+'_rate']

for i in range(1,9):
    combined_set['comp'+str(i)+'_inv'].fillna(0, inplace=True)
combined_set['comp_inv_sum'] = combined_set['comp1_inv']
for i in range(2,9):
    combined_set['comp_inv_sum'] += combined_set['comp'+str(i)+'_inv']

# cols_to_drop={'date_time','site_id','visitor_location_country_id','prop_country_id','srch_destination_id',"visitor_hist_starrating","prop_log_historical_price","srch_query_affinity_score","comp1_inv","comp2_inv","comp3_inv","comp4_inv","comp5_inv","comp6_inv","comp7_inv","comp8_inv","comp1_rate","comp2_rate","comp3_rate","comp4_rate","comp5_rate","comp6_rate","comp7_rate","comp8_rate"}
cols_to_drop={'date_time','site_id','visitor_location_country_id','prop_country_id','srch_destination_id',"visitor_hist_starrating","prop_log_historical_price","srch_query_affinity_score"}
# cols_to_drop={'date_time',"visitor_hist_starrating","srch_query_affinity_score"}
combined_set.drop(cols_to_drop,axis=1,inplace=True)
for i in range(1,9):
    combined_set.drop('comp'+str(i)+'_rate',axis=1,inplace=True)
    combined_set.drop('comp'+str(i)+'_inv',axis=1,inplace=True)
    combined_set.drop('comp'+str(i)+'_rate_percent_diff',axis=1,inplace=True)

train_new=combined_set[:4958347]
test_new=combined_set[4958347:]
test_to_drop={'position','gross_bookings_usd','click_bool','booking_bool'}
test_new.drop(test_to_drop,axis=1,inplace=True)

train_new.to_csv('data/train_new.csv',index=False)
test_new.to_csv('data/test_new.csv',index=False)


