import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

df_train_new = pd.read_csv("data/test_new0521.csv")
df_test_new = pd.read_csv("data/train_new0521.csv")

df_test_new_positive=df_train_new[df_train_new['click_bool']==1]
df_test_new_negative=df_train_new[df_train_new['click_bool']==0]
df_test_new_negative=df_test_new_negative.sample(frac=0.05)

combined_use=df_test_new_positive.append(df_test_new_negative)

y_click=np.array(combined_use['click_bool'])
y_book=np.array(combined_use['booking_bool'])
position = np.array(combined_use['position'])
cols_to_drop={"click_bool","booking_bool","position","gross_bookings_usd"}
combined_use.drop(cols_to_drop,axis=1,inplace=True)

# 130, 140, 150
RF1 = RandomForestClassifier(n_estimators=130,n_jobs=-1)
RF1.fit(combined_use,y_click)
RF2 = RandomForestClassifier(n_estimators=130,n_jobs=-1)
RF2.fit(combined_use,y_book)
y1 = RF1.predict(df_test_new)
y2 = RF2.predict(df_test_new)
df_test_new["result_value"]=4*y2+y1
df_test_new = df_test_new.sort_values(['srch_id','result_value'],ascending=[True,False])
df_test_new=df_test_new.reset_index(drop=True)
result=df_test_new.filter(['srch_id','prop_id'],axis=1)
result.to_csv("data/resultnew.csv",index=False)
