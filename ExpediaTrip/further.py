import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

df_train_new = pd.read_csv("data/train_new.csv")
df_test_new = pd.read_csv("data/test_new.csv")
df_train_new = df_train_new.sample(frac=0.2)
y_click=np.array(df_train_new['click_bool'])
y_book=np.array(df_train_new['booking_bool'])
position = np.array(df_train_new['position'])
cols_to_drop={"click_bool","booking_bool","position","gross_bookings_usd"}
df_train_new.drop(cols_to_drop,axis=1,inplace=True)

RF1 = RandomForestClassifier(n_estimators=150,n_jobs=-1)
RF1.fit(df_train_new,y_click)
RF2 = RandomForestClassifier(n_estimators=150,n_jobs=-1)
RF2.fit(df_train_new,y_book)

y1 = RF1.predict(df_test_new)
y2 = RF2.predict(df_test_new)

df_test_new["result_value"]=4*y2+y1
df_test_new = df_test_new.sort_values(['srch_id','result_value'],ascending=[True,False])
df_test_new=df_test_new.reset_index(drop=True)
result=df_test_new.filter(['srch_id','prop_id'],axis=1)
result.to_csv("data/result1.csv",index=False)
