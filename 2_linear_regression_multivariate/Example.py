#https://www.youtube.com/watch?v=J_LnPL3Qg70&list=PLeo1K3hjS3uvCeTYTeyfe0-rN5r8zn9rw&index=3

import pandas as pd
import matplotlib as plt
from sklearn import linear_model

#%%
df = pd.read_csv('homeprices.csv')
print(round(df['bedrooms'].mean()))
df['bedrooms']=df['bedrooms'].fillna(round(df['bedrooms'].mean()))

#%%
reg = linear_model.LinearRegression()
prices = df['price']
df1 = df.drop('price',axis='columns')
reg.fit(df1,prices)
print(reg.predict([[3000,3,40]]))
print(reg.predict([[2500,4,5]]))
