import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model

df = pd.read_csv('homeprices.csv')
df.head()

# %%
new_df = df.drop('price', axis='columns')
price = df.price
reg = linear_model.LinearRegression()
reg.fit(new_df, price)

print(reg.predict([[3300]]))
print(reg.coef_)
print(reg.intercept_)

# %%
plt.scatter(df.area, df.price, marker='+', color='red')
plt.plot(df.area,reg.predict(df[['area']]),color='blue')
plt.xlabel('Area (sqr ft)')
plt.ylabel('Price ($USD)')

plt.show()

