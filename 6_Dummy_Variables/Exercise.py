import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer


df = pd.read_csv('carprices.csv')
print(df)

sns.scatterplot(data=df,x='Mileage',y='Sell Price($)',hue='Car Model')
plt.show()

sns.scatterplot(data=df,x='Age(yrs)',y='Sell Price($)',hue='Car Model')
plt.show()

model = LinearRegression()
le = LabelEncoder()

dfle = df
dfle['Car Model'] = le.fit_transform(dfle['Car Model'])
X = dfle[['Car Model','Mileage','Age(yrs)']].values
y = dfle['Sell Price($)'].values

ct = ColumnTransformer([('Car Model', OneHotEncoder(),[0])], remainder='passthrough')

X = ct.fit_transform(X)
X = X[:,1:]

model.fit(X,y)
print(model.score(X,y))


