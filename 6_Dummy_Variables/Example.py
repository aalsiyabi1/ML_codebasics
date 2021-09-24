import pandas as pd
df = pd.read_csv("homeprices.csv")
print(df)

dummies = pd.get_dummies(df.town)
print(dummies)

merged = pd.concat([df,dummies],axis='columns')
print(merged)

final = merged.drop(['town'], axis='columns')
final = final.drop(['west windsor'], axis='columns')

print(final)

X = final.drop('price', axis='columns')
y = final.price

from sklearn.linear_model import LinearRegression
model = LinearRegression()

print(model.fit(X,y))
print(model.predict(X))
print(model.score(X,y))

print(model.predict([[3400,0,0]]))

print(model.predict([[2800,0,1]]))

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

dfle = df
dfle.town = le.fit_transform(dfle.town)
X = dfle[['town','area']].values
y = dfle.price.values

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([('town', OneHotEncoder(), [0])], remainder = 'passthrough')

X = ct.fit_transform(X)
X = X[:,1:]

print(model.fit(X,y))
print(model.predict([[0,1,3400]]))
print(model.predict([[1,0,2800]]))

