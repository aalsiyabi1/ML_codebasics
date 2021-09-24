import pandas as pd
import matplotlib.pyplot as plt
import pylab as pl
from sklearn import linear_model
from word2number import w2n
import seaborn as sns

# %%
df = pd.read_csv(r'2_linear_regression_multivariate/hiring.csv')

experience = df['experience'].fillna('zero')
df['experience_num'] = experience.apply(w2n.word_to_num)

mean_test_score = int(df['test_score(out of 10)'].mean())
df['test_score(out of 10)']=df['test_score(out of 10)'].fillna(mean_test_score)
df.rename(columns={'test_score(out of 10)':'test_score','interview_score(out of 10)':'interview_score'},inplace=True)
df1 = df[['experience_num','test_score','interview_score','salary($)']]

#%%
salary = df1['salary($)']
df2 = df1.drop('salary($)',axis='columns')

#%%
reg = linear_model.LinearRegression()
reg.fit(df2,salary)

#%%
print(reg.predict([[2,9,6]]))
