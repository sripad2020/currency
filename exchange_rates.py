import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
import pandas as pd
data=pd.read_csv('exchange_rate.csv')
print(data.columns)
abc=[data.columns]
for i in range(53):
    row1=data.iloc[i].values
    import seaborn as sn
    colms = data.iloc[:, 0].values
    sn.boxplot(row1[1:])
    plt.xlabel(colms[i])
    plt.show()
sn.heatmap(data.corr())
plt.show()
print('-----The Shape of the data---------')
print(data.shape)
print('----The Nan value information---')
print(data.isna().sum())
print('----The data in HTML format-----')
print(data.to_html('file.html'))
print('---The data description---')
print(data.describe())
print('-----The plots of data-----')
data.plot()
plt.show()
print('--The unique values---')
print(data.nunique(axis=0))
print('----The data types of the datasets--')
print(data.dtypes)
#colms=['eur','gbp','inr','aud','cad','sgd','chf','myr','jpy','cny','ars','bhd','bwp','brl','bnd','bgn','clp','cop','hrk','czk','dkk','hkd','huf','isk','idr','irr','ils','kzt','krw','kwd','lyd','mur','mxn','npr','nzd','nok','omr','pkr','php','pln','qar','ron','rub','sar','zar','lkr','sek','twd','thb','ttd','try','aed','vef']
for i in range(0,53):
    curr=data.iloc[i].values
    col=data.columns.values
    colms = data.iloc[:, 0].values
    plt.plot(col[1:15],curr[1:15])
    plt.xlabel(colms[i])
    plt.legend()
    plt.show()
import keras.activations,keras.losses
from keras.models import  Sequential
from keras.layers import Dense
colms=data.columns.values
INR=data.iloc[:,2].values
inr_50=np.array(INR[1:50])
colms=np.array(colms[1:50])
from sklearn.preprocessing import LabelEncoder
lab=LabelEncoder()
from sklearn.model_selection import train_test_split
df=pd.DataFrame(inr_50)
df1=pd.DataFrame(colms)
df1['dates']=lab.fit_transform(colms)
from sklearn.model_selection import train_test_split
x_tr,x_te,y_tr,y_te=train_test_split(df[1:25],df1.dates[1:25])
model=Sequential()
model.add(Dense(units=50,input_dim=30,activation=keras.activations.relu))
model.add(Dense(units=50,activation=keras.activations.relu))
model.add(Dense(units=50,activation=keras.activations.relu))
model.add(Dense(units=50,activation=keras.activations.relu))
model.add(Dense(units=50,activation=keras.activations.relu))
model.add(Dense(units=1,activation=keras.activations.relu))
model.compile(optimizer='adam',loss=keras.losses.mean_absolute_error,metrics=['mae'])
#model.fit(df,df1.dates,batch_size=10,epochs=25)
#survival analytics
from lifelines import KaplanMeierFitter
kap=KaplanMeierFitter()
kap.fit(df[1:25],df1.dates[1:25])
pr=kap.predict(15)
print(pr)
kap.plot()
plt.ylabel('This is probability with 6 intervals')
plt.legend()
plt.show()
#df=pd.PeriodIndex(df,df1.dates)
#Time Series analysis
from statsmodels.api import tsa
exp_smoothing=tsa.ExponentialSmoothing(df).fit()
fore=exp_smoothing.forecast(50)
print(exp_smoothing.level)
print(exp_smoothing.forecast(50))
print(exp_smoothing.season)
plt.figure(figsize=(16,8))
plt.plot(df,label='not_forecasted')
plt.plot(fore,label='forecasted')
plt.plot(exp_smoothing.sse,label='sse')
plt.legend()
plt.show()