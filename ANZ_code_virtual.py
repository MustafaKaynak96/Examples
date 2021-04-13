# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 00:32:41 2021

@author: Mustafa
"""


import pandas as pd
import numpy as np

database= pd.read_excel('ANZ_database.xlsx')
#print(database.shape)
#print(database.info)
print(database.info)
print(database.shape)
print(database.isnull().sum())

database=database.fillna(method="ffill")


database = database[["status", "card_present_flag", "balance", "txn_description","gender", "age",
    "merchant_suburb", "merchant_state", "amount","date",
     "movement"]]
database["date"] = pd.to_datetime(database["date"])

print(database.info)
print(database.shape)
print(database.isnull().sum())

database=database.fillna(method="ffill")



from sklearn.preprocessing import LabelEncoder
le= LabelEncoder()
database['status']=le.fit_transform(database['status'])
database['txn_description']=le.fit_transform(database['txn_description'])
database['gender']=le.fit_transform(database['gender'])

database['merchant_suburb']=le.fit_transform(database['merchant_suburb'])
database['merchant_state']= le.fit_transform(database['merchant_state'])
database['movement']=le.fit_transform(database['movement'])



x= database.iloc[:,0:9]
y=database.iloc[:,10:11]






from sklearn.feature_selection import SelectKBest #for the best result of chi square test
from sklearn.feature_selection import chi2 #selecting nonparametrics test for my dataset

best_attributes = SelectKBest(score_func=chi2, k=6)
fitting=best_attributes.fit(x,y)
DataFrameScores = pd.DataFrame(fitting.scores_)
DataFrameFeatures = pd.DataFrame(x.columns)

#concat two dataframes for better visualization 
result_features = pd.concat([DataFrameFeatures,DataFrameScores],axis=1)
result_features.columns = ['Features','Score']
result_features.sort_values(by=['Score'], inplace=True)
print(result_features.nlargest(10,'Score'))



"""
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.35,random_state=14)


from sklearn.ensemble import RandomForestClassifier
rfc= RandomForestClassifier(n_estimators=100)
rfc.fit(x_train,y_train)
predic= rfc.predict(x_test)
print(rfc.score(x_train,y_train))
print(rfc.score(x_test,y_test))

from sklearn.metrics import confusion_matrix
cm= confusion_matrix(y_test, predic)
print(cm)

"""



import statsmodels.api as sm
result= sm.OLS(y,x).fit()
print(result.summary())




cc = database[["status", "card_present_flag", "balance", "txn_description","gender", "age",
    "merchant_suburb", "merchant_state", "amount","date",
     "movement"]].corr()
print(cc)



import matplotlib.pyplot as plt
import seaborn as sns


#Visulization of dataset step

virt_data= pd.read_excel('ANZ_database.xlsx')
#print(database.shape)
#print(database.info)

virt_data = virt_data[["status", "card_present_flag", "balance", "txn_description","gender", "age",
    "merchant_suburb", "merchant_state", "amount","date",
     "movement"]]
virt_data["date"] = pd.to_datetime(virt_data["date"])

virt_data=virt_data.fillna(method="ffill")

plt.figure(figsize=(12,8))
sns.set_context('paper',font_scale=1.4)

cras_m= database.corr()
sns.heatmap(cras_m,annot=True)


ng= sns.PairGrid(virt_data)
ng = ng.map_diag(plt.hist)
ng = ng.map_offdiag(plt.scatter)


databa_g= sns.PairGrid(virt_data,hue='movement')
databa_g.map(plt.scatter)
#plt.savefig('AZN3-figure.png')

g = sns.PairGrid(virt_data, hue="movement")
g.map_diag(plt.hist)
g.map_offdiag(plt.scatter)
g.add_legend()




#list_countplot=[]
dataliste= virt_data[["status", "txn_description", "card_present_flag","gender", "age",
    "merchant_suburb", "merchant_state","movement"]]

for i in dataliste:
    new_x= dataliste[i]
    cou=sns.countplot(x=new_x,data=dataliste)
    sns.set(rc={'figure.figsize':(11.7,8.27)})
    plt.show()
    #list_countplot.append(cou)
    




