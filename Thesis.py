# -*- coding: utf-8 -*-
"""
Created on Sat Apr 10 16:18:26 2021

@author: Mustafa
"""


import pandas as pd #calling dataset and create DataFrame
import numpy as np #Statistical Calculation
import matplotlib.pyplot as plt #Visulization of dataset and variables

#Calling database froam excel with pandas library
database= pd.read_excel('okyanus.xlsx',index_col='Date')
print(database.shape)


#This step is definition of dependent and independent variables 
x= database.iloc[:,2:22]
y= database.iloc[:,0]#Thats my defect rate variables

"""
import statsmodels.api as sm
re= sm.OLS(y,x).fit()
print(re.summary())
"""


"""
this step is preprocessing for selecting features to more efficient machine learining accuray

"""

from sklearn.feature_selection import SelectKBest #for the best result of chi square test
from sklearn.feature_selection import chi2 #selecting nonparametrics test for my dataset

best_attributes = SelectKBest(score_func=chi2, k=10)
fitting_s=best_attributes.fit(x,y)
DataFrameScores = pd.DataFrame(fitting_s.scores_)
DataFrameFeatures = pd.DataFrame(x.columns)

#concat two dataframes for better visualization 
result_features = pd.concat([DataFrameFeatures,DataFrameScores],axis=1)
result_features.columns = ['Features','Score']
result_features.sort_values(by=['Score'], inplace=True)



x = database[['Facade Cladding Demands',           
              'Deformation Rate of Tools Rate',
              'Efficiency Rate',
          'Personnel Working Rate',
  'Transportation time in machine',
              'Station 4 (Bonding)',
                    'Using 5S Rate',
             'loss time in Machine',
            'Station 6 (Assembly)',  
          'Station 3(Pvc and Cord)']] #thats new values of x indepented variables



"""
This preprocessing step is Statistical Analysis. We look up something extreme and outlier values in dataset
I measure to dataset weather appropriate parametrics and nonparametrics values with normality test
It is including kurtosis and skewness values. 
İf my independent variables is in interval -1,50 and +1,50 as kurtosis and skewness  
,it is not going to be eliminated. This info belong to Mahalanobis

"""
from fitting_lib import Statisticc_dist#that is my library algoritms
#My library is including kurtosis, skewness, standard deviation, and mean values

list_skewness=[]
names_features=[]
list_kurtosis=[]
list_mean=[]
list_standardDev=[]
for i in x:
    for_url= x[i]
    Stdist= Statisticc_dist(url=for_url)
    skewness_result= float(Stdist.skewness_normality())
    kurtosis_result= float(Stdist.kurtosis_normality())
    mean_result= float(Stdist.mean_std())
    standardDev_result= float(Stdist.std_dev())
    list_skewness.append(skewness_result)
    list_kurtosis.append(kurtosis_result)
    list_mean.append(mean_result)
    list_standardDev.append(standardDev_result)
    names_features.append(i)
    
results = pd.DataFrame()
results['Features'] = names_features
results['Skewness result']=list_skewness
results['Kurtosis result']= list_kurtosis
results['Standard deviation']=list_standardDev
results['Mean']=list_mean
"""
This step is dimension reduction after eliminate unnecessary variables 
It is including PCA Prenciple component analyse. We reach to best dimension independent variables
Than, overfitting and underfitting situation is going to reduce with PCA

"""

from sklearn.decomposition import PCA
#♠pca_analysis= PCA(n_components=5)

pca= PCA(n_components=6)
pca.fit(x)
x= pca.transform(x)


"""
This Step is including train test split rate and something parameters about fitting algorithms

"""


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.23,random_state=14)




"""
This step is including main algorithms. these algorithms are supervised regression machine learning alg.
These algorithms
LinearRegression
LogisticsRegression
RandomForestRegressor
XGBosttRegressor
Keras ANN

"""


#First Algorithm is Artifical Neural Network
"""
from sklearn.neural_network import MLPRegressor
rke= MLPRegressor(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2),random_state=14)
rke.fit(x_train,y_train)
pre= rke.predict(x_test)

print(rke.score(x_train,y_train))
print(rke.score(x_test,y_test))
"""



#Second is xgboost regression

"""

from xgboost import XGBRegressor
xx= XGBRegressor()
xx.fit(x_train,y_train)
predi= xx.predict(x_test)

predict_xgb= xx.predict(x_test)
xg_train_score=xx.score(x_train,y_train)
xg_test_score=xx.score(x_test, y_test)
print(xg_train_score)
print(xg_test_score)
"""

#Third algorithm is Random Forest
"""
from sklearn.ensemble import RandomForestRegressor
rfc= RandomForestRegressor(n_estimators=900,random_state=18)
rfc.fit(x_train,y_train)
predic_forest= rfc.predict(x_test)
train_forest_score=rfc.score(x_train,y_train)
acc_score_forest=rfc.score(x_test,y_test)
print(train_forest_score)
print(acc_score_forest)
"""
"""

#In order to run best parameters
estimat= 80,85,90,95,100,105,110,115,120
min_sample= 1,2,3,4,5,6,7,8,9,10

best_score_forest=0
for i in estimat:
    for j in min_sample:
            rfc= RandomForestRegressor(n_estimators=i,max_depth=j)
            rfc.fit(x_train,y_train)
            acc_score_forest=rfc.score(x_test,y_test)
            if best_score_forest<acc_score_forest:
                best_score_forest=acc_score_forest
                bi=i
                bj=j
                print('optimization of parameters are : ',bi, bj)


"""
"""
from sklearn.linear_model import LinearRegression
lreg= LinearRegression()
lreg.fit(x_train,y_train)
lreg_predict= lreg.predict(x_test)

print(lreg.score(x_train,y_train))
print(lreg.score(x_test,y_test))

"""
"""
from sklearn.linear_model import LogisticRegression
log_re= LogisticRegression(solver='liblinear').fit(x_train,y_train)
log_re_predict= log_re.predict(x_test)

print(log_re.score(x_train,y_train))
print(log_re.score(x_test,y_test))
"""
"""
from sklearn.svm import SVR
svr_reg= SVR(kernel='rbf',degree=5,gamma='scale')
svr_reg.fit(x_train,y_train)
svr_reg_predict = svr_reg.predict(x_test)

print(svr_reg.score(x_train,y_train))
print(svr_reg.score(x_test,y_test))
"""

from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
lreg= LinearRegression()
lreg.fit(x_train,y_train)


from sklearn.preprocessing import PolynomialFeatures


Input=[('polynomial',PolynomialFeatures(degree=1)),('modal',LinearRegression())]
pipe=Pipeline(Input)
pipe.fit(x_train,y_train)
pipe_pre=pipe.predict(x_test)

print(pipe.score(x_train,y_train))
print(pipe.score(x_test,y_test))



