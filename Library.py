# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 15:45:58 2021

@author: Mustafa
This library algorithm wroten by Mustafa KAYNAK.

    This algorithm concern normal distributions. Used tools are skewness,kurtosis,mean,standard deviation

Normal Distributiion:
    f(x)=e−(x−μ)²/(2σ²)σ√2π
    
For univariate data Y1, Y2, ..., YN, the formula for kurtosis is:
    kurtosis=∑Ni=1(Yi−Y¯)⁴/N /s⁴

Skewness:
    Skewness = ∑Ni (Xi – X)³ / (N-1) * σ³

"""
import pandas as pd
import numpy as np


class Statisticc_dist:
    def __init__(self,url):
        self.url= url 
        y_std=self.url

    def std_dev(self):
        import statistics as stat
        y_std=self.url
        result_st= stat.stdev(y_std)
        #listeler_ysted= []
        #listeler_ysted.append(y_std)
        #result_st= np.std(listeler_ysted)
        #print('Standard deviation is :',result_st)
        return result_st
    
    def mean_std(self):
        import numpy as np
        y_std= self.url
        result_mean= np.mean(y_std)
        #print('median is: ',result_mean)
        return result_mean
    def kurtosis_normality(self):
        from scipy.stats import skew, kurtosis
        y_std= self.url
        result_kurt= kurtosis(y_std)
        #print('resuslt of kurtosis is: ',result_kurt)
        return result_kurt
       
    def skewness_normality(self):
        from scipy.stats import skew,kurtosis
        y_std= self.url
        result_sk= skew(y_std)
        #print('result of skewness is: ',result_sk)
        return result_sk 
       
