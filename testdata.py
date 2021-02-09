import numpy as np
import pandas as pd
from scipy.stats import shapiro


def test2(data, col, lim, string):
    l=0
    h=0
    if np.any(data[col]<lim[0])==True:
        l=1
    if np.any(data[col]>lim[1])==True:
        h=1
    if l==1:
        print('Warning: Some '+string+' value(s) is(are) unexpectly low. Check your data!')
        ans=input()
    if h==1:
        print('Warning: Some '+string+' value(s) is(are) unexpectly high. Check your data!')
        ans=input()
        
        
def normality(x, x_name, a):
        
    if len(x)>=3:
        stat, p = shapiro(x)
          
        if p > a:
            norm=1
            print(x_name+': passed')
        else:
            norm=0
            print(x_name+': failed')
            
    else:
        p=1
        norm=1
            
    return p, norm
    

def correlation(x, dt, x_name, R):
        
    dt_num=pd.to_numeric(dt, downcast='float')
    corr = np.corrcoef(x, dt_num)
    r = corr[0,1]
    
    if abs(r)<R:
        corr=0
        print(x_name+': passed')
    elif abs(r)>=R:
        corr=1
        print(x_name+': failed')
        
    return r, corr