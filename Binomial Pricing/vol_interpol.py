# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 13:52:53 2019

@author: rbgud
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import interp1d
import datetime as dt_

#%%
df = pd.read_csv('C:/Users/rbgud/OneDrive/바탕 화면/UIUC/FE2/vol.csv', 
                 index_col=['Maturity'], parse_dates=['Maturity'],engine='python')

Moneyness80 = df['80']
vol = pd.DataFrame(Moneyness80)

#%%
def interpolation(start,date1,rate1,date2,rate2,target_date):
    
    x1 = (date1 - start).days
    x2 = (date2 - start).days
    x_target = (target_date - start).days
    target = (x_target - x1)*(rate2 - rate1)/(x2 - x1) + rate1
    
    return target


#%%
real_date = ([(dt_.date(2019,1,28) + dt_.timedelta(days=x)) for x in range(0, 1825)])
vol2 = pd.DataFrame(np.zeros((len(real_date),1)),index=real_date)
vol2.index = pd.to_datetime(vol2.index,format ='%Y-%m-%d')

#%%
for date in vol.index :
    
    if date in vol2.index:
       vol2.loc[date][0] = vol.loc[date]['80']
       
vol2.loc['2024-01-26']=0.224

#%%
for i in range(1,len(vol)):
       
    for date in vol2.index:
            
       target = date
       
       if target > vol.index[i-1] and target < vol.index[i]:
           
          start = dt_.datetime(2019,1,28)
          date1 = vol.index[i-1]
          date2 = vol.index[i]
          rate1 = vol.iloc[i-1]['80']
          rate2 = vol.iloc[i]['80']
          vol2.loc[target][0] = interpolation(start,date1,rate1,date2,rate2,target)  
              
  
              
              
              
  

