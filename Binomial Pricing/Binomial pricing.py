# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 15:01:26 2019

@author: rbgud
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date
import datetime as dt_
from tqdm import tqdm
import scipy as sp
from pandas.tseries.holiday import AbstractHolidayCalendar, Holiday, nearest_workday, \
    USMartinLutherKingJr, USPresidentsDay, GoodFriday, USMemorialDay, \
    USLaborDay, USThanksgivingDay
#%% Calculate real trading day(except holiday and weekend)

class USTradingCalendar(AbstractHolidayCalendar):
    rules = [
        Holiday('NewYearsDay', month=1, day=1, observance=nearest_workday),
        USMartinLutherKingJr,
        USPresidentsDay,
        GoodFriday,
        USMemorialDay,
        Holiday('USIndependenceDay', month=7, day=4, observance=nearest_workday),
        USLaborDay,
        USThanksgivingDay,
        Holiday('Christmas', month=12, day=25, observance=nearest_workday)
    ]
def business_dates(start, end):
    us_cal = USTradingCalendar()
    kw = dict(start=start, end=end)
    return pd.DatetimeIndex(freq='B', **kw).drop(us_cal.holidays(**kw))

bd_list_2019_ = []
bd_other = []
bd_list_2019_real = []
bd_other_real = []
for j in range(6):
    if j==0:
        for i in range(11):
            if i+2 == 2:
               bd = len(business_dates(start=date(2019+j,1+i,28), end=date(2019+j,2+i,26)))
               bd_ = business_dates(start=date(2019+j,1+i,28), end=date(2019+j,2+i,26))
            else:
                 if  (i+2)%2 == 1:
                     bd = len(business_dates(start=date(2019+j,1+i,27), end=date(2019+j,2+i,26)))
                     bd_ = business_dates(start=date(2019+j,1+i,27), end=date(2019+j,2+i,26))
                 else:
                     bd = len(business_dates(start=date(2019+j,1+i,27), end=date(2019+j,2+i,26))) 
                     bd_ = business_dates(start=date(2019+j,1+i,27), end=date(2019+j,2+i,26))
            
            bd_list_2019_.append(bd)
            bd_list_2019_real.append(bd_)
    else:
        for i in range(12):
            if i+1 == 1 :
               bdc_ = len(business_dates(start=date(2019+j-1,12,27), end=date(2019+j,1+i,26)))
               bdc = business_dates(start=date(2019+j-1,12,27), end=date(2019+j,1+i,26))
            else:
                 if  (i+1)%2 == 1:
                     bdc_ = len(business_dates(start=date(2019+j,i,27), end=date(2019+j,1+i,26)))
                     bdc = business_dates(start=date(2019+j,i,27), end=date(2019+j,1+i,26))
                 else:
                     bdc_ = len(business_dates(start=date(2019+j,i,27), end=date(2019+j,1+i,26))) 
                     bdc = business_dates(start=date(2019+j,i,27), end=date(2019+j,1+i,26))
                  
            bd_other.append(bdc_)
            bd_other_real.append(bdc)

bd_node = bd_list_2019_ + bd_other
bd_node = bd_node[:59]
cpn_node = list(np.cumsum(bd_node))
cpn_node_ = cpn_node
bd_node_real = bd_list_2019_real + bd_other_real  

#%% Interpolate Interest rate and Volatility to maturity

def interpolation(start,date1,rate1,date2,rate2,target_date):
    
    x1 = (date1 - start).days
    x2 = (date2 - start).days
    x_target = (target_date - start).days
    target = (x_target - x1)*(rate2 - rate1)/(x2 - x1) + rate1
    
    return target

st = dt_.date(2019,1,28)
d1 = dt_.date(2023,10,3)
d2 = dt_.date(2024,10,3)
td = dt_.date(2024,1,26)

stv = dt_.date(2023,10,2)
d1v = dt_.date(2023,10,2)
d2v = dt_.date(2024,2,2)
tdv = dt_.date(2024,1,26)

r_est = interpolation(st,d1,0.03056,d2,0.03065,td)
vol_est = interpolation(stv,d1v,0.22527,d2v,0.22456,tdv)
[s,r,sigma,T,N,B,cpn,FV,q] = [2643.85,r_est,vol_est,5,1259,0.8*2643.85,0.0615/12,1000,0.0199]
cpn_node = [0]+list(np.cumsum(bd_node))

#%% Get volatilty data to make volatility term-structure

df = pd.read_csv('C:/Users/rbgud/OneDrive/바탕 화면/UIUC/FE2/vol.csv', 
                 index_col=['Maturity'], parse_dates=['Maturity'],engine='python')

Moneyness80 = df['80']
vol = pd.DataFrame(Moneyness80)
vol = vol.loc[vol.index < '2024-01-26']
vol = pd.concat([vol,pd.DataFrame(data = {'80':[vol_est]},index=[dt_.datetime(2024,1,26)])],axis=0)

real_date = ([(dt_.date(2019,1,28) + dt_.timedelta(days=x)) for x in range(0, 1825)])
implied_vol = pd.DataFrame(np.zeros((len(real_date),1)),index=real_date)
implied_vol.index = pd.to_datetime(implied_vol.index,format ='%Y-%m-%d')

for date_ in vol.index :
    
    if date_ in implied_vol.index:
       implied_vol.loc[date_][0] = vol.loc[date_]['80']
       
implied_vol.loc['2024-01-26']=vol_est

for i in range(1,len(vol)):
       
    for date_ in implied_vol.index:

       target = date_
       
       if target > vol.index[i-1] and target < vol.index[i]:
           
          start = dt_.datetime(2019,1,28)
          date1 = vol.index[i-1]
          date2 = vol.index[i]
          rate1 = vol.iloc[i-1]['80']
          rate2 = vol.iloc[i]['80']
          implied_vol.loc[target][0] = interpolation(start,date1,rate1,date2,rate2,target) 
                    
bd_real_date = pd.DataFrame()
for i in range(60):
    bd_real_date = pd.concat([bd_real_date,pd.Series(bd_node_real[i])],axis=0,ignore_index=True)

bd_real_date = bd_real_date.set_index(bd_real_date[0])
bd_implied_vol = implied_vol.loc[implied_vol.index.isin(bd_real_date.index)]  
bd_implied_vol.index = np.arange(0,len(bd_implied_vol))  
bd_implied_vol_ = list(bd_implied_vol[0]) 

def interpolation_vol(start,date1,rate1,date2,rate2,target_date):
    
    x1 = (date1 - start)
    x2 = (date2 - start)
    x_target = (target_date - start)
    target = (x_target - x1)*(rate2 - rate1)/(x2 - x1) + rate1
    
    return target

#%% Smoothing volatility term structure with respect to Moneyness 80%    
    
sigma_list = bd_implied_vol    
sigma_list.loc[340:415]=0
for j in np.arange(340,416,1):
    target = j              
    start = 339
    date1 = j
    date2 = 416
    rate1 = sigma_list.iloc[j-1][0]
    rate2 = sigma_list.iloc[416][0]
    sigma_list.loc[target] = interpolation_vol(start,date1,rate1,date2,rate2,target) 

#%% Pricing securities 
    
def RAN_tree(tree_type,r,q,sigma_list,cpn,bd_node,FV,B,T,N,step_multiplier):
    
    N_step = N*step_multiplier
    value_path = np.zeros([N_step + 1, N_step + 1])
    dt = T/N_step
    
    def impvol_interpol(N,step_multiplier,sigma_list):
        impvol = np.zeros(N*step_multiplier+1)
        sigma_list = np.reshape(np.array(sigma_list),(len(sigma_list)))
        for i in range(len(sigma_list)):
            impvol[i*step_multiplier] = sigma_list[i]
        
        for j in np.arange(0,N*step_multiplier,step_multiplier):
            for i in range(1,len(impvol)):
                if i%step_multiplier==0:
                   pass
                else:            
                   target = i
                   
                   if target > j and target < step_multiplier+j:
                       
                      start = 0
                      date1 = j
                      date2 = step_multiplier+j
                      rate1 = impvol[j]
                      rate2 = impvol[step_multiplier+j]
                      impvol[target] = interpolation_vol(start,date1,rate1,date2,rate2,target) 
        return pd.DataFrame(impvol)       

    sigma_list_ = impvol_interpol(N,step_multiplier,sigma_list)
      
    if tree_type == 'CRR':
        u  = np.exp(sigma_list_*np.sqrt(dt))
        d = 1/u
        
    elif tree_type == 'Binomial':
        u  = np.exp(r*dt+sigma_list_*np.sqrt(dt))
        d = np.exp(r*dt-sigma_list_*np.sqrt(dt))
        
    elif tree_type == 'Rendleman':
        u = np.exp((r-q-0.5*sigma_list_**2)*dt+sigma_list_*np.sqrt(dt))
        d = np.exp((r-q-0.5*sigma_list_**2)*dt-sigma_list_*np.sqrt(dt))
    
    cpn_node = list(np.array([0]+list(np.cumsum(bd_node)))*step_multiplier)   
    p = (np.exp((r-q)*dt)-d)/ (u-d)
    p = list(p[0])
    time_step1 = cpn_node[-1]
    time_step2 = 252*step_multiplier
    u.loc[0][0]=1
    d.loc[0][0]=1
    u = list(u[0])
    d = list(d[0])
    stock_path = np.zeros([N_step + 1, N_step + 1])
    for i in tqdm(range(N_step + 1)):
        for j in tqdm(range(i + 1)):
            stock_path[j, i] = s * (d[i-j]**(i-j))*(u[j]**j)   
        
    # Maturity payoff    
    for i in np.arange(N_step+1):
        if stock_path[i,-1] >B:
            value_path[i,-1] = FV
        else:
            value_path[i,-1] = FV * (1-(B-stock_path[i,-1])/s)
    
    # From last coupon day to maturity
    for i in np.arange(N_step-1,time_step1,-1):
        for j in range(i+1):
            value_path[j,i] =np.exp(-r*dt) * (p[i]* value_path[j+1,i+1]+(1-p[i])*value_path[j,i+1])
        
    # Early redemption with coupon payment
    for i in np.arange(time_step1,time_step2,-1):
        for j in range(i+1):
            if stock_path[j,i] >= B:
                if i in cpn_node:
                    value_path[j,i] = min(FV*np.exp(-r*(cpn_node[-1]-i)*dt),np.exp(-r*dt) * (p[i]* value_path[j+1,i+1]+(1-p[i])*value_path[j,i+1]))+\
                    (FV*cpn/(cpn_node[-1]-cpn_node[-2]))*np.exp(-r*(cpn_node[-1]-i)*dt)
                else:
                    value_path[j,i] =np.exp(-r*dt) * (p[i]* value_path[j+1,i+1]+(1-p[i])*value_path[j,i+1]) + (FV*cpn/(cpn_node[-1]-cpn_node[-2]))*np.exp(-r*(cpn_node[-1]-i)*dt)
            else:
                if i in cpn_node:
                    value_path[j,i] = min(FV*np.exp(-r*(cpn_node[-1]-i)*dt),np.exp(-r*dt) * (p[i]* value_path[j+1,i+1]+(1-p[i])*value_path[j,i+1]))
                else:
                    value_path[j,i] =np.exp(-r*dt) * (p[i]* value_path[j+1,i+1]+(1-p[i])*value_path[j,i+1])
                   
        if i<cpn_node[-2]:
            cpn_node = cpn_node[:-1]
            
    # No early redemption with coupon payment
    for i in np.arange(time_step2,-1,-1):
        for j in range(i+1):
            if stock_path[j,i] >= B:
                value_path[j,i] =np.exp(-r*dt) * (p[i]* value_path[j+1,i+1]+(1-p[i])*value_path[j,i+1]) + (FV*cpn/(cpn_node[-1]-cpn_node[-2]))*np.exp(-r*(cpn_node[-1]-i)*dt)
            else:
                value_path[j,i] =np.exp(-r*dt) * (p[i]* value_path[j+1,i+1]+(1-p[i])*value_path[j,i+1])
        
        
        while i<cpn_node[-2]:
            if  np.size(cpn_node)==2:
                break
            else:
                cpn_node = cpn_node[:-1]
    
    return value_path[0,0]            
     
#%% Get pricing results
    
crr = []
for i in range(1,5):
    crr_ = RAN_tree('CRR',r,q,sigma_list,cpn,bd_node,FV,B,T,N,i)
    crr.append(crr_)
    
binomial = []
for i in range(1,5):
    binomial_ = RAN_tree('Binomial',r,q,sigma_list,cpn,bd_node,FV,B,T,N,i)
    binomial.append(binomial_)  
    
rendleman = []
for i in range(1,5):
    rendleman_ = RAN_tree('Rendleman',r,q,sigma_list,cpn,bd_node,FV,B,T,N,i)
    rendleman.append(rendleman_)  
    
#%% We store the pricing results(since it is time consuming)
    
crr = [960.341,961.522,961.396,960.996,961.373,960.884,960.783,960.926,961.229]
binomial = [961.293,961.215,961.178,961.168,961.164,961.156,961.155,961.151,961.150]
rendleman = [961.313,961.193,961.176,961.168,961.164,961.159,961.153,961.152,961.146]

crr = pd.DataFrame(crr).set_index(np.arange(1259,11332,1259))
binomial = pd.DataFrame(binomial).set_index(np.arange(1259,11332,1259))
rendleman = pd.DataFrame(rendleman).set_index(np.arange(1259,11332,1259))

crr_ = pd.DataFrame(np.zeros((10073,1)),index = np.arange(1259,11332,1)) 
binomial_ = pd.DataFrame(np.zeros((10073,1)),index = np.arange(1259,11332,1))
rendleman_ = pd.DataFrame(np.zeros((10073,1)),index = np.arange(1259,11332,1))

#%% Interpolate security price with respect to Number of steps

for i in crr.index:
    if i in crr_.index:
       crr_.loc[i][0] = crr.loc[i][0]
       binomial_.loc[i][0] = binomial.loc[i][0]
       rendleman_.loc[i][0] =  rendleman.loc[i][0]
# Interpolate CRR model
for i in range(1,len(crr)):
       
    for node in crr_.index:
       if node in crr.index:
           pass
       else:

           target = node
       
       if target > crr.index[i-1] and target < crr.index[i]:
           
          start = 1259
          date1 = crr.index[i-1]
          date2 = crr.index[i]
          rate1 = crr.iloc[i-1][0]
          rate2 = crr.iloc[i][0]
          crr_.loc[target][0] = interpolation_vol(start,date1,rate1,date2,rate2,target) 
          
# Interpolate Binomial model
for i in range(1,len(binomial)):
       
    for node in binomial_.index:
       if node in binomial.index:
           pass
       else:

           target = node
       
       if target > binomial.index[i-1] and target < binomial.index[i]:
           
          start = 1259
          date1 = binomial.index[i-1]
          date2 = binomial.index[i]
          rate1 = binomial.iloc[i-1][0]
          rate2 = binomial.iloc[i][0]
          binomial_.loc[target][0] = interpolation_vol(start,date1,rate1,date2,rate2,target)   

# Interpolate Rendleman & Bartter model
for i in range(1,len(rendleman)):
       
    for node in rendleman_.index:
       if node in rendleman.index:
           pass
       else:

           target = node
       
       if target > rendleman.index[i-1] and target < rendleman.index[i]:
           
          start = 1259
          date1 = rendleman.index[i-1]
          date2 = rendleman.index[i]
          rate1 = rendleman.iloc[i-1][0]
          rate2 = rendleman.iloc[i][0]
          rendleman_.loc[target][0] = interpolation_vol(start,date1,rate1,date2,rate2,target)
            
#%% Error Analysis
          
offered_price = 953.22          
crr_error = (crr_- offered_price)
binomial_error = (binomial_ - offered_price)
rendleman_error = (rendleman_ - offered_price)
crr_error.columns = ['CRR Error']
binomial_error.columns = ['Binomial Error']
rendleman_error.columns = ['Rendleman & Bartter Error']

#%% Plot Error 

plot_list = [crr_error,binomial_error,rendleman_error]
name_list = ['CRR Error','Binomial Error','Rendleman & Bartter Error']
for i in range(len(plot_list)):
    
    plt.figure()
    plt.scatter(plot_list[i].index,plot_list[i],s=1.5,label=name_list[i])
    plt.xlabel('Number of Steps')
    plt.ylabel('Error of price($)')
    plt.title('Dollar Error of price (FV = 1000)')
    plt.legend(loc='best')
    plt.show()

