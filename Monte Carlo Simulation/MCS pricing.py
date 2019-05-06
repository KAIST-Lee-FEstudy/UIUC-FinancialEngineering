# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 21:54:04 2019

@author: rbgud
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as web
from tqdm import tqdm
from matplotlib import style
import datetime as dt
style.use('ggplot')

#interpolation formula
def interpolation(start,date1,rate1,date2,rate2,target_date):
    
    x1 = (date1 - start).days
    x2 = (date2 - start).days
    x_target = (target_date - start).days
    target = (x_target - x1)*(rate2 - rate1)/(x2 - x1) + rate1
    return target
#%%
#parameter setting
#get historical data from yahoo finance
start = dt.datetime(2008,4,3)
end = dt.datetime(2019,4,3)

pricing_date = dt.date(2018,4,4)
maturity_date = dt.date(2023,9,11)
avg_start = dt.date(2023,6,6)
avg_end = dt.date(2023,9,6)
#Get indices data from yahoo finance to calculate correlation
stx = web.DataReader("^GSPC","yahoo",start,end)['Close']
stx_ret = (np.log(stx) - np.log(stx).shift(1))[1:]
rty = web.DataReader("^RUT","yahoo",start,end)['Close']
rty_ret = (np.log(rty) - np.log(rty).shift(1))[1:]
rho = np.corrcoef(stx_ret,rty_ret)[0][1]
#rho = 0.92

#five year tresury note
five_year = dt.date(2023,4,4)
r_1 = web.DataReader("^FVX","yahoo",pricing_date).loc[pricing_date]['Close'] / 100
#ten year tresury note
ten_year = dt.date(2028,4,4)
r_2 = web.DataReader("^TNX","yahoo",pricing_date).loc[pricing_date]['Close'] / 100

r = interpolation(pricing_date,five_year,r_1,ten_year,r_2,maturity_date)
#r = 0.02
stx_div = 0.015
rty_div = 0.01
#initial value는 4월 3일 기준. vol도 이 기준에 맞게 뽑기
stx0 = 2614.45
rty0 = 1512.155
n_trials = 30000
T = (maturity_date-pricing_date).days/365
FV = 1000
n_steps = (maturity_date - pricing_date).days
avg_node = [(avg_start - pricing_date).days, (avg_end - pricing_date).days]
#estimate sigma
stx_vol = pd.read_csv("VOL_STX.csv",\
                         index_col =['Mat'], parse_dates=['Mat'],engine='python') /100
rty_vol = pd.read_csv("VOL_RTY.csv",\
                         index_col =['Mat'], parse_dates=['Mat'],engine='python') /100
stx_sigma = interpolation(pricing_date,dt.date(2022,12,30),stx_vol.iloc[2,1],dt.date(2023,12,29),stx_vol.iloc[3,1],maturity_date)
rty_sigma = interpolation(pricing_date,dt.date(2022,12,30),rty_vol.iloc[2,1],dt.date(2023,12,29),rty_vol.iloc[3,1],maturity_date)

#%%
def project3_sim(stx0,rty0,r,stx_div,rty_div,stx_sigma,rty_sigma,rho,T,FV,avg_node,n_steps,n_trials):
    maximum_value = FV*2.1164
    d_t = T/n_steps
    avg_start_node = avg_node[0] ; avg_end_node = avg_node[1]
    z_matrix1 = np.random.standard_normal((n_trials,n_steps))
    z_matrix2 = np.random.standard_normal((n_trials,n_steps))
    z_matrix_stx = z_matrix1
    z_matrix_rty = rho*z_matrix1 + np.sqrt(1 - rho**2)*z_matrix2
    stx_matrix = np.zeros((n_trials,n_steps))
    rty_matrix = np.zeros((n_trials,n_steps))
    stx_matrix[:,0] = stx0
    rty_matrix[:,0] = rty0
    for j in range(n_steps-1):
        stx_matrix[:,j+1] = stx_matrix[:,j]*np.exp((r-stx_div-0.5*stx_sigma**2)*d_t+stx_sigma*np.sqrt(d_t)*z_matrix_stx[:,j])
        rty_matrix[:,j+1] = rty_matrix[:,j]*np.exp((r-rty_div-0.5*rty_sigma**2)*d_t + rty_sigma*np.sqrt(d_t)*z_matrix_rty[:,j])

    avg_stx= np.mean(stx_matrix[:,avg_start_node:avg_end_node],axis=1)
    avg_rty = np.mean(rty_matrix[:,avg_start_node:avg_end_node],axis = 1)
    payoff = np.zeros(n_trials)

    for i in tqdm(range(n_trials)):
        if avg_stx[i] >= 1.21*stx0 and avg_rty[i] >= 1.21*rty0:
            payoff[i] = min(FV+ FV*(min(avg_stx[i]/stx0,avg_rty[i]/rty0)-1.21)*3.34 + 415,maximum_value)
        elif avg_stx[i] <1.21*stx0 or avg_rty[i] < 1.21*rty0:
            if avg_stx[i] >=stx0 and avg_rty[i]>=rty0:
                payoff[i] = FV+FV*(min(avg_stx[i]/stx0,avg_rty[i]/rty0)-1)*1.5 + 100
            elif avg_stx[i] < stx0 or avg_rty[i] < rty0:
                if avg_stx[i]>=0.95*stx0 and avg_rty[i]>= 0.95*rty0:
                    payoff[i] = FV+FV*(min(avg_stx[i]/stx0,avg_rty[i]/rty0)-0.95)*2
                elif avg_stx[i] < 0.95*stx0 or avg_rty[i] < 0.95*rty0:
                    payoff[i] = FV*min(avg_stx[i]/stx0,avg_rty[i]/rty0) + 50
                    
    value = np.mean(payoff)*np.exp(-r*T)
    
    return value
#%%
price = project3_sim(stx0,rty0,r,stx_div,rty_div,stx_sigma,rty_sigma,rho,T,FV,avg_node,n_steps,n_trials)
print('-'*50)
print('price is %4.3f'%(price))
print('-'*50)
#%%
# =============================================================================
# Sensitivity analysis
# =============================================================================

#1. depending on correlation
#rho_range = np.arange(-1,1.01,0.1)
#rho_price = []
#for est_rho in rho_range:
#    est_pice = project3_sim(stx0,rty0,r,stx_div,rty_div,stx_sigma,rty_sigma,est_rho,T,FV,avg_node,n_steps,n_trials)
#    rho_price.append(est_pice)
#
#plt.figure(0)
#plt.plot(rho_range,rho_price,label = 'price')
#plt.scatter(rho,price,label = 'current price',color='b')
#plt.title('price depending on rho change')
#plt.ylabel('price')
#plt.xlabel('rho')
#plt.legend()
#
##2.depending on dividend yield
#div_range = np.arange(0,0.1,0.01)
#div_price = []
#for est_div in div_range:
#    est_price = project3_sim(stx0,rty0,r,est_div,rty_div,stx_sigma,rty_sigma,rho,T,FV,avg_node,n_steps,n_trials)
#    div_price.append(est_price)
#    
#plt.figure(1)
#plt.plot(div_range,div_price,label = 'price')
#plt.scatter(stx_div,price,label = 'current price',color='b')
#plt.title('price depending on stx dividend change')
#plt.ylabel('price')
#plt.xlabel('dividend yield')
#plt.legend()
#
##3depending on Volatility
#vol_range = np.arange(0,0.5,0.05)
#vol_price = []
#for est_vol in vol_range:
#    est_price = project3_sim(stx0,rty0,r,stx_div,rty_div,est_vol,rty_sigma,rho,T,FV,avg_node,n_steps,n_trials)
#    vol_price.append(est_price)
#    
#plt.figure(2)
#plt.plot(vol_range,vol_price,label = 'price')
#plt.scatter(stx_sigma,price,label = 'current price',color='b')
#plt.title('price depending on stx volatility change')
#plt.ylabel('price')
#plt.xlabel('standard deviation')
#plt.legend()
#%%
#4.depending on risk free rate
r_range= np.arange(0,0.09,0.01)
r_price = []
for est_r in r_range:
    est_price = project3_sim(stx0,rty0,est_r,stx_div,rty_div,stx_sigma,rty_sigma,rho,T,FV,avg_node,n_steps,n_trials)
    r_price.append(est_price)

plt.figure(3)
plt.plot(r_range,r_price,label = 'price')
plt.scatter(r,price,label = 'current price',color='b')
plt.title('price depending on risk free rate change')
plt.ylabel('price')
plt.xlabel('risk free rate')
plt.legend()
#%%
trial_range = range(10000,100000,10000)
trials_price = []
for est_trials in trial_range:
    est_price = project3_sim(stx0,rty0,r,stx_div,rty_div,stx_sigma,rty_sigma,rho,T,FV,avg_node,n_steps,est_trials)
    trials_price.append(est_price)
    
plt.figure(4)
plt.plot(trial_range,trials_price,label = 'price')
plt.scatter(n_trials,price,label = 'current price',color='b')
plt.title('price depending on the number of simulation change')
plt.ylabel('price')
plt.xlabel('the number of simulation')
plt.legend()
