# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 17:59:18 2018

@author: rbgud

"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
from tqdm import tqdm
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

[sigma,r,div,s0,Barrier_level,cpn_rate,FV,upper_ratio,S_node,T_node,T]=\
[0.25,0.025,0.007,195.09,0.78,0.020375,1000,2,100,260,1.25]
issue_date = dt.date(2019,3,26)
cpn_date = [dt.date(2019,6,26),dt.date(2019,9,26),dt.date(2019,12,27),dt.date(2020,3,26),dt.date(2020,6,25)]

#%%
def FDM_pricing(s0,r,div,sigma,Barrier_level,cpn_rate,FV,upper_ratio,S_node,T_node,T,issue_date,cpn_date,method,print_result=False):
    
    Barrier = s0*Barrier_level
    lower = 0
    upper = s0*upper_ratio
    ds = (upper-lower)/S_node
    d_t = T/T_node
    s_range = np.linspace(lower,upper,S_node+1)
    barrier_node = int(Barrier/ds)
    s0_node = int(S_node/2)
    
    def TDMAsolver(a, b, c, d):
 
        nf = len(d)     # number of edivuations
        ac, bc, cc, dc = map(np.array, (a, b, c, d))     # copy the array
        for it in range(1, nf):
            mc = ac[it-1]/bc[it-1]
            bc[it] = bc[it] - mc*cc[it-1] 
            dc[it] = dc[it] - mc*dc[it-1]
    
        xc = bc
        xc[-1] = dc[-1]/bc[-1]
    
        for il in range(nf-2, -1, -1):
            xc[il] = (dc[il]-cc[il]*xc[il+1])/bc[il]
    
#        del bc, cc, dc  # delete variables from memory
    
        return xc  
    
    # Change coupon payment date to node 
    cnp_node = []
    for i in range(len(cpn_date)):
        cnp_node.append(int(((cpn_date[i]-issue_date).days/365)/d_t))
        
    ## Boundary condition    
    FDM_value = np.zeros((S_node,T_node+1))
    
    # Terminal coundary condition
    for i in range(S_node): 
        if s_range[i] >= s0 or s_range[i]>=Barrier:
            FDM_value[i,-1] = 1000*(1+cpn_rate)
            
        elif s_range[i] < s0 and s_range[i]>Barrier:
            FDM_value[i,-1] = 1000*(cpn_rate+(s_range[i]-s0)/s0)
            
        else:
            FDM_value[i,-1] = 1000*(1+(s_range[i]-s0)/s0)
            
    # Lower boundary condition
    FDM_value[0,:]= 0
    # Upper boundary condition
    indexing = 0
    for i in range(T_node):
        FDM_value[-1,i] = 1000*(1+cpn_rate) * np.exp(-(cnp_node[indexing] - i)*d_t)
        if i>= cnp_node[indexing]:
            indexing = indexing +1
    
    FDM_LU_value = FDM_value.copy()         
      
    # Calculate payoff for each node 
    for i in tqdm(np.arange(T_node-1,-1,-1)):
        if method == 'IFDM' or method =='CN':
           a = np.zeros(S_node); b = np.zeros(S_node); c = np.zeros(S_node); d = np.zeros(S_node)
           a[-1]=0 ; b[0] = 1 ; b[-1]=1 ; c[0]=0 ; d[0] = 0 ;
        else: pass   
                   
        for j in np.arange(1,S_node-1):
            
            if method == 'EFDM':
                a = (0.5*(sigma**2)*(j**2)+0.5*(r-div)*j)*d_t
                b = 1-r*d_t -(sigma**2)*(j**2)*d_t
                c = (0.5*(sigma**2)*(j**2)-0.5*(r-div)*j)*d_t
                FDM_value[j,i] = a * FDM_value[j+1,i+1] + b*FDM_value[j,i+1]+ c * FDM_value[j-1,i+1]
            
            elif method == 'IFDM':
                 a[j] = 0.5*((r-div)*j-(sigma**2)*(j**2))*d_t
                 b[j] = 1+((sigma**2)*(j**2))*d_t+r*d_t
                 c[j] = 0.5*(-(r-div)*j-(sigma**2)*(j**2))*d_t
                 d[j] = FDM_value[j,i+1]
                   
            else:
                 a[j] = 0.25*((sigma**2)*(j**2)-(r-div)*j)
                 b[j] = -0.5*((sigma**2)*(j**2)+r+2/d_t)
                 c[j] = 0.25*((sigma**2)*(j**2)+(r-div)*j)
                 d[j] = -0.25*((sigma**2)*(j**2)-(r-div)*j)*FDM_value[j-1,i+1] \
                        +0.5*((sigma**2)*(j**2)+r-(2/d_t))*FDM_value[j,i+1] \
                        -0.25*((sigma**2)*(j**2)+(r-div)*j)*FDM_value[j+1,i+1]
                              
        if method != 'EFDM':        
            d[-1] = FDM_value[-1,i] 
                   
            FDM_value[:,i] = TDMAsolver(a[1:],b,c[:-1],d)      
                
            x = np.matrix(np.diag(b)+np.diag(a[1:],k=-1)+np.diag(c[:-1],k=1))
            FDM_LU_value[:,i] = np.linalg.solve(x,d)
        else: pass    
           
        if i in cnp_node[1:-1]:
            
            if method == 'EFDM':
               FDM_value[barrier_node:,i] += FV*cpn_rate
               FDM_value[s0_node:,i] = FV*(1+cpn_rate)
                 
            else:
               FDM_LU_value[barrier_node:,i] += FV*cpn_rate
               FDM_LU_value[s0_node:,i] = FV*(1+cpn_rate)
               FDM_value[barrier_node:,i] += FV*cpn_rate
               FDM_value[s0_node:,i] = FV*(1+cpn_rate)
               
    if print_result == True: 
       print('{0} price is'.format(method),round(FDM_value[int(S_node/2),0],2))
    
    return FDM_value , round(FDM_value[int(S_node/2),0],2)  

#%% Sensitivity Analysis
    
real_price = 958.90
T_range = np.arange(100,5000,200)
error = []
for time_node in tqdm(T_range):  
    
    est_price = FDM_pricing(s0,r,div,sigma,Barrier_level,cpn_rate,FV,upper_ratio,S_node,time_node,T,issue_date,cpn_date,'CN')[1]
    diff = est_price - real_price
    error.append(diff)

#%%
plt.scatter(T_range,error,label='Errors')
plt.plot(T_range,error)
plt.xlabel('Time Nodes')
plt.ylabel('Errors(FDM price - Real price)')
plt.title('Errors wrt different Time nodes')
plt.legend(loc='best')

#%%
S_range = np.arange(100,2000,200)
error2 = []
for stock_node in tqdm(S_range):  
    
    est_price2 = FDM_pricing(s0,r,div,sigma,Barrier_level,cpn_rate,FV,upper_ratio,stock_node,T_node,T,issue_date,cpn_date,'CN')[1]
    diff2 = est_price2 - real_price
    error2.append(diff2)

plt.scatter(S_range,error2,label='Errors')
plt.plot(S_range,error2)
plt.xlabel('Stock Nodes')
plt.ylabel('Errors(FDM price - Real price)')
plt.title('Errors wrt different Stock nodes')
plt.legend(loc='best')
    

#%%
result = FDM_pricing(s0,r,div,sigma,Barrier_level,cpn_rate,FV,upper_ratio,S_node,T_node,T,issue_date,cpn_date,'CN')

price = result[1]
value_matrix = result[0]

#%%
# Computational domain
s = np.linspace(0 , s0*upper_ratio , S_node)
t = np.linspace(0,T, T_node+1)

tnew,snew = np.meshgrid(t,s)
fig = plt.figure(1)
ax = fig.gca(projection='3d')
surface = ax.plot_surface(tnew,snew,value_matrix,cmap = cm.coolwarm)
plt.xlabel('Time nodes(0 to Maturity)')
plt.ylabel('Stock nodes')
plt.title('FDM price surface')
plt.show()