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

[r,y,s0,Barrier_level,cpn_rate,FV,upper_ratio,S_node,T_node,T]=\
[0.024,0.03,145.5,250,0.025,100,2,100,260,5]
issue_date = dt.date(2018,10,1)
cpn_date = [dt.date(2019,4,1),dt.date(2019,10,1),dt.date(2020,4,1),dt.date(2020,10,1),dt.date(2021,4,1),\
            dt.date(2021,10,1),dt.date(2022,4,1),dt.date(2022,10,1),dt.date(2023,4,1),dt.date(2023,10,1)]
d_t = T/T_node
convert_ratio = 0.59735 #conversion ratio
call_date= dt.date(2021,10,1)
call_node = int(((call_date-issue_date).days/365)/d_t)
method = 'IFDM'
indexing_cpn = -2
acc_list = []
#%%
def FDM_pricing(r,y,Barrier_level,cpn_rate,FV,upper_ratio,S_node,T_node,T,issue_date,cpn_date,method,print_result=False):

    indexing_cpn = -2
    Barrier = Barrier_level
    lower = 0
    upper = s0*upper_ratio
    ds = (upper-lower)/S_node
    d_t = T/T_node
#    s_range = np.linspace(lower,upper,S_node+1)
    barrier_node = int(Barrier/ds)
#    s0_node = int(S_node/2)
    
    def TDMAsolver(a, b, c, d):
 
        nf = len(a)     # number of edivuations
        ac, bc, cc, dc = map(np.array, (a, b, c, d))     # copy the array
        for it in range(1, nf):
            mc = ac[it-1]/bc[it-1]
            bc[it] = bc[it] - mc*cc[it-1] 
            dc[it] = dc[it] - mc*dc[it-1]
    
        xc = bc
        xc[-1] = dc[-1]/bc[-1]
    
        for il in range(nf-2, -1, -1):
            xc[il] = (dc[il]-cc[il]*xc[il+1])/bc[il]
    
        return xc  
    
    # Change coupon payment date to node 
    cpn_node = []
    for i in range(len(cpn_date)):
        cpn_node.append(int(((cpn_date[i]-issue_date).days/365)/d_t))
        
    ## Boundary condition    
    FDM_value = np.zeros((S_node,T_node+1))
    
    # Terminal coundary condition
    for i in range(S_node):
        FDM_value[i,-1] = max(FV*(1+cpn_rate),convert_ratio*i*ds+FV*cpn_rate)
            
    # Lower boundary condition
    FDM_value[0,:]= 0
    # Upper boundary condition
    indexing = 0
    for i in range(T_node):
        FDM_value[-1,i] = FV*(1+cpn_rate) * np.exp(-(cpn_node[indexing] - i)*d_t)
        if i>= cpn_node[indexing]:
            indexing = indexing +1
    
    FDM_LU_value = FDM_value.copy()         
      
    # Calculate payoff for each node 
    for i in tqdm(np.arange(T_node-1,-1,-1)):
        FDM_value[0,:]= 0
        if method == 'IFDM' or method =='CN':
           a = np.zeros(S_node); b = np.zeros(S_node); c = np.zeros(S_node); d = np.zeros(S_node)
           a[-1]=0 ; b[0] = 1 ; b[-1]=1 ; c[0]=0 ; d[0] = 0 ;
        else: pass   
                   
        for j in np.arange(1,S_node-1):
            sigma = 8.4/np.sqrt(j*ds)
            if method == 'EFDM':
                a = (0.5*(sigma**2)*(j**2)+0.5*(r-y)*j)*d_t
                b = 1-r*d_t -(sigma**2)*(j**2)*d_t
                c = (0.5*(sigma**2)*(j**2)-0.5*(r-y)*j)*d_t
                FDM_value[j,i] = a * FDM_value[j+1,i+1] + b*FDM_value[j,i+1]+ c * FDM_value[j-1,i+1]
            
            elif method == 'IFDM':
                 a[j] = 0.5*((r-y)*j-(sigma**2)*(j**2))*d_t
                 b[j] = 1+((sigma**2)*(j**2))*d_t+r*d_t
                 c[j] = 0.5*(-(r-y)*j-(sigma**2)*(j**2))*d_t
                 d[j] = FDM_value[j,i+1]
                 
            else:
                 a[j] = 0.25*((sigma**2)*(j**2)-(r-y)*j)
                 b[j] = -0.5*((sigma**2)*(j**2)+r+2/d_t)
                 c[j] = 0.25*((sigma**2)*(j**2)+(r-y)*j)
                 d[j] = -0.25*((sigma**2)*(j**2)-(r-y)*j)*FDM_value[j-1,i+1] \
                        +0.5*((sigma**2)*(j**2)+r-(2/d_t))*FDM_value[j,i+1] \
                        -0.25*((sigma**2)*(j**2)+(r-y)*j)*FDM_value[j+1,i+1]    

        if method != 'EFDM':        
            d[-1] = FDM_value[-1,i] 
                   
            FDM_value[:,i] = TDMAsolver(a[1:],b,c[:-1],d)      
                
            x = np.matrix(np.diag(b)+np.diag(a[1:],k=-1)+np.diag(c[:-1],k=1))
            FDM_LU_value[:,i] = np.linalg.solve(x,d)
        else: pass    

        for k in range(S_node):

            if i > call_node:
                if k > barrier_node:
                    FDM_value[k,i] = min(max(FDM_value[k,i],convert_ratio*k*ds),\
                                 max(convert_ratio*k*ds,FV*(1+cpn_rate)))
                    
                    FDM_value[k,i] =  max(convert_ratio*k*ds,FDM_value[k,i])
                    
                else:
                    FDM_value[k,i] = max(convert_ratio*k*ds,FDM_value[k,i])
            else:
                 FDM_value[k,i] = max(convert_ratio*k*ds,FDM_value[k,i])
                 
        if i in cpn_node:
            FDM_value[:,i] = FDM_value[:,i] +FV*cpn_rate
            
        while i < cpn_node[indexing_cpn]:
            if cpn_node[indexing_cpn]==cpn_node[1]: break
            indexing_cpn = indexing_cpn -1
                
    if print_result == True: 
       print('{0} price is'.format(method),round(FDM_value[int(S_node/2),0],2))
       print('LU price is %4.3f',round(FDM_LU_value[int(S_node/2),0],2))
       
    return FDM_value, round(FDM_value[int(S_node/2),0],2) ,round(FDM_LU_value[int(S_node/2),0],2)

#%% Sensitivity Analysis
    
result = FDM_pricing(r,y,Barrier_level,cpn_rate,FV,upper_ratio,S_node,T_node,T,issue_date,cpn_date,'CN',print_result=False)

value_matrix = result[0]

#%%
# Computational domain
s = np.linspace(0 , s0*upper_ratio , S_node)
t = np.linspace(0,T, T_node+1)

tnew,snew = np.meshgrid(t,s)
fig = plt.figure(1)
ax = fig.gca(projection='3d')
surface = ax.plot_surface(tnew,snew,value_matrix,cmap = cm.coolwarm)
plt.xlabel('Time nodes (0 to Maturity)')
plt.ylabel('Stock nodes')
plt.title('FDM price surface')
plt.show()
