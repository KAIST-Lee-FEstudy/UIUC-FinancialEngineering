# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 07:01:05 2019

@author: rbgud
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
from scipy.stats import norm
from tqdm import tqdm
#%% Brownian motion

def Brownian(seed, N):
    
    np.random.seed(seed)                         
    dt = 1./N                                         # time step
    b = np.random.normal(0., 1., int(N))*np.sqrt(dt)  # brownian increments
    W = np.cumsum(b)                                  # brownian path
    return W, b 

seed = 20      
N  = 2.**6     # increments

b = Brownian(seed, N)[1]

# brownian motion

W = Brownian(seed, N)[0]
W = np.insert(W, 0, 0.)                               # W_0 = 0. for brownian motion

# brownian increments
plt.figure(1)
plt.rcParams['figure.figsize'] = (10,8)
xb = np.linspace(1, len(b), len(b))
plt.plot(xb, b)
plt.title('Brownian Increments')

# brownian motion
plt.figure(2)
xw = np.linspace(1, len(W), len(W))
plt.plot(xw, W)
plt.title('Brownian Motion')
#%% Stock path simulation- Geometric Brownian motion

# Parameters
#
# So:     initial stock price
# mu:     returns (drift coefficient)
# sigma:  volatility (diffusion coefficient)
# W:      brownian motion
# T:      time period
# N:      number of increments

def GBM(So, r, sigma, seed, T, N):    
    t = np.linspace(0.,1.,N+1)
    S = []
    S.append(So)
    for i in range(1,int(N+1)):
        drift = (r - 0.5 * sigma**2) * t[i]
        diffusion = sigma * Brownian(seed, N)[0][i-1]
        S_temp = So*np.exp(drift + diffusion)
        S.append(S_temp)
    return S, t

So = 102.8
r = 0.0271
sigma = 0.2704
W = Brownian(seed, N)[0]
T = 1.
N = 2.**6
seed = 51

soln = GBM(So, r, sigma, seed, T, N)[0]    # Exact solution
t = GBM(So, r, sigma, seed, T, N)[1]       # time increments for  plotting

plt.plot(t, soln)
plt.ylabel('Stock Price, $')
plt.title('Geometric Brownian Motion')

def Simulation_GBM(So,r,sigma,T,N,NumSim,disp=None):
    
    save_path = pd.DataFrame()
    for seed in range(1,NumSim):
        if disp == True:
           t = GBM(So, r, sigma, seed, T, N)[1]       # time increments for  plotting
           plt.plot(t,GBM(So, r, sigma, seed, T, N)[0])
           plt.xlabel('Time period')
           plt.ylabel('Stock Price')
           plt.title('Stock Path Simulation')
           
        else:
           sample_path = pd.Series(GBM(So, r, sigma, seed, T, N)[0])
           save_path = pd.concat([save_path,sample_path],axis=1,ignore_index=True)
                     
    return save_path.T

#%% Montecarlo simulation(fastest)
    
So = 102.8
maturity= 1
r = 0.0271
sigma = 0.2704
Numsim = 10000
NumSteps = 50
T = 1
ann_div = 0.0184
#%%
def Simulate_Stockpaths(So,r,sigma,T,Numsim,NumSteps):
    delta_t =T/NumSteps
    z_matrix = np.random.standard_normal(size =(Numsim,(NumSteps)))
    st_matrix = np.zeros((Numsim,NumSteps))
    st_matrix[:,0] = So
    for i in range(NumSteps-1):
    
        st_matrix[:,i+1] = st_matrix[:,i]*np.exp((r-0.5*sigma**2)*delta_t + sigma*np.sqrt(delta_t)*z_matrix[:,i])

    return st_matrix
           
    

a= Simulate_Stockpaths(So,r,sigma,T,Numsim,NumSteps)   
     
#%%

%time paths = Simulation_GBM(So,r,sigma,T,N,10,disp=True) 
     
#%% Binomial path simulation

def Simulation_BinomialTree(So,r,sigma,T,N,tree_type=None, ann_div=None, divperiod=None, disp=None):
    df = pd.DataFrame(np.zeros((N+1,N+1)))
    df.loc[0,0] = So
    dt = T/N
       
    def showTree(tree):
        t = np.linspace(T/NumSteps, T, NumSteps+1)
        fig, ax = plt.subplots(1,1,figsize=(6,4))
        for i in range(len(t)):
            for j in range(i+1):
                ax.plot(t[i], tree[i][j], '.b')
                if i<len(t)-1:
                    ax.plot([t[i],t[i+1]], [tree[i][j], tree[i+1][j]], '-b')
                    ax.plot([t[i],t[i+1]], [tree[i][j], tree[i+1][j+1]], '-b')
        fig.show()
        
    """ q = h(d2)
    u = q'*(exp((r-div)*t))/q
    d = ((exp((r-div)*t))-q*u)/(1-q)

    d1 = (ln(S/K)+(r-div+0.5*sigma^2)(T-t))/sigma(T-t)
    d2 = (ln(S/K)+(r-div-0.5*sigma^2)(T-t))/sigma(T-t)
    
    h(x) = 0.5 + sign(x)*sqrt([0.25-0.25*exp(-(x/(N+1/3))^2*(N+1/6))])
    q' = h(d1)
    """    
        
    def Leisen_Reimer(So,k,sigma,r,ann_div,T,N):
        
        d1 = (np.log(s/k)+(r-ann_div+0.5*sigma**2)*T)/(sigma*np.sqrt(T))
        d2 = (np.log(s/k)+(r-ann_div-0.5*sigma**2)*T)/(sigma*np.sqrt(T))
        
        def h_func(d):
            
            if d < 0:
               
               h = 0.5 - np.sqrt(0.25-0.25*np.exp(-np.power((d/(N+(1/3))),2)*(N+(1/6))))
               
            else:
                
               h = 0.5 + np.sqrt(0.25-0.25*np.exp(-np.power((d/(N+(1/3))),2)*(N+(1/6))))
               
            return h
        
        q_ = h_func(d1)
        q = h_func(d2)
        u = q_*np.exp((r-ann_div)*T/N)/q
        d = (np.exp((r-ann_div)*T/N)-q*u)/(1-q)
        
        return q,u,d

    if tree_type == 'CRR':
        u  = np.exp(sigma*np.sqrt(T/N))
        d = 1/u
        
    elif tree_type == 'Binomial':
        u  = np.exp(r*T/N+sigma*np.sqrt(T/N))
        d = np.exp(r*T/N-sigma*np.sqrt(T/N))
        
    elif tree_type == 'Rendleman':
        u = np.exp((r-ann_div-0.5*sigma**2)*dt+sigma*np.sqrt(dt))
        d = np.exp((r-ann_div-0.5*sigma**2)*dt-sigma*np.sqrt(dt))
        
    elif tree_type =='LR':
        u = Leisen_Reimer(s,k,sigma,r,ann_div,T,N)[1]
        d = Leisen_Reimer(s,k,sigma,r,ann_div,T,N)[2]    
    
    
    if divperiod == 'M':
       div = ann_div/12 
       div_m = np.arange(1,13)
       div_time = np.floor(div_m*N/12)+1
       
       for j in range(1,N+1):
           for i in range(j+1):
               if i==0:
                   df.loc[i,j]=df.loc[i,j-1]*d
                   if j in div_time:
                      df.loc[i,j]=df.loc[i,j-1]*d*(1-div)
               else:
                   df.loc[i,j] = df.loc[i-1,j-1]*u
                   if j in div_time:
                      df.loc[i,j] = df.loc[i-1,j-1]*u*(1-div)

       
    elif divperiod == 'Q':
       div = ann_div/4
       div_q = np.arange(1,13,3)
       div_time = np.floor(div_q*N/12)+1
        
       for j in range(1,N+1):
           for i in range(j+1):
               if i==0:
                   df.loc[i,j]=df.loc[i,j-1]*d
                   if j in div_time:
                      df.loc[i,j]=df.loc[i,j-1]*d*(1-div)
               else:
                   df.loc[i,j] = df.loc[i-1,j-1]*u
                   if j in div_time:
                      df.loc[i,j] = df.loc[i-1,j-1]*u*(1-div)
                      
    elif divperiod== None:              
          for j in range(1,N+1):
              for i in range(j+1):
                  if i==0:
                       df.loc[i,j]=df.loc[i,j-1]*d
                  else:
                       df.loc[i,j] = df.loc[i-1,j-1]*u
                       
    if disp :
       showTree(df) 
                                   
    return df

#%%
def option_tree(So,k,r,T,sigma,N,option_type1,option_type2,tree_type,ann_div=None,disp=None):
    dt = T/N
    df = pd.DataFrame(np.zeros((N+1,N+1)))
    
    def showTree(tree):
        t = np.linspace(T/N, T, N+1)
        fig, ax = plt.subplots(1,1,figsize=(6,4))
        for i in range(len(t)):
            for j in range(i+1):
                ax.plot(t[i], tree[i][j], '.b')
                if i<len(t)-1:
                    ax.plot([t[i],t[i+1]], [tree[i][j], tree[i+1][j]], '-b')
                    ax.plot([t[i],t[i+1]], [tree[i][j], tree[i+1][j+1]], '-b')
        fig.show()
    
    def Leisen_Reimer(So,k,sigma,r,ann_div,T,N):
    
        d1 = (np.log(s/k)+(r-ann_div+0.5*sigma**2)*T)/(sigma*np.sqrt(T))
        d2 = (np.log(s/k)+(r-ann_div-0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    
        def h_func(d):
            
            if d < 0:
               
               h = 0.5 - np.sqrt(0.25-0.25*np.exp(-np.power((d/(N+(1/3))),2)*(N+(1/6))))
               
            else:
                
               h = 0.5 + np.sqrt(0.25-0.25*np.exp(-np.power((d/(N+(1/3))),2)*(N+(1/6)))) 
               
            return h
        
        q_ = h_func(d1)
        q = h_func(d2)
        u = q_*np.exp((r-ann_div)*T/N)/q
        d = (np.exp((r-ann_div)*T/N)-q*u)/(1-q)
        
        return q,u,d
    
    if tree_type == 'CRR':
        u  = np.exp(sigma*np.sqrt(T/N))
        d = 1/u
    elif tree_type == 'Binomial':
        u  = np.exp(r*T/N+sigma*np.sqrt(T/N))
        d = np.exp(r*T/N-sigma*np.sqrt(T/N))
    elif tree_type == 'Rendleman':
        u = np.exp((r-ann_div-0.5*sigma**2)*dt+sigma*np.sqrt(dt))
        d = np.exp((r-ann_div-0.5*sigma**2)*dt-sigma*np.sqrt(dt))
    elif tree_type =='LR':
        u = Leisen_Reimer(s,k,sigma,r,ann_div,T,N)[1]
        d = Leisen_Reimer(s,k,sigma,r,ann_div,T,N)[2]

    p = (np.exp(r*dt)-d)/(u-d)
    
    if option_type1=='call':    
        sign = 1
    else:
        sign = -1
    
    if option_type2 == 'European':
        v = np.zeros(N+1)
        vv =np.zeros(N+1)
        for i in range(N+1):
            s1 = s* (u**i)*(d**(N-i))
            v[i] = k-s1
            if v[i] <=0:
                v[i] = 0
        
        for j in range(1,N+1):
            for i in range(N+1-j):
                vv[i] = np.exp(-r*dt)*(p*v[i+1]+(1-p)*v[i])
            for i in range(N+1):
                v[i] = vv[i]
    
    elif option_type2=='American':
        v = np.zeros(N+1)
        vv =np.zeros(N+1)
        S =np.zeros(N+1)
        for i in range(N+1):
            s1 = s* (u**i)*(d**(N-i))
            v[i] = k-s1
            if v[i] <=0:
                v[i] = 0
             
        
        for j in range(1,N+1):
            
            for i in range(N+1-j):
                vv[i] = np.exp(-r*dt)*(p*v[i+1]+(1-p)*v[i])
                s1 = So*(u**i)*(d**(N-j-i))
                if vv[i] < k-s1:
                    vv[i] = k-s1
                    S[j] = s1
            for i in range(N+1):
                v[i] = vv[i]
                    
    if disp :
        showTree(df)
        
    return v[0]

#%%
def cpn_period(cpn_type=None):
                
    if cpn_type == 'M':
        T = np.arange(1,13,1)
        term = 12
        period = T/term
        
    elif cpn_type == 'Q':
        T = np.arange(1,13,3)
        term = 12
        period = T/term
        
    elif cpn_type == 'Y':
        T = 12
        term = 12
        period = T/term
                      
    return period   

cpn_time = cpn_period('M')
time_period= np.arange(0,1.01,0.02)
#%%
#Parameter
s = 102.8
k=90.46
sigma = 0.2704
r = 0.0271
q = 0.0184
T= 1
N = 50
q_t= np.array([1,4,7,10])
cpn_time = cpn_period('M')
FV = 1000
ratio = FV/k
coupon_rate = 0.0905
autocall_period = np.array([3,6,9])/12

stock_path = Simulation_BinomialTree(s,r,sigma,T,N=50,tree_type='CRR', ann_div=0.0184, divperiod='Q', disp=None)
#%%
def Autocallable_bond(stock_path,FV,ratio,coupon_rate,T,N,sigma,cpn_time,autocall_period):
    dt = T/N
    u  = np.exp(r*T/N+sigma*np.sqrt(T/N))
    d = np.exp(r*T/N-sigma*np.sqrt(T/N))
    p = (np.exp(r*dt)-d)/(u-d)
    df=pd.DataFrame(np.zeros((N+1,N+1)))
    cpn = coupon_rate * FV / np.size(cpn_time)
    for j in np.arange(N,-1,-1):
        if j==N:
            cpn_time = cpn_time[:-1]     
        for i in range(j+1):
            if j==N:
                if stock_path.loc[i,j] < k:
                     df.loc[i,j]=ratio*stock_path.loc[i,j] + cpn
                else:
                    df.loc[i,j]= FV + cpn
                    
            else:
                if j*dt < cpn_time[-1]:
                    if j*dt < autocall_period[-1]:
                        if stock_path.loc[i,j]*np.exp(r*(autocall_period[-1]-j*dt)) < stock_path.loc[0,0]:
                            df.loc[i,j] = np.exp(-r*dt)*(p*df.loc[i+1,j+1]+(1-p)*df.loc[i,j+1]) + cpn*np.exp(-r*(autocall_period[-1]-j*dt))
                        else:
                            df.loc[i,j]= (FV+cpn)*np.exp(-r*(autocall_period[-1]-j*dt))
                        
                    else:
                        df.loc[i,j] = np.exp(-r*dt)*(p*df.loc[i+1,j+1]+(1-p)*df.loc[i,j+1]) + cpn*np.exp(-r*(cpn_time[-1]-j*dt))
                else:
                     df.loc[i,j] = np.exp(-r*dt)*(p*df.loc[i+1,j+1]+(1-p)*df.loc[i,j+1])
                     
        if j*dt <cpn_time[-1]:
            cpn_time = cpn_time[:-1]
            
        if np.size(cpn_time)==0:
            cpn_time = np.array([0])
            
        if j*dt <autocall_period[-1]:
            autocall_period = autocall_period[:-1]
            
        if np.size(autocall_period)==0:
            autocall_period = np.array([0])
    return df

x =  Autocallable_bond(stock_path,FV,ratio,coupon_rate,T,N,sigma,cpn_time,autocall_period)

#%%
#BLACK SHOLES PRICE
def d1(s,k,r,q,T,sigma):
    return (np.log(s/k) + (r-q+0.5*sigma**2)*T)/(sigma*np.sqrt(T))

def d2(s,k,r,q,T,sigma):
    return (np.log(s/k) + (r-q-0.5*sigma**2)*T)/(sigma*np.sqrt(T))

def bs_price(s,k,r,q,T,sigma,option_type):
    if option_type == 'call':
        x = 1;
    if option_type == 'put':
        x = -1;
    d_1 = d1(s,k,r,q,T,sigma)
    d_2 = d2(s,k,r,q,T,sigma)
    option_price = x * s * np.exp(-q*T) * norm.cdf(x*d_1) -x*k*np.exp(-r*T) *norm.cdf(x*d_2);
    return option_price;

#%%
s= 100;
k = 105;
r = 0.1;
q = 0;
sigma = 0.3;
T = 0.2
option_type = 'put'
option_type2 = 'American'
N= 100
    
a_1_European_put = bs_price(s,k,r,q,T,sigma,option_type)

crr = option_tree(s,k,r,T,sigma,N,option_type,option_type2,'CRR',q,disp=None)
Rendleman = option_tree(s,k,r,T,sigma,N,option_type,option_type2,'Rendleman',q,disp=None)
Lr = option_tree(s,k,r,T,sigma,N,option_type,option_type2,'LR',q,disp=None)

#%%
crr_result = []
RB_result = []

start_time = time.time() 
for i in tqdm(range(50,1001)):

    crr = option_tree(s,k,r,T,sigma,i,option_type,option_type2,'CRR',q,disp=None)
    rb = option_tree(s,k,r,T,sigma,i,option_type,option_type2,'Rendleman',q,disp=None)
    
    crr_result.append(crr)
    RB_result.append(rb)
      
print("Start LR simulation")
#%%
LR_result = []

start_time = time.time() 
for i in tqdm(np.arange(51,1000,2)):
   
    lr = option_tree(s,k,r,T,sigma,i,option_type,option_type2,'LR',q,disp=None)
       
    LR_result.append(lr)
        
print("--- %s seconds ---" %(time.time() - start_time)) 

#%% 
American_exact = np.float64(7.36308)
Error_CRR = crr_result - American_exact
Error_RB = RB_result - American_exact
Error_LR = LR_result - American_exact

x_axis = np.arange(50,1001)
plt.figure(1)
plt.scatter(x_axis,Error_CRR,label='Error of Cox,Ross & Rubinstein')
plt.xlabel('Number of Steps')
plt.ylabel('Error')
plt.legend(loc='best')
plt.title('Error of CRR - BS model')
plt.show()

plt.figure(2)
plt.scatter(x_axis,Error_RB,label='Error of Renlement & Bartter')
plt.xlabel('Number of Steps')
plt.ylabel('Error')
plt.legend(loc='best')
plt.title('Error of RB - BS model')
plt.show()

x_axis_LR = np.arange(51,1000,2)
plt.figure(3)
plt.scatter(x_axis_LR,Error_LR,label='Error of Leisen and Reimer')
plt.xlabel('Number of Steps')
plt.ylabel('Error')
plt.legend(loc='best')
plt.title('Error of LR - BS model')
plt.show()
#%%
crr_result_ = pd.DataFrame(crr_result)
RB_result_ = pd.DataFrame(RB_result)
LR_result_ = pd.DataFrame(LR_result)
def BD_method(s,k,r,sigma,div,T,N):
    dt = T/N
    u  = np.exp(sigma*np.sqrt(dt))
    d = 1/u
    p = (np.exp((r-div)*dt)-d)/(u-d)
    
    v=np.zeros(N+1)
    vv = np.zeros(N+1)
    
    for i in range(N):
        s1 = s*(u**i)*(d**(N-1-i))
        d1_ = (np.log(s1/k)+((r-div+0.5*sigma**2)*dt))/(sigma * np.sqrt(dt))
        d2_ = (np.log(s1/k)+((r-div-0.5*sigma**2)*dt))/(sigma * np.sqrt(dt))
        v[i] = max(-s1 * np.exp(-div*dt)*norm.cdf(-d1_) + k*np.exp(-r*dt)*norm.cdf(-d2_),k-s1)
    
    for j in range(2,N+1):
        for i in range(N+1-j):
            vv[i] = np.exp(-r*dt)*(p*v[i+1]+(1-p)*v[i])
            s1 = s * (u**i) * (d**(N-j-i))
            vv[i] = max(vv[i],(k-s1))
        for i in range(N+1):
            v[i] = vv[i]
            
    return np.round(v[0],6)

BD_result = []

start_time = time.time() 
for i in tqdm(range(50,1001):
    
    bd = BD_method(s,k,r,sigma,q,T,i)
       
    BD_result.append(bd)
        
print("--- %s seconds ---" %(time.time() - start_time)) 

Error_BD = BD_result - American_exact
plt.figure(4)
plt.scatter(x_axis,Error_BD,label='Error of Broadie and Detemple')
plt.xlabel('Number of Steps')
plt.ylabel('Error')
plt.legend(loc='best')
plt.title('Error of BD - BS model')
plt.show()

#%%

dt = T/N
u  = np.exp(sigma*np.sqrt(dt))
d = 1/u
p = (np.exp((r-div)*dt)-d)/(u-d)
v=np.zeros(N+1)
vv = np.zeros(N+1)
exercise_s = pd.DataFrame(np.zeros((N+1,N+1)))
for i in range(N):
    s1 = s*(u**i)*(d**(N-1-i))
    d1_ = (np.log(s1/k)+((r-div+0.5*sigma**2)*dt))/(sigma * np.sqrt(dt))
    d2_ = (np.log(s1/k)+((r-div-0.5*sigma**2)*dt))/(sigma * np.sqrt(dt))
    v[i] = max(-s1 * np.exp(-div*dt)*norm.cdf(-d1_) + k*np.exp(-r*dt)*norm.cdf(-d2_),k-s1)
    
for j in range(1,N+1):
    for i in range(N+1-j):
        vv[i] = np.exp(-r*dt)*(p*v[i+1]+(1-p)*v[i])
        s1 = s * (u**i) * (d**(N-j-i))
        if k-s1 > vv[i]:
            exercise_s.loc[i,j] = s1
        vv[i] = max(vv[i],(k-s1))
    for i in range(N+1):
        v[i] = vv[i]
        
exercise_s = exercise_s.replace(0,np.nan)
boundary =exercise_s.max(axis=0)
boundary_new = boundary.sort_index(ascending=False)
boundary_new = pd.DataFrame(boundary_new.values)
time_index = np.arange(0,1.01,0.01)
plt.figure(5)
plt.plot(time_index,boundary_new,label='Boundary')
plt.xlabel('Time Step')
plt.ylabel('Stock price')
plt.legend(loc='best')
plt.title('Put option early excersize boundary')

#%%
s = 100
k = 100
b=95
r=0.1
q = 0
sigma = 0.3
T =0.2
N=50
Exact_DOprice = bs_price(s,k,r,q,T,sigma,'call')-(s/b)**(1-2*r/(sigma**2))*bs_price((b**2)/s,k,r,q,T,sigma,'call')
            
#%% 
"""In-out parity : In + Out = Vanila"""

def Down_out_call(s,r,b,k,sigma,N,T,option_type,tree_type,Barrier_number=None):
        
    dt = T/N
    v = np.zeros(N+1)
    vv =np.zeros(N+1)
    if Barrier_number != None:
        dn = int(N/(Barrier_number+1))
        Barrier_time = np.arange(1,Barrier_number+1)
        barrier_node = dn*Barrier_time
    
    if tree_type == 'CRR':
        u  = np.exp(sigma*np.sqrt(T/N))
        d = 1/u
    elif tree_type == 'Binomial':
        u  = np.exp(r*T/N+sigma*np.sqrt(T/N))
        d = np.exp(r*T/N-sigma*np.sqrt(T/N))
    elif tree_type == 'Rendleman':
        u = np.exp((r-ann_div-0.5*sigma**2)*dt+sigma*np.sqrt(dt))
        d = np.exp((r-ann_div-0.5*sigma**2)*dt-sigma*np.sqrt(dt))
    elif tree_type =='LR':
        u = Leisen_Reimer(s,k,sigma,r,ann_div,T,N)[1]
        d = Leisen_Reimer(s,k,sigma,r,ann_div,T,N)[2]
        
    if option_type =='call':
        sign = 1
    else:
        sign = -1
    
    p = (np.exp(r*dt)-d)/(u-d)
    stop = 0
    for i in range(N+1):
        s1 = s * (u**i) * (d**(N - i))
        v[i] = max(sign*(s1-k),0)
        if ((s1 > b) & (stop == 0)):
             S_up = s1
             S_down = s * (u**(i - 1)) * (d**(N - (i - 1)))
             lambda_ = (S_up - b) / (S_up - S_down)
             stop = 1
    
    for j in range(1,N+1):
            
        for i in range(N+1-j):
            vv[i] = np.exp(-r*dt)*(p*v[i+1]+(1-p)*v[i])
            s1 = s*(u**i)*(d**(N-j-i))
            if Barrier_number!= None and j in barrier_node and s1 < b:
               vv[i]=0
            elif Barrier_number == None and s1 < b:
                vv[i] = 0   
        for i in range(N+1):
            v[i] = vv[i]
                
    return v[0], lambda_ 
#%%
#Do_value = Down_out_call(s,r,b,k,sigma,N,T,'call','CRR',None)[0]
Do_value = Down_out_call(s,r,b,k,sigma,N,T,'call','CRR',4)[0]
Lambda_value = Down_out_call(s,r,b,k,sigma,N,T,'call','CRR',4)[1]
#%% DO call (continuous barrier)
Do_result = []
Lambda_result = []

for i in tqdm(np.arange(50,1001)):
    
    do = Down_out_call(s,r,b,k,sigma,i,T,'CRR')[0]
    lam = Down_out_call(s,r,b,k,sigma,i,T,'CRR')[1]
    Do_result.append(do)
    Lambda_result.append(lam)
    
Do_result = pd.DataFrame(Do_result)
DO_error = Do_result-Exact_DOprice
Lambda_result = pd.DataFrame(Lambda_result)
#%%
plt.figure(6)
plt.plot(np.arange(50,1001),DO_error)
plt.xlabel('Number of Steps')
plt.ylabel('Error')
plt.title('Down and out call error')
plt.show()

plt.figure(7)
plt.scatter(np.arange(50,1001),Lambda_result)
plt.xlabel('Number of Steps')
plt.ylabel('Lambda')
plt.title('Down and out call lambda')
plt.show()

fig, ax1 = plt.subplots()
t = np.arange(50,1001)
color = 'tab:red'
ax1.set_xlabel('Number of Steps')
ax1.set_ylabel('Error', color=color)
ax1.plot(t, DO_error, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('Lambda', color=color)  # we already handled the x-label with ax1
ax2.scatter(t, Lambda_result, color=color,s=0.5)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()
#%% DO call (discrete barrier)

Do_disc_result = []
Lambda_disc_result = []

for i in tqdm(np.arange(50,1001)):
    
    do_ = Down_out_call(s,r,b,k,sigma,i,T,'call','CRR',4)[0]
    lam_ = Down_out_call(s,r,b,k,sigma,i,T,'call','CRR',4)[1]
    Do_disc_result.append(do_)
    Lambda_disc_result.append(lam_)
    
Do_disc_result = pd.DataFrame(Do_disc_result)
Do_disc_error = Do_disc_result-Exact_DOprice
Lambda_disc_result = pd.DataFrame(Lambda_disc_result)