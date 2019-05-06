# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 14:14:34 2019

@author: rbgud
"""

from pandas.tseries.offsets import *
import pandas as pd
from datetime import date
#import datetime as dt
import numpy as np

from pandas.tseries.holiday import AbstractHolidayCalendar, Holiday, nearest_workday, \
    USMartinLutherKingJr, USPresidentsDay, GoodFriday, USMemorialDay, \
    USLaborDay, USThanksgivingDay


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
#%%
def business_dates(start, end):
    us_cal = USTradingCalendar()
    kw = dict(start=start, end=end)
    return pd.DatetimeIndex(freq='B', **kw).drop(us_cal.holidays(**kw))

bd_list_2019 = []
bdc = []
for j in range(6):
    if j==0:
        for i in range(11):
            if i+2 == 2:
               bd = len(business_dates(start=date(2019+j,1+i,28), end=date(2019+j,2+i,26)))
            else:
                 if  (i+2)%2 == 1:
                     bd = len(business_dates(start=date(2019+j,1+i,27), end=date(2019+j,2+i,26)))
                 else:
                     bd = len(business_dates(start=date(2019+j,1+i,27), end=date(2019+j,2+i,26))) 
            
            bd_list_2019.append(bd)
            
    else:
        for i in range(12):
            if i+1 == 1 :
               bdc_ = len(business_dates(start=date(2019+j-1,12,27), end=date(2019+j,1+i,26)))
            else:
                 if  (i+1)%2 == 1:
                     bdc_ = len(business_dates(start=date(2019+j,i,27), end=date(2019+j,1+i,26)))
                 else:
                     bdc_ = len(business_dates(start=date(2019+j,i,27), end=date(2019+j,1+i,26))) 
                  
            bdc.append(bdc_)        
#%%
bd2019 = pd.DataFrame()
for i in range(len(bd_list_2019)):        
    bd2019 = pd.concat([bd2019,pd.DataFrame(bd_list_2019[i])],axis=0)

bd2019 = bd2019.set_index(bd2019.values)

bdremain = pd.DataFrame()
for i in range(len(bdc)):        
    bdremain = pd.concat([bdremain,pd.DataFrame(bdc[i])],axis=0)

bdremain = bdremain.set_index(bdremain.values)   
bd_total = pd.concat([bd2019,bdremain],axis=0)
bd_total.index = pd.to_datetime(bd_total[0],format='%Y-%m-%d')
bd_total_new = pd.DataFrame(bd_total.index[bd_total.index < '2024-02-01'])
bd_total_new.index = pd.to_datetime(bd_total_new[0],format='%Y-%m-%d')
bd_total_new[0] = np.arange(len(bd_total_new))
(bd_total_new.ix['2019-02-01']-bd_total_new.ix['2019-01-28'])

