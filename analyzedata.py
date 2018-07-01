#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Hikari Software
# Y-Enterprise

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import json

def enum_two(l,full=True):
    length = len(l)
    if length < 2:
        return
    if not full:
        for i in range(length-1):
            for j in range(i+1,length):
                yield [l[i],l[j]]
    else :
        for i in range(length):
            for j in range(length):
                yield [l[i],l[j]]
        
            

def main():
    # Defining aliases for column names
    COUNTRY = 'Country Name'
    CODE = 'Country Code'
    COUNTERPART = 'Counterpart Country Name'
    COUNTERCODE='Counterpart Country Code'
    TIME='Time Period'
    EXPORT = 'Goods, Value of Exports, Free on board (FOB), US Dollars (TXG_FOB_USD)'
    STATUS='Status' # DEL
    STATUS1='Status.1' # DEL
    IMPORTCIF='Goods, Value of Imports, Cost, Insurance, Freight (CIF), US Dollars (TMG_CIF_USD)'
    # del
    IMPORTFOB='Goods, Value of Imports, Free on board (FOB), US Dollars (TMG_FOB_USD)'
    # del
    STATUS2='Status.2' # DEL
    BALANCE = 'Goods, Value of Trade Balance, US Dollars (TBG_USD)'
    STATUS3='Status.3' # DEL
    UNNAME='Unnamed: 13' # DEL
    IMPORT='import'
    ISCIF='iscif'
      
    # Open the data file
    data = pd.read_csv("indata.csv")

    # Discarding useless columns
    del data[STATUS]
    del data[STATUS1]
    del data[STATUS2]
    del data[STATUS3]
    del data[UNNAME]
    data[[CODE,TIME,COUNTERCODE,EXPORT,IMPORTFOB,IMPORTCIF,BALANCE]] = data[[CODE,TIME,COUNTERCODE,EXPORT,IMPORTFOB,IMPORTCIF,BALANCE]].apply(pd.to_numeric)

    country_pool = ['Austria',
'Belgium',
'Bulgaria',
'Croatia',
'Czech Republic',
'Denmark',
'Estonia',
'Finland',
'France',
'Germany',
'Greece',
'Hungary',
'Ireland',
'Italy',
'Latvia',
'Lithuania',
'Netherlands',
'Poland',
'Portugal',
'Romania',
'Slovak Republic',
'Slovenia',
'Spain',
'Sweden',
'Moldova',
'Armenia, Republic of',
'Switzerland',
'Norway',
'Iceland',
'Ukraine',
'Albania',
'Georgia',
'United Kingdom',
'Turkey',
'Belarus',
'Bosnia and Herzegovina',
'Serbia, Republic of',
'Russian Federation']
    year_pool = [2000,2015,2008] 

    country_list = []
    country_data = []
    group_by_country = data.groupby(data[COUNTRY])
    for i in group_by_country:
        country_list.append(i[0])
        group_by_year = i[1].groupby(i[1][TIME])
        year_list = []
        year_data = []
        for j in group_by_year:
            year_list.append(j[0])
            year_data.append(j[1])
        year_dict = {k:v for (k,v) in zip(year_list,year_data)}
        country_data.append(year_dict)

    country_dict = {k:v for (k,v) in zip(country_list,country_data)}
    print(country_dict[country_pool[0]][2000])
    print(type((country_dict[country_pool[0]][2000])))

    info_dict = dict(year=year_pool, country=country_pool)

    if not os.path.exists("result"):
        os.mkdir("result")

    with open("result/info.json",'w') as w:
        w.write(json.dumps(info_dict))

    for year in year_pool:
        print('Doing Year:',year)
        year_panel = []
        for i in country_pool:
            print('Doing Country',i)
            countryi = []
            for j in country_pool:
                if i==j:
                    countryi.append(0.0)
                    continue
                for l in country_dict[i][year].index:
                    if country_dict[i][year].loc[l,COUNTERPART]==j:
                        countryi.append(country_dict[i][year].loc[l,IMPORTCIF] + country_dict[i][year].loc[l,EXPORT])
                        break
                else :
                    countryi.append(np.nan)
            year_panel.append(countryi)
        with open('result/%d.csv' % year,'w') as t:
            for line in year_panel:
                s = [str(item) for item in line]
                t.write(','.join(s) + '\n')

    # SETTING country pool
    # int



if __name__ == "__main__":
    main()


