#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Hikari Software
# Y-Enterprise

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import solve
from scipy.optimize import root, fsolve
from sklearn.linear_model import LinearRegression
from statsmodels.formula.api import ols

EU_COUNTRY = ['Austria','Belgium','Bulgaria','Croatia','Cyprus','Czech Republic','Denmark','Estonia', 'Finland','France', 'Germany','Greece','Hungary','Ireland','Italy','Latvia', 'Lithuania','Luxembourg', 'Malta','Netherlands','Poland','Portugal', 'Romania','Slovenia','Slovak Republic','Spain', 'Sweden','United Kingdom']

def main():
    with open('result/info.json','r') as info:
        d=json.loads(info.read())
        year_pool = [2008,2015]
        country_pool = d['country']
    # print(year_pool)
    length = len(country_pool)
    # print('year',year_pool)
    # print('country',country_pool)

    sigma = 0.36

    delta_mat = []
    for i in range(length):
        delta_sub = []
        for j in range(length):
            if country_pool[i] in EU_COUNTRY and country_pool[j] in EU_COUNTRY:
                delta_sub.append(1)
            else :
                delta_sub.append(0)
        delta_mat.append(delta_sub)

    delta_mat = np.array(delta_mat)
    # print(delta_mat)
    # print(type(delta_mat))

   
    distance = pd.read_csv("distance.csv",index_col='Country')
    del distance['Capital']
    distance_mat = []
    for i in country_pool:
        distance_sub = []
        for j in country_pool:
            if '/' in distance.loc[i,j]:
                distance.loc[i,j] = 0.0
            distance_sub.append(float(distance.loc[i,j]))
        distance_mat.append(distance_sub)
    distance_mat = np.array(distance_mat)

    gdp = pd.read_csv('gdp.csv',encoding='ISO-8859-1',index_col='Country')
    gdp_dict_list = []
    for year in year_pool:
        gdp_dict_sub = []
        for country in country_pool:
            try :
                gstr = str(gdp.loc[country,str(year)])
                gstr = gstr.replace(',','')
                gdp_dict_sub.append(float(gstr))
               
            except Exception as e: 
                gdp_dict_sub.append(np.nan)
        gdp_dict_list.append(np.array(gdp_dict_sub))
    gdp_dict = {k:v for (k,v) in zip(year_pool,gdp_dict_list)}
    # gdp_dict: 是一个array
    
    
    
    # 2维全部变为1维，1维拉长到等长度
    
    for year in year_pool:
        with open("result/%d.csv" % year,'r') as d:
            trade_mat = []
            for line in d:
                numbers = line.strip().split(',')
                trade_mat_sub = [float(i) for i in numbers]
                trade_mat.append(trade_mat_sub)

        lnzij = []
        for i in range(length):
            for j in range(length):
                if i==j:
                    continue
                if trade_mat[i][j] is np.nan:
                    trade_mat[i][j] = 50000
                lnzij.append(np.log(float(trade_mat[i][j]/(gdp_dict[year][i] * gdp_dict[year][j]))))

        lnzij = np.array(lnzij)

        full_lnzij_mat = []
        for i in range(length):
            for j in range(length):
                if i==j:
                    full_lnzij_mat.append(0.0)
                    continue
                if trade_mat[i][j] is np.nan:
                    trade_mat[i][j] = 50000
                full_lnzij_mat.append(np.log(float(trade_mat[i][j]/(gdp_dict[year][i] * gdp_dict[year][j]))))
        full_lnzij_mat = np.array(full_lnzij_mat).reshape([length,length])

        lndij = []
        for i in range(length):
            for j in range(length):
                if i==j:
                    continue
                lndij.append(np.log(distance_mat[i][j]))
        lndij = np.array(lndij)

        # Processing Dummy
        notBothEU = []
        for i in range(length):
            for j in range(length):
                if i==j:
                    continue
                notBothEU.append(1-delta_mat[i][j])
        notBothEU = np.array(notBothEU)

        y_sum = np.sum(gdp_dict[year])
        
        # ----------------以上：数据处理--------------以下：重新线性回归----------------
        
        loop_max = 1
        epsilon = 1e-3
        esti_list = np.array([15.35078072,-1.4224712,-0.93298144]) # k a1 a2

        k=esti_list[0]
        a1=esti_list[1]
        a2=esti_list[2]
        full_delta_mat = 1-delta_mat
        x = (np.exp(a1*full_lnzij_mat+a2*full_delta_mat).T * (gdp_dict[year]/y_sum)).T
        # print(x)
        p = fsolve(solve.gen(x), np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])*2)
        p = np.log(p)
        # 此时的p: -ln(P^(1-s))
        pi = list(p)*len(p)
        pj = list(np.array(list(p)*len(p)).reshape([len(p), len(p)]).T.reshape([len(p)*len(p), ]))
        for i in range(len(p)):
            del pi[i*len(p)]
            del pj[i*len(p)]
        pi = np.array(pi)
        pj = np.array(pj)
        # print("Value of P:")
        # 被回归量：lnzij
        # 回归元：Pi Pj lndij delta
        data_reg = pd.DataFrame(data=dict(price=(pi+pj)[:200],lndij=lndij[:200],delta=notBothEU[:200],lnzij=lnzij[:200],pricedlnzij=lnzij[:200]+(pi+pj)[:200]))
        # sns.pairplot(data_reg, x_vars=['price', 'lndij','delta'],
                     # y_vars='lnzij', size=200, aspect=0.8, kind='reg')
        plt.scatter(data_reg['lndij'],data_reg['lnzij'])
        plt.xlabel(r'$ln(d_{ij})$')
        plt.ylabel(r'$ln(z_{ij})$')
        plt.legend(['Fraction of International Trade VS Distance'],fontsize='xx-small')

        linreg = LinearRegression()  
        model=linreg.fit(data_reg[['lndij','delta']], data_reg['lnzij']+data_reg['price'])  
        print(model)
        print(linreg.intercept_)
        print(linreg.coef_)

        # plt.show()
        model2 = ols('pricedlnzij ~ lndij + delta',data_reg).fit()
        print(model2.summary())




    
    
if __name__ == "__main__":
    main()


