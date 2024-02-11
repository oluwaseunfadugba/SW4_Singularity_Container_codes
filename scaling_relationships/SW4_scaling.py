#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 13:34:24 2023

@author: oluwaseunfadugba
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math    
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

data =np.array([[28,  7246.42,  'no elem = 1.44E+7', '0.1', 1.44E+7], 
                [84,  2353.01,  'no elem = 1.44E+7', '0.1', 1.44E+7],
                [140, 1516.59,  'no elem = 1.44E+7', '0.1', 1.44E+7],
                [196, 1085.11,  'no elem = 1.44E+7', '0.1', 1.44E+7],
                [280, 809.64,   'no elem = 1.44E+7', '0.1', 1.44E+7],
                [28,  90493.01, 'no elem = 1.14E+8', '0.2', 1.14E+8],
                [84,  26825.37579203,'no elem = 1.14E+8', '0.2', 1.14E+8], # slurm-26287221.out 7 hours 27 minutes 5.37579203e+00 seconds
                #[84,  251105.91,'no elem = 1.14E+8', '0.2', 1.14E+8],
                [140, 17066.24, 'no elem = 1.14E+8', '0.2', 1.14E+8],
                [196, 11554.0022531, 'no elem = 1.14E+8', '0.2', 1.14E+8],# slurm-26287222.out 3 hours 12 minutes 3.40022531e+01 seconds
                #[196, 26793.06, 'no elem = 1.14E+8', '0.2', 1.14E+8],
                [280, 8267.61,  'no elem = 1.14E+8', '0.2', 1.14E+8],
                [84,  130239.57,'no elem = 3.83E+8', '0.3', 3.83E+8],
                [140, 71956.89, 'no elem = 3.83E+8', '0.3', 3.83E+8],
                [196, 54935.94, 'no elem = 3.83E+8', '0.3', 3.83E+8],
                [280, 36542.98, 'no elem = 3.83E+8', '0.3', 3.83E+8]
                ],dtype=object)

data_c =np.array([#[84, 2405.72, 'no elem = 1.44E+7', '0.1_c', 1.44E+7],
                  [28, 6819.6090605,  'no elem = 1.44E+7', '0.1_c', 1.44E+7], # 1 hour  53 minutes 3.96090605e+01 seconds
                  [84,  2423.6033895,  'no elem = 1.44E+7', '0.1_c', 1.44E+7], # 40 minutes 2.36033895e+01 seconds
                  [140, 1663.3597121,  'no elem = 1.44E+7', '0.1_c', 1.44E+7], # 27 minutes 4.33597121e+01 seconds
                  [196, 1327.20882797,  'no elem = 1.44E+7', '0.1_c', 1.44E+7], # 22 minutes 7.20882797e+00 seconds
                  [280, 1054.5065281,   'no elem = 1.44E+7', '0.1_c', 1.44E+7], # 17 minutes 3.45065281e+01 seconds
                  [28,  112776.7058458, 'no elem = 1.14E+8', '0.2_c', 1.14E+8], # 31 hours 19 minutes 3.67058458e+01 seconds
                  [84,  37434.450156,'no elem = 1.14E+8', '0.2_c', 1.14E+8], # 10 hours 23 minutes 5.44501560e+01 seconds
                  [140, 23563.7074928, 'no elem = 1.14E+8', '0.2_c', 1.14E+8], # 6 hours 32 minutes 4.37074928e+01 seconds
                  [196,  17922.9687543, 'no elem = 1.14E+8', '0.2_c', 1.14E+8], # 4 hours 58 minutes 4.29687543e+01  #21 hours 18 minutes 4.01890597e+01 seconds
                  [280, 13232.6864769,  'no elem = 1.14E+8', '0.2_c', 1.14E+8], # 3 hours 40 minutes 3.26864769e+01 seconds
                  [84,   188964.9021537,'no elem = 3.83E+8', '0.3_c', 3.83E+8], # 52 hours 29 minutes 2.49021537e+01 seconds
                  [140, 117388.7648811, 'no elem = 3.83E+8', '0.3_c', 3.83E+8], # 32 hours 36 minutes 2.87648811e+01 seconds
                  [196, 92373.0414758, 'no elem = 3.83E+8', '0.3_c', 3.83E+8], #25 hours 39 minutes 3.30414758e+01    50 hours 24 minutes 2.68490782e+01 seconds
                  [280, 63367.6520977, 'no elem = 3.83E+8', '0.3_c', 3.83E+8] # 17 hours 36 minutes 7.65209770e+00 seconds  
                ],dtype=object)

data_c3 =np.array([[28, 4646.26,  'no elem = 1.44E+7', '0.1_c3', 1.44E+7,   1,  16,  1.502e+01], # 
                  [84,  1504.961,  'no elem = 1.44E+7', '0.1_c3', 1.44E+7,   0,  24,  5.791e+01], # 
                  [140, 1065.02,  'no elem = 1.44E+7', '0.1_c3', 1.44E+7,   0,  17,  1.070e+01 ], #  
                  [196, 795.83,  'no elem = 1.44E+7', '0.1_c3', 1.44E+7,   0,  13,  3.776e+01 ], # 
                  [280, 632.54,   'no elem = 1.44E+7', '0.1_c3', 1.44E+7,   0,  10,  5.993e+01], # 
                  [28,  95279.49, 'no elem = 1.14E+8', '0.2_c3', 1.14E+8,   28,  9,  1.403e+01], # 
                  [84,  35678.96,'no elem = 1.14E+8', '0.2_c3', 1.14E+8,   10,  0,  2.826e+01], # 
                  [140, 22295.86, 'no elem = 1.14E+8', '0.2_c3', 1.14E+8,   6,  8,  5.259e+01], # 
                  [196,  16476.33, 'no elem = 1.14E+8', '0.2_c3', 1.14E+8,   4,  30,  3.585e+01], # 
                  [280, 11805.7,  'no elem = 1.14E+8', '0.2_c3', 1.14E+8,   3,  17,  1.300e+01], # 
                  [84, 204485.236,'no elem = 3.83E+8', '0.3_c3', 3.83E+8,   57,  11,  5.060e+01], # 
                  [140,  136063.19, 'no elem = 3.83E+8', '0.3_c3', 3.83E+8,   37,  9,  3.156e+01], # 
                  [196, 100773.26, 'no elem = 3.83E+8', '0.3_c3', 3.83E+8,   27,  59,  3.225e+01], #
                  [280, 69602.283, 'no elem = 3.83E+8', '0.3_c3', 3.83E+8,   20,  47,  4.139e+01] # 
                ],dtype=object)


data_c3[:,1] =  (data_c3[:,5]*3600)+(data_c3[:,6]*60)+data_c3[:,7]

#%% Plotting scaling relationship
# default parameters
# tcb10 = ['#006BA4', '#FF800E', '#ABABAB', '#595959', '#5F9ED1', '#C85200', 
#          '#898989', '#A2C8EC', '#FFBC79', '#CFCFCF']
tcb10 = ['#006BA4',  '#898989','#595959', '#FF800E','#5F9ED1',  '#ABABAB','#C85200',
         '#A2C8EC', '#CFCFCF', '#FFBC79', ]

markers = ['o', '^', 'D', 'P', '>', '-']
ncores = ['28', '84', '140', '196', '280']
freq = ['0.1', '0.2', '0.3']
freq_c = ['0.1_c', '0.2_c', '0.3_c']
freq_c3 = ['0.1_c3', '0.2_c3', '0.3_c3']
nelem = ['1.44E+7', '1.14E+8', '3.83E+8']

lw = 3
ls = ['-','--','-.','-','--','-.','-','--','-.','-','--']

s = 200
label_font = 100
fontsize=30
title_pad = 30
nbins = 15

# 
fig, axes = plt.subplots(1,2,figsize=(12, 7))
fig.tight_layout(h_pad=10,w_pad=10)#, w_pad=0.5, h_pad=7.0,rect=[0, 0.15, 0.8, 0.9]) #0, 0.03, 0.85, 0.90]
#axes=axes.flatten()

# Walltime vs no of cores
for i in range(len(freq)):
    n_elem=[]
    walltime=[]
    n_cores = data[data[:,3]==freq[i],0] 
    walltime = data[data[:,3]==freq[i],1]/3600 
    axes[0].scatter(n_cores,walltime,c=tcb10[i],marker = markers[i],s = s,label='no elem = '+ nelem[i])
    axes[0].plot(n_cores,walltime,lw=lw,ls = '--',c=tcb10[i])

for i in range(len(freq_c)):
    n_elem=[]
    walltime=[]
    n_cores = data_c[data_c[:,3]==freq_c[i],0] 
    walltime = data_c[data_c[:,3]==freq_c[i],1]/3600 
    axes[0].scatter(n_cores,walltime,linewidths = 2,c=tcb10[i],marker = markers[i],s = s,edgecolor ="red")#,label='no elem = '+ nelem[i])
    axes[0].plot(n_cores,walltime,lw=lw,ls = '-',c=tcb10[i])
    
for i in range(len(freq_c3)):
    n_elem=[]
    walltime=[]
    n_cores = data_c3[data_c3[:,3]==freq_c3[i],0] 
    walltime = data_c3[data_c3[:,3]==freq_c3[i],1]/3600 
    axes[0].scatter(n_cores,walltime,linewidths = 2,c=tcb10[i],marker = markers[i],s = s,edgecolor ="magenta")#,label='no elem = '+ nelem[i])
    axes[0].plot(n_cores,walltime,lw=lw,ls = '-',c=tcb10[i])
    
axes[0].set_xlim([10,1000])
axes[0].set_ylim([0.1,100])
axes[0].set_xlabel('No of Cores',fontsize=fontsize)
#axes[0].legend(loc='lower left',fontsize=fontsize/2)
axes[0].set_ylabel('Walltime (hours)',fontsize=fontsize)

[j.set_linewidth(2) for j in axes[0].spines.values()]
axes[0].grid(b=True, which='both',color = 'k', linestyle = markers[5], linewidth = 0.2)

axes[0].xaxis.set_tick_params(labelsize=fontsize)
axes[0].yaxis.set_tick_params(labelsize=fontsize)

axes[0].set_xscale("log");
axes[0].set_yscale("log");

axes[0].set_title('Scaling Relationship for SW4 v2.01 \n and v3.0 (Total time = 220 s)',fontsize=fontsize+5, y=1.05, x=0.95)#,fontdict={"weight": "bold"}) 
#pad =title_pad,

axes[1].axis('off')

# Creating legend
legend_elements = []

legend_elements.append(Line2D([],[],color=tcb10[0], marker=markers[0], markerfacecolor=tcb10[0],linewidth=0,markeredgecolor=tcb10[0],
                              markersize=s/10,alpha=1.0,label='no elem = '+ nelem[0]))

legend_elements.append(Line2D([0],[0],color=tcb10[0], marker=markers[1], markerfacecolor=tcb10[1],linewidth=0,markeredgecolor=tcb10[1],
                              markersize=s/10,alpha=1.0,label='no elem = '+ nelem[1]))

legend_elements.append(Line2D([0],[0],color=tcb10[0], marker=markers[2], markerfacecolor=tcb10[2],linewidth=0,markeredgecolor=tcb10[2],
                              markersize=s/10,alpha=1.0,label='no elem = '+ nelem[2]))

legend_elements.append(Line2D([],[],color=tcb10[0], marker=markers[0], markerfacecolor=tcb10[0],linestyle=ls[1], linewidth=2,
                              markersize=s/10,alpha=0))


legend_elements.append(Line2D([],[],color=tcb10[0], marker=markers[0], markerfacecolor=tcb10[0],linestyle=ls[1], linewidth=2,
                              markersize=s/10,alpha=1.0,label='SW4 on Talapas'))

legend_elements.append(Line2D([],[],color=tcb10[0], marker=markers[0], markerfacecolor=tcb10[0],linestyle=ls[0], linewidth=3,
                              markersize=s/10,alpha=1.0,label='SW4 Container (v2.01)',markeredgewidth=3,  markeredgecolor=(1, 0, 0, 0))) #,edgecolor ="red"

legend_elements.append(Line2D([],[],color=tcb10[0], marker=markers[0], markerfacecolor=tcb10[0],linestyle=ls[0], linewidth=3,
                              markersize=s/10,alpha=1.0,label='SW4 Container (v3.0)',markeredgewidth=3,  markeredgecolor="magenta")) #,edgecolor ="red" (1, 0, 0, 0)


plt.legend(handles=legend_elements, bbox_to_anchor=(-0.3, 0.7 ), loc='upper left', 
            fontsize=fontsize-5,frameon=False,labelspacing = 1)

plt.text(-0.25, 0.75, 'LEGEND', color='k', fontsize=fontsize)#,fontdict={"weight": "bold"})


plt.savefig('scaling_relationship.png',bbox_inches='tight', dpi=200)

#%% Determining regression for each subsets of the dataset
import numpy as np
from scipy.optimize import curve_fit

def func(x, a, b):
    exp = a + b*np.log10(x)
    #print(exp)
    return 10**exp

def def_reg(x,y):
    dp = 2
    
    popt, pcov = curve_fit(func, x, y)
    
    perr = np.round(np.sqrt(np.diag(pcov)),dp)
    param = np.round(popt,dp)
    
    return param,perr
    
    
data[:,1] = data[:,1]/3600 # converting walltime to hours
data_c[:,1] = data_c[:,1]/3600
data_c3[:,1] = data_c3[:,1]/3600

# data[:,4] = data[:,4]/1e9 # converting walltime to hours
# data_c[:,4] = data_c[:,4]/1e9
# data_c3[:,4] = data_c3[:,4]/1e9


# Scaling relationships
print()
for fmax in ['0.1','0.2','0.3']:
    n_cores = data[data[:,3]==fmax,0] 
    walltime = data[data[:,3]==fmax,1] 
    param,perr = def_reg(n_cores,walltime)
    print(fmax, 'Hz: ',param,perr)

print()
for fmax in ['0.1_c','0.2_c','0.3_c']:
    n_cores = data_c[data_c[:,3]==fmax,0] 
    walltime = data_c[data_c[:,3]==fmax,1] 
    param,perr = def_reg(n_cores,walltime)
    print(fmax, 'Hz: ',param,perr)

print()
for fmax in ['0.1_c3','0.2_c3','0.3_c3']:
    n_cores = data_c3[data_c3[:,3]==fmax,0] 
    walltime = data_c3[data_c3[:,3]==fmax,1] 
    param,perr = def_reg(n_cores,walltime)
    print(fmax, 'Hz: ',param,perr)
    
    
    
# Talapas: for every ncores
print()
for cores in [28,84,140,196,280]:
    n_el = data[data[:,0]==cores,4] 
    walltime = data[data[:,0]==cores,1] 
    param,perr = def_reg(n_el,walltime)
    print('N_cores',cores,':',param,perr)

print()
for cores in [28,84,140,196,280]:
    n_el = data_c[data_c[:,0]==cores,4] 
    walltime = data_c[data_c[:,0]==cores,1] 
    param,perr = def_reg(n_el,walltime)
    print('N_cores_c',cores,':',param,perr)

print()


#%% Plotting curve fit

#%% Using curve_fit

## https://stackoverflow.com/questions/28372597/python-curve-fit-with-multiple-independent-variables
# # You can pass curve_fit a multi-dimensional array for the independent variables, but then your func must accept 
# # the same thing. For example, calling this array X and unpacking it to x, y for clarity:

import numpy as np
from scipy.optimize import curve_fit

def func_3param(X, a, b, c):
    x,y = X
    exp = a + b*np.log10(x) + c*np.log10(y) #*x
    return 10**exp


# initial guesses for a,b,c:
p0 = 0, 0., 0.
dp = 2

data_n = data[:,[0,1,4]] # n_cores, walltime, n_elem
data_cn = data_c[:,[0,1,4]]
data_c3n = data_c3[:,[0,1,4]]

data_n[:,1] = data_n[:,1]#/3600 # converting walltime to hours
data_cn[:,1] = data_cn[:,1]#/3600
data_c3n[:,1] = data_c3n[:,1]#/3600

data_n[:,2] = data_n[:,2]#/1e9 # converting walltime to hours
data_cn[:,2] = data_cn[:,2]#/1e9
data_c3n[:,2] = data_c3n[:,2]#/1e9


popt, pcov = curve_fit(func_3param, (data_n[:,2],data_n[:,0]), data_n[:,1], p0)
perr = np.sqrt(np.diag(pcov))
param = np.round(popt,dp)
a_t = param[0]
b_t = param[1]
c_t = param[2]
print(param,np.round(perr,dp))

popt, pcov = curve_fit(func_3param, (data_cn[:,2],data_cn[:,0]), data_cn[:,1], p0)
perr = np.sqrt(np.diag(pcov))
param = np.round(popt,dp)
a_c2 = param[0]
b_c2 = param[1]
c_c2 = param[2]
print(param,np.round(perr,dp))

popt, pcov = curve_fit(func_3param, (data_c3n[:,2],data_c3n[:,0]), data_c3n[:,1], p0)
perr = np.sqrt(np.diag(pcov))
param = np.round(popt,dp)
a_c3 = param[0]
b_c3 = param[1]
c_c3 = param[2]
print(param,np.round(perr,dp))

print()



# default parameters
# tcb10 = ['#006BA4', '#FF800E', '#ABABAB', '#595959', '#5F9ED1', '#C85200', 
#          '#898989', '#A2C8EC', '#FFBC79', '#CFCFCF']
tcb10 = ['#006BA4',  '#898989','#595959', '#FF800E','#5F9ED1',  '#ABABAB','#C85200',
          '#A2C8EC', '#CFCFCF', '#FFBC79', ]

markers = ['o', '^', 'D', 'P', '>', '-']
ncores = ['28', '84', '140', '196', '280']
freq = ['0.1', '0.2', '0.3']
freq_c = ['0.1_c', '0.2_c', '0.3_c']
freq_c3 = ['0.1_c3', '0.2_c3', '0.3_c3']
nelem = ['1.44E+7', '1.14E+8', '3.83E+8']

lw = 3
ls = ['-','--','-.','-','--','-.','-','--','-.','-','--']

s = 200
label_font = 100
fontsize=30
title_pad = 30
nbins = 15

# 
fig, axes = plt.subplots(1,2,figsize=(12, 7))
fig.tight_layout(h_pad=10,w_pad=10)

def calc_walltime(n_cores_scal,n_ele,a,b,c):
    
    return 10**(a + b*np.log10(n_ele) + c*np.log10(n_cores_scal))



n_cores_scal = np.linspace(1.e1,1.e3,101)

# Talapas
walltime_scal_t = calc_walltime(n_cores_scal,1.44e7,a_t,b_t,c_t)
axes[0].plot(n_cores_scal,walltime_scal_t,lw=lw,ls = ls[1],c=tcb10[9])

walltime_scal_t = calc_walltime(n_cores_scal,1.14e8,a_t,b_t,c_t)  
axes[0].plot(n_cores_scal,walltime_scal_t,lw=lw,ls = ls[1],c=tcb10[9])

walltime_scal_t = calc_walltime(n_cores_scal,3.83e8,a_t,b_t,c_t)  
axes[0].plot(n_cores_scal,walltime_scal_t,lw=lw,ls = ls[1],c=tcb10[9])
    
# Container v2.01
walltime_scal_t = calc_walltime(n_cores_scal,1.44e7,a_c2,b_c2,c_c2)
axes[0].plot(n_cores_scal,walltime_scal_t,lw=lw,ls = ls[2],c=tcb10[6])

walltime_scal_t = calc_walltime(n_cores_scal,1.14e8,a_c2,b_c2,c_c2)  
axes[0].plot(n_cores_scal,walltime_scal_t,lw=lw,ls = ls[2],c=tcb10[6])

walltime_scal_t = calc_walltime(n_cores_scal,3.83e8,a_c2,b_c2,c_c2)  
axes[0].plot(n_cores_scal,walltime_scal_t,lw=lw,ls = ls[2],c=tcb10[6])


# Walltime vs no of cores
for i in range(len(freq)):
    n_elem=[]
    walltime=[]
    n_cores = data[data[:,3]==freq[i],0] 
    walltime = data[data[:,3]==freq[i],1]
    axes[0].scatter(n_cores,walltime,c=tcb10[i],marker = markers[i],s = s,label='no elem = '+ nelem[i])
    axes[0].plot(n_cores,walltime,lw=lw,ls = '--',c=tcb10[i])

for i in range(len(freq_c)):
    n_elem=[]
    walltime=[]
    n_cores = data_c[data_c[:,3]==freq_c[i],0] 
    walltime = data_c[data_c[:,3]==freq_c[i],1]
    axes[0].scatter(n_cores,walltime,linewidths = 2,c=tcb10[i],marker = markers[i],s = s,edgecolor ="red")#,label='no elem = '+ nelem[i])
    axes[0].plot(n_cores,walltime,lw=lw,ls = '-',c=tcb10[i])
    

axes[0].set_xlim([10,1000])
axes[0].set_ylim([0.1,100])
axes[0].set_xlabel('No of Cores',fontsize=fontsize)
#axes[0].legend(loc='lower left',fontsize=fontsize/2)
axes[0].set_ylabel('Walltime (hours)',fontsize=fontsize)

[j.set_linewidth(2) for j in axes[0].spines.values()]
axes[0].grid(b=True, which='both',color = 'k', linestyle = markers[5], linewidth = 0.2)

axes[0].xaxis.set_tick_params(labelsize=fontsize)
axes[0].yaxis.set_tick_params(labelsize=fontsize)

axes[0].set_xscale("log");
axes[0].set_yscale("log");

axes[0].set_title('Scaling Relationship for SW4 v2.01 \n(Total time = 220 s) with curvefit results',
fontsize=fontsize+5, y=1.05, x=0.95)#,fontdict={"weight": "bold"}) 
#pad =title_pad,

axes[1].axis('off')

# Creating legend
legend_elements = []

legend_elements.append(Line2D([],[],color=tcb10[0], marker=markers[0], markerfacecolor=tcb10[0],linewidth=0,markeredgecolor=tcb10[0],
                              markersize=s/10,alpha=1.0,label='no elem = '+ nelem[0]))

legend_elements.append(Line2D([0],[0],color=tcb10[0], marker=markers[1], markerfacecolor=tcb10[1],linewidth=0,markeredgecolor=tcb10[1],
                              markersize=s/10,alpha=1.0,label='no elem = '+ nelem[1]))

legend_elements.append(Line2D([0],[0],color=tcb10[0], marker=markers[2], markerfacecolor=tcb10[2],linewidth=0,markeredgecolor=tcb10[2],
                              markersize=s/10,alpha=1.0,label='no elem = '+ nelem[2]))

legend_elements.append(Line2D([],[],color=tcb10[0], marker=markers[0], markerfacecolor=tcb10[0],linestyle=ls[1], linewidth=2,
                              markersize=s/10,alpha=0))


legend_elements.append(Line2D([],[],color=tcb10[0], marker=markers[0], markerfacecolor=tcb10[0],linestyle=ls[1], linewidth=2,
                              markersize=s/10,alpha=1.0,label='SW4 on Talapas'))

legend_elements.append(Line2D([],[],color=tcb10[0], marker=markers[0], markerfacecolor=tcb10[0],linestyle=ls[0], linewidth=3,
                              markersize=s/10,alpha=1.0,label='SW4 Container',markeredgewidth=3,  markeredgecolor=(1, 0, 0, 0))) #,edgecolor ="red"

legend_elements.append(Line2D([],[],color=tcb10[9], marker=markers[0], markerfacecolor=tcb10[9],linestyle=ls[1], linewidth=3,
                              markersize=s/10000,alpha=1.0,label='SW4 Talapas Curve Fit'))

legend_elements.append(Line2D([],[],color=tcb10[6], marker=markers[0], markerfacecolor=tcb10[6],linestyle=ls[2], linewidth=3,
                              markersize=s/10000,alpha=1.0,label='SW4 Container Curve Fit',markeredgewidth=3,  markeredgecolor=(1, 0, 0, 0))) #,edgecolor ="red"


plt.legend(handles=legend_elements, bbox_to_anchor=(-0.3, 0.7 ), loc='upper left', 
            fontsize=fontsize-5,frameon=False,labelspacing = 1)

plt.text(-0.25, 0.75, 'LEGEND', color='k', fontsize=fontsize)#,fontdict={"weight": "bold"})


plt.savefig('scaling_relationship_curvefit.png',bbox_inches='tight', dpi=200)



#%% Plotting figures
fig, axes = plt.subplots(2,3,figsize=(12, 7))
fig.tight_layout(h_pad=8,w_pad=5)#, w_pad=0.5, h_pad=7.0,rect=[0, 0.15, 0.8, 0.9]) #0, 0.03, 0.85, 0.90]
axes=axes.flatten()

s = 200
label_font = 80
fontsize=20
title_pad = 30

# Walltime vs No of elements
for i in range(len(ncores)):
    n_elem=[]
    walltime=[]
    j=0
    n_elem = data[data[:,0]==int(ncores[i]),4] 
    walltime = data[data[:,0]==int(ncores[i]),1]#/3600 
    axes[i].scatter(n_elem,walltime,c=tcb10[j],marker = markers[j],s = s)
    axes[i].plot(n_elem,walltime,lw=lw,ls = '--',c=tcb10[j])
    axes[i].set_title(label='No cores = ' + ncores[i],fontsize=fontsize+5)

    n_elem = data_c[data_c[:,0]==int(ncores[i]),4] 
    walltime = data_c[data_c[:,0]==int(ncores[i]),1]#/3600 
    axes[i].scatter(n_elem,walltime,c="red",marker = markers[j],s = s,edgecolor ="red") #tcb10[j]
    axes[i].plot(n_elem,walltime,lw=lw,ls = '-',c="red")#tcb10[j])
    
    axes[i].set_xlim([1e7,1e9])
    axes[i].set_ylim([0.1,100])
    axes[i].set_xlabel('No of Elements',fontsize=fontsize)
    
    if i==0 or i == 3:
        axes[i].set_ylabel('Walltime (hours)',fontsize=fontsize)
    #axes[i].legend(loc='lower right',fontsize=fontsize/2)


# Add other figure properties   
for i in range(len(axes)-1):
    [j.set_linewidth(2) for j in axes[i].spines.values()]
    axes[i].grid(b=True, which='both',color = 'k', linestyle = markers[5], linewidth = 0.2)

    axes[i].xaxis.set_tick_params(labelsize=fontsize)
    axes[i].yaxis.set_tick_params(labelsize=fontsize)
    
    axes[i].set_xscale("log");
    axes[i].set_yscale("log");

axes[-1].axis('off')

# Creating legend
legend_elements = []

legend_elements.append(Line2D([],[],color=tcb10[0], marker=markers[0], markerfacecolor=tcb10[0],linestyle=ls[1], linewidth=2,
                              markersize=s/15,alpha=1.0,label='SW4 on Talapas'))

legend_elements.append(Line2D([],[],color="red", marker=markers[0], markerfacecolor="red",linestyle=ls[0], linewidth=3,
                              markersize=s/15,alpha=1.0,label='SW4 Container',markeredgewidth=3,  markeredgecolor=(1, 0, 0, 0))) #,edgecolor ="red"

plt.legend(handles=legend_elements, bbox_to_anchor=(-0.2, 0.7 ), loc='upper left', 
            fontsize=fontsize+5,frameon=False,labelspacing = 1)

plt.text(-0.15, 0.75, 'LEGEND', color='k', fontsize=fontsize+5)#,fontdict={"weight": "bold"})


plt.suptitle('Scaling for SW4 v2.01 (Total time = 220 s)',fontsize=fontsize+10, y=1.13)#,fontdict={"weight": "bold"}) #pad =title_pad,

plt.savefig('scaling_relationship2.png',bbox_inches='tight', dpi=200)



#%% SW4 v2.01 on Talapas vs Container
fig, ax = plt.subplots(1,2,figsize=(12, 7))
fig.tight_layout(h_pad=10,w_pad=10)#, w_pad=0.5, h_pad=7.0,rect=[0, 0.15, 0.8, 0.9]) #0, 0.03, 0.85, 0.90]

ax[0].axline((0, 0), slope=1, color='C1', label='slope=1.0', linewidth = lw, linestyle = ls[0])
ax[0].axline((0, 0), slope=1.5, color='red', label='slope=1.5', linewidth = lw, linestyle = ls[1])
ax[0].axline((0, 0), slope=2, color='C8', label='slope=2.0', linewidth = lw, linestyle = ls[2])


ax[0].scatter(data[:,1],data_c[:,1],c=tcb10[0],marker = markers[1],s = s/1)

ax[0].set_xlim([0,216000/3600])
ax[0].set_ylim([0,216000/3600])
ax[0].set_aspect('equal')

ax[0].legend(loc='lower right',fontsize=fontsize/1.5)
ax[0].grid(b=True, which='both',color = 'k', linestyle = markers[5], linewidth = 0.2)

ax[0].set_xlabel('Talapas (hours)',fontsize=fontsize)
ax[0].set_ylabel('Container (hours)',fontsize=fontsize)

[j.set_linewidth(2) for j in ax[0].spines.values()]
ax[0].grid(b=True, which='both',color = 'k', linestyle = markers[5], linewidth = 0.2)

ax[0].xaxis.set_tick_params(labelsize=fontsize)
ax[0].yaxis.set_tick_params(labelsize=fontsize)

ax[0].set_title('Walltime for SW4 v2.01 installed on Talapas and \n using the container (Total time = 220 s)',
                fontsize=fontsize+5, y=1.1, x=1.2)#,fontdict={"weight": "bold"}) #pad =title_pad,



HIST_BINS = np.linspace(0.5, 2, nbins)
ax[1].hist(data_c[:,1]/data[:,1],HIST_BINS,ec="white", fc="blue", alpha=0.5)#,c=tcb10[0])#,marker = markers[1],s = s/2)
ax[1].set_aspect(0.5)

ax[1].set_xlim([0.5,2])
#ax[1].set_ylim([0,2])

ax[1].grid(b=True, which='both',color = 'k', linestyle = markers[5], linewidth = 0.2)

ax[1].set_xlabel('Walltime ratio (Container/Talapas)',fontsize=fontsize)
ax[1].set_ylabel('frequency',fontsize=fontsize)

[j.set_linewidth(2) for j in ax[1].spines.values()]
ax[1].grid(b=True, which='both',color = 'k', linestyle = markers[5], linewidth = 0.2)

ax[1].xaxis.set_tick_params(labelsize=fontsize)
ax[1].yaxis.set_tick_params(labelsize=fontsize)


plt.savefig('scaling_relationship3.png',bbox_inches='tight', dpi=200)


#%% Container SW4 v2.01 vs v3.0
fig, ax = plt.subplots(1,2,figsize=(12, 6))
fig.tight_layout(h_pad=10,w_pad=10)#, w_pad=0.5, h_pad=7.0,rect=[0, 0.15, 0.8, 0.9]) #0, 0.03, 0.85, 0.90]

ax[0].scatter(data_c[:,1],data_c3[:,1],c=tcb10[0],marker = markers[1],s = s/1)

ax[0].axline((0, 0), slope=1, color='C1', label='slope=1', linewidth = lw, linestyle = ls[0])

ax[0].set_xlim([0,210000/3600])
ax[0].set_ylim([0,210000/3600])

ax[0].set_aspect('equal')


ax[0].legend(loc='lower right',fontsize=fontsize/1.5)
ax[0].grid(b=True, which='both',color = 'k', linestyle = markers[5], linewidth = 0.2)

ax[0].set_xlabel('SW4 v2.01 (hours)',fontsize=fontsize)
ax[0].set_ylabel('SW4 v3.0 (hours)',fontsize=fontsize)

[j.set_linewidth(2) for j in ax[0].spines.values()]
ax[0].grid(b=True, which='both',color = 'k', linestyle = markers[5], linewidth = 0.2)

ax[0].xaxis.set_tick_params(labelsize=fontsize)
ax[0].yaxis.set_tick_params(labelsize=fontsize)

ax[0].set_xlim([0,60])
ax[0].set_ylim([0,60])

ax[0].set_title('Walltime for Container versions of SW4 v2.01 vs v3.0 \n (Total time = 220 s)',
                fontsize=fontsize+5, y=1.1, x=1.2)#,fontdict={"weight": "bold"}) #pad =title_pad,



HIST_BINS = np.linspace(0.5, 2, nbins)
ax[1].hist(data_c[:,1]/data_c3[:,1],HIST_BINS,ec="white", fc="blue", alpha=0.5)#,c=tcb10[0])#,marker = markers[1],s = s/2)

ax[1].set_aspect(0.36)

ax[1].set_xlim([0.5,2])
#ax[1].set_ylim([0,2])

ax[1].grid(b=True, which='both',color = 'k', linestyle = markers[5], linewidth = 0.2)

ax[1].set_xlabel('Walltime ratio (v2.01 / v3.0)',fontsize=fontsize)
ax[1].set_ylabel('frequency',fontsize=fontsize)

[j.set_linewidth(2) for j in ax[1].spines.values()]
ax[1].grid(b=True, which='both',color = 'k', linestyle = markers[5], linewidth = 0.2)

ax[1].xaxis.set_tick_params(labelsize=fontsize)
ax[1].yaxis.set_tick_params(labelsize=fontsize)

plt.savefig('scaling_relationship4.png',bbox_inches='tight', dpi=200)

