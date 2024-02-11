#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 13:48:47 2023

@author: oluwaseunfadugba
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 11:39:29 2022

@author: oluwaseunfadugba
"""

import os
from obspy.core import read
import matplotlib.pyplot as plt
from scipy import integrate
import pandas as pd
import numpy as np
from glob import glob
import time
import obspy
from matplotlib.lines import Line2D
from string import ascii_lowercase as alphab

start = time.time()
current_dir = '/Users/oluwaseunfadugba/Documents/Projects/SW4_Singularity_Container/figures/full_vs_slant_rfile/'

os.chdir(current_dir)

full_rfile_dir = current_dir + 'full_rfile/ibaraki2011_scrmod_srf3d_rupt5.sw4output/'
slant_rfile_dir = current_dir + 'slant_rfile/ibaraki2011_scrmod_srf3d_rupt5.sw4output/'


#%% Functions
def extract_n_process_sta(stadir,sta):
    Z = read(stadir + sta + '.u');  #LYE.sac
    N = read(stadir + sta + '.n'); 
    E = read(stadir + sta + '.e'); 
    
    Z.filter('lowpass', freq=0.45, corners=2, zerophase=True)
    N.filter('lowpass', freq=0.45, corners=2, zerophase=True)
    E.filter('lowpass', freq=0.45, corners=2, zerophase=True)
    
    Z.resample(1)
    N.resample(1)
    E.resample(1)
    
    Z_int = Z.integrate() #integrate.cumtrapz(Z[0].data,Z[0].times(),initial=0)
    N_int = N.integrate()#integrate.cumtrapz(N[0].data,N[0].times(),initial=0)
    E_int = E.integrate()#integrate.cumtrapz(E[0].data,E[0].times(),initial=0)

    # Z_int = Z[0].data
    # N_int = N[0].data
    # E_int = E[0].data

    return Z_int,N_int,E_int

def plot_figures(Z_f_int,N_f_int,E_f_int,Z_sl_int,N_sl_int,E_sl_int):
    # ------------------------------------------------------------------------------------------
    # Figures
    fontsize = 80
    linewd_obs = 7
    xlim_range = 400
    suptitle = 'Validating the Slant 3D velocity Model Rfile'
    
    fig = plt.figure(figsize=(70, 25))
    fig.suptitle(suptitle, fontsize=fontsize+10)
    #'1D Velocity and Pseudo3D Simulations (Ibaraki2011_srcmod_00000 STF)'
    time_shift =0
    
    # ------------------------------------------------------------------------------------------
    # Z-component
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.set_title('Station '+ str(sta) + ' (Z-comp)', fontsize=fontsize)
    
    ax1.plot(Z_f_int, "k-",label='Full rfile',linewidth=linewd_obs)
    ax1.plot(Z_sl_int, "r-",label='Slant rfile',linewidth=linewd_obs)
    
    ax1.xaxis.set_tick_params(labelsize=fontsize)
    ax1.yaxis.set_tick_params(labelsize=fontsize)
    ax1.set_xlabel('time (s)',fontsize=fontsize)
    ax1.set_ylabel('displacement',fontsize=fontsize)
    ax1.set_xlim(0,xlim_range)
    ax1.legend(fontsize=fontsize-5)
    plt.grid()
    
    # # ------------------------------------------------------------------------------------------
    # # N-component
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.set_title('Station '+ str(sta) + ' (N-comp)', fontsize=fontsize)
    
    ax2.plot(N_f_int, "k-",label='Full rfile',linewidth=linewd_obs)
    ax2.plot(N_sl_int, "r-",label='Slant rfile',linewidth=linewd_obs)
    
    ax2.xaxis.set_tick_params(labelsize=fontsize)
    ax2.yaxis.set_tick_params(labelsize=fontsize)
    ax2.set_xlabel('time (s)',fontsize=fontsize)
    ax2.set_ylabel('displacement',fontsize=fontsize)
    ax2.set_xlim(0,xlim_range)
    ax2.legend(fontsize=fontsize-5)
    plt.grid()
    
    # # ---------------------------------------------
    # # E-component
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.set_title('Station '+ str(sta) + ' (E-comp)', fontsize=fontsize)
    
    ax3.plot(E_f_int, "k-",label='Full rfile',linewidth=linewd_obs)
    ax3.plot(E_sl_int, "r-",label='Slant rfile',linewidth=linewd_obs)
    
    ax3.xaxis.set_tick_params(labelsize=fontsize)
    ax3.yaxis.set_tick_params(labelsize=fontsize)
    ax3.set_xlabel('time (s)',fontsize=fontsize)
    ax3.set_ylabel('displacement',fontsize=fontsize)
    ax3.set_xlim(0,xlim_range)
    ax3.legend(fontsize=fontsize-5)
    plt.grid()
    
    figpath = os.getcwd() +'/fig.full_vs_slant_rfiles_wfs_'+ str(sta) +'.Ibaraki2011_srcmod_rupt5.png'
    plt.savefig(figpath, bbox_inches='tight', dpi=100)
    plt.show()
    #plt.close()  

#%% Driver

stas = ['0165','0037','0042']

fig, axes = plt.subplots(3,3,figsize=(90, 90))
fig.tight_layout(h_pad=70,w_pad=70)#, w_pad=0.5, h_pad=7.0,rect=[0, 0.15, 0.8, 0.9]) #0, 0.03, 0.85, 0.90]

plt.text(-1150, 1.03, 'Validating the Slant 3D Velocity Model (rfile)', color='k', fontsize=250,fontdict={"weight": "bold"})


tcb10 = ['#006BA4', '#FF800E', '#ABABAB', '#595959', '#5F9ED1', '#C85200', '#898989', '#A2C8EC', '#FFBC79', '#CFCFCF']
plt.style.use('tableau-colorblind10') #('seaborn-colorblind') #

colors = [tcb10[0],tcb10[1]]
linewd_obs = 20
ls = ['-','--']

fontsize = 180
title_pad = 60

for i in range(len(stas)):
    sta = stas[i]
    
    print('Working on station '+sta)
   
    Z_f_int,N_f_int,E_f_int = extract_n_process_sta(full_rfile_dir,sta)
    Z_sl_int,N_sl_int,E_sl_int = extract_n_process_sta(slant_rfile_dir,sta)
    
    
    axes[i,0].plot(Z_f_int[0].times(), Z_f_int[0].data, colors[1],\
              linewidth=linewd_obs,ls=ls[1])
    axes[i,1].plot(N_f_int[0].times(), N_f_int[0].data, colors[1],\
              linewidth=linewd_obs,ls=ls[1])
    axes[i,2].plot(E_f_int[0].times(), E_f_int[0].data, colors[1],\
              linewidth=linewd_obs,ls=ls[1])
         
    axes[i,0].plot(Z_sl_int[0].times(), Z_sl_int[0].data, colors[0],\
              linewidth=linewd_obs,ls=ls[0])
    axes[i,1].plot(N_sl_int[0].times(), N_sl_int[0].data, colors[0],\
              linewidth=linewd_obs,ls=ls[0])
    axes[i,2].plot(E_sl_int[0].times(), E_sl_int[0].data, colors[0],\
              linewidth=linewd_obs,ls=ls[0])        
    

    axes[i,0].set_title('Station '+sta+' (Z)',fontsize=fontsize+15,
                        fontdict={"weight": "bold"},pad =title_pad)
    axes[i,1].set_title('Station '+sta+' (N)',fontsize=fontsize+15,
                        fontdict={"weight": "bold"},pad =title_pad)
    axes[i,2].set_title('Station '+sta+' (E)',fontsize=fontsize+15,
                        fontdict={"weight": "bold"},pad =title_pad)
    
    
    
    axes[i,0].xaxis.set_tick_params(labelsize=fontsize)
    axes[i,0].yaxis.set_tick_params(labelsize=fontsize)
    
    axes[i,1].xaxis.set_tick_params(labelsize=fontsize)
    axes[i,1].yaxis.set_tick_params(labelsize=fontsize)
    
    axes[i,2].xaxis.set_tick_params(labelsize=fontsize)
    axes[i,2].yaxis.set_tick_params(labelsize=fontsize)
    
    
    axes[i,0].set_xlabel('time (s)',fontsize=fontsize)
    axes[i,1].set_xlabel('time (s)',fontsize=fontsize)
    axes[i,2].set_xlabel('time (s)',fontsize=fontsize)
    
    axes[i,0].set_ylabel('displacement',fontsize=fontsize)
    
    axes[i,0].set_xlim(0,400)
    axes[i,1].set_xlim(0,400)
    axes[i,2].set_xlim(0,400)
    
    axes[i,0].tick_params(axis='x',labelsize=fontsize,labelrotation=0,length=40, width=10)
    axes[i,0].tick_params(axis='y',labelsize=fontsize,labelrotation=0,length=40, width=10)
    
    axes[i,1].tick_params(axis='x',labelsize=fontsize,labelrotation=0,length=40, width=10)
    axes[i,1].tick_params(axis='y',labelsize=fontsize,labelrotation=0,length=40, width=10)
    
    axes[i,2].tick_params(axis='x',labelsize=fontsize,labelrotation=0,length=40, width=10)
    axes[i,2].tick_params(axis='y',labelsize=fontsize,labelrotation=0,length=40, width=10)
    
    # Increasing the linewidth of the frame border 
    for pos in ['right', 'top', 'bottom', 'left']:
        axes[i,0].spines[pos].set_linewidth(linewd_obs/3)
        axes[i,1].spines[pos].set_linewidth(linewd_obs/3)
        axes[i,2].spines[pos].set_linewidth(linewd_obs/3)
        
        
# Adding legend
legend_elements = []

legend = ['Slant rfile','Full rfile']
colors = [tcb10[0],tcb10[1]]


legend_elements.append(Line2D([0],[0], linewidth=linewd_obs, linestyle='-',  color=colors[0], 
                              alpha=1.0,label=legend[0]))

legend_elements.append(Line2D([0],[0], linewidth=linewd_obs, linestyle='--',  color=colors[1], 
                              alpha=1.0,label=legend[1]))

plt.legend(handles=legend_elements, bbox_to_anchor=(-2, -0.8 ), loc='lower left', 
            fontsize=200,frameon=False, ncol=2)

plt.text(-760, -0.12, 'LEGEND', color='k', fontsize=200,fontdict={"weight": "bold"})

# subplot label
ax = axes.flatten()
subplt_labelpos=[-0.25, 1.125]
for j in range(9):
    # Add alphabet labels to the subplots
    ax[j].text(subplt_labelpos[0], subplt_labelpos[1], '('+alphab[j].upper()+')', 
              transform=ax[j].transAxes, fontsize=180, fontweight="bold", va="top")


figpath = os.getcwd() +'/fig.full_vs_slant_rfiles_Ibaraki2011_srcmod_rupt5.png'
plt.savefig(figpath, bbox_inches='tight', dpi=100)
plt.show()


# os.system('rm -rf full_vs_slant_figures')
# os.system('mkdir full_vs_slant_figures')
# os.system('mv fig.full_vs_slant_rfiles* full_vs_slant_figures')


# ####################################################################
end = time.time()
time_elaps = end - start
if time_elaps < 60:
    print(f'Duration: {round(time_elaps)} seconds')
else:
    print(f'Duration: {round(time_elaps/60)} minutes')
    
    
  
    
    