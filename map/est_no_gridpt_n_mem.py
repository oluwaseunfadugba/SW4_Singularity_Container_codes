#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  3 12:07:42 2023

@author: oluwaseunfadugba
"""
import numpy as np
import sys
flush = sys.stdout.flush()

#(600*900*((75/(0.3**3))+(125/(0.6**3))))*500e-9/4/24

lat0 =  34.5; lon0 = 133.1
xdim = 600; ydim= 900; 
maxdep = 200; reflayer = 75
minvs = 1200
PPW = 8
fmax = np.array([0.1,0.2,0.3,0.4,0.5])
N = len(fmax)
hmin = minvs/(PPW*fmax)/1e3

n_grid_pts = np.zeros(N)
req_mem = np.zeros(N)
n_cores_min = np.zeros(N)
n_nodes_min  = np.zeros(N)

fields = '{:>14}|{:>5}|{:>8}|{:>7}|{:>7}|{:>7}'
data = '{:>14}|{:>5}|{:>7.2e}|{:>7.2f}|{:7.0f}|{:>7.0f}'

#print(fields.format('grid', 'nx', 'ny', 'dx', 'dy'))
print(fields.format('SN', 'fmax', 'n_elem', 'req_mem', 'n_cores', 'n_nodes'))
flush

for i in range(len(fmax)):
    n_grid_pts[i] = (xdim*ydim*((reflayer/(hmin[i]**3))+
                             ((maxdep-reflayer)/((hmin[i]*2)**3))))
    
    req_mem[i] = n_grid_pts[i]*500e-9
    n_cores_min[i] = np.ceil(req_mem[i]/4)
    n_nodes_min[i] = np.ceil(n_cores_min[i]/24)
    
    print(data.format(i, fmax[i], n_grid_pts[i], req_mem[i], n_cores_min[i], n_nodes_min[i]))
    flush
print('')
print('Constant parameters:')
print('lat0 =',lat0,'lon0=',lon0,'xdim =',xdim,'; ydim=',ydim)
print('maxdepth=',maxdep,'; refinement layer dep = ',reflayer,'; PPW=',PPW)