#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  4 11:27:06 2023

@author: oluwaseunfadugba

"""
import time
from glob import glob

start = time.time()

def extract_source_allpts(rupt):
    #Read mudpy file
    f=np.genfromtxt(rupt)
    
    lon_s = np.array([])
    lat_s = np.array([])
    depth_s = np.array([])
    strike_s = np.array([])
    dip_s = np.array([])
    area_s = np.array([])
    rigidity_s = np.array([])
    
    #loop over subfaults
    for kfault in range(len(f)):

        zero_slip=False

        #Get subfault parameters
        lon=f[kfault,1]
        lat=f[kfault,2]
        depth=f[kfault,3]*1000 #in m for sw4
        strike=f[kfault,4]
        dip=f[kfault,5]
        area=f[kfault,10]*f[kfault,11] #in meters, cause this isn't dumb SRF
        #tinit=f[kfault,12]+time_pad
        #rake=rad2deg(arctan2(f[kfault,9],f[kfault,8]))
        slip=np.sqrt(f[kfault,8]**2+f[kfault,9]**2)
        rise_time=f[kfault,7]
        rigidity=f[kfault,13]
            
        lon_s = np.append(lon_s, lon)
        lat_s = np.append(lat_s, lat)
        depth_s = np.append(depth_s, depth)
        strike_s = np.append(strike_s, strike)
        dip_s = np.append(dip_s, dip)
        area_s = np.append(area_s, area)
        rigidity_s = np.append(rigidity_s, rigidity)
           
    return lon_s,lat_s,depth_s,strike_s,dip_s,area_s,rigidity_s

#%%
import numpy as np
#from numpy import genfromtxt
import os
current_dir = '/Users/oluwaseunfadugba/Documents/Projects/SW4_Singularity_Container/create_sw4inputs/'
os.chdir(current_dir)
os.sys.path.insert(0, "/Users/oluwaseunfadugba/code/MudPy/src/python")

import pandas as pd
#import matplotlib.pyplot as plt
import os
#import obspy
#from mudpy.forward import lowpass
#from obspy.core import UTCDateTime
#import tsueqs_main_fns as tmf

eqname = 'ibaraki2011'

hh = '/Users/oluwaseunfadugba/Documents/Projects/TsE_ValerieDiego/TsE_1D_vs_3D/1D_Modeling_using_FQs_Mudpy/'
flatfile = hh+'IM_Residuals/Results_ibaraki2011_srcmod_IM_residuals/Flatfiles_IMs_ibaraki2011_srcmod.csv'
rupt_5 = hh+'Running_FakeQuakes_now/ibaraki2011_srcmod/ruptures/ibaraki2011_srcmod.000005.rupt'

orig_time = "2011-03-11T06:15:34"; 
hypolon = 141.2653; hypolat = 36.1083; dept = 43.2
SNR_thresh = 3;   msize = 25

fmax = [0.1, 0.2, 0.3,0.4]#, 0.4, 0.5]
n_nodes = [1,3, 5, 7, 10, 15, 20]#, 15]#, 20]
runs_type = ['CTAL3', 'CTAL', 'TAL', 'CQZ', 'QZ']

lat0 =  34.5; lon0 = 133.1 # deg
xdim = 600; ydim= 900; # kmcd
maxdep = 200; reflayer = 75 # km
minvs = 1200 # m/s
PPW = 8; tot_time = 220

#%% Setting file structure

output_dir = current_dir+'sw4in2_'+eqname+'/'
os.system('rm -rf ' + output_dir)
os.system('mkdir ' + output_dir)
os.system('mkdir ' + output_dir+'talapas_sw4in/');
os.system('mkdir ' + output_dir+'quartz_sw4in/');

os.system('cp ibaraki2011_srcmod_srf3d_rupt5.srf ' + output_dir+'talapas_sw4in/')
os.system('cp ibaraki2011_srcmod_srf3d_rupt5.srf ' + output_dir+'quartz_sw4in/')

#%% extracting station and rupture data

stadata = pd.read_csv(flatfile)

new_data = stadata.loc[(stadata['rupt_no'] == 1)& 
                        (stadata['SNR_obs'] >= SNR_thresh)].reset_index()

[lon_s,lat_s,depth_s,strike_s,dip_s,area_s,rigidity_s] = extract_source_allpts(rupt_5); 


#%% Generate SW4 Input files

def create_sw4in(fmax_i,n_node_i,runs_type_i,ydim,xdim,
                 maxdep,reflayer,lat0,lon0,tot_time,minvs,PPW):
    
    hmax = 2*(minvs/(fmax_i*PPW))
    direc = 'simul_f_'+str(fmax_i)+'_n_'+str(n_node_i)+'_'+runs_type_i

    fout=open(output_dir+direc+'.sw4input','w')
    
    fout.write('# SW4 input file\n')
    fout.write('\n')
    fout.write('fileio path=%s pfs=1 \n'%(direc))
    fout.write('\n')
    fout.write('# grid size is set in the grid command\n')
    fout.write('# DO NOT CHANGE AZ!\n')
    fout.write('grid x=%se3 y=%se3 z=%se3 h=%s lat=%s lon=%s az=35 mlat=120000 mlon=80000\n'%(ydim,xdim,maxdep,hmax,lat0,lon0))
    fout.write('refinement zmax=%se3\n'%(reflayer))
    fout.write('\n')
    fout.write('# wave speeds are specified at 1.0 Hz, 3 viscoelastic mechanisms by default\n')
    #fout.write('# attenuation phasefreq=1.0 nmech=3 maxfreq=0.5\n')
    fout.write('attenuation minppw=%s\n'%(PPW))
    fout.write('\n')
    fout.write('time t=%s utcstart=03/11/2011:06:15:14.0\n'%(tot_time))
    fout.write('\n')
    fout.write('developer reporttiming=1\n')
    fout.write('\n')
    fout.write('# threshold on vp and vs\n')
    fout.write('globalmaterial vsmin=1200 vpmin=2500\n')
    fout.write('\n')
    fout.write('supergrid gp=30\n')
    fout.write('\n')
    fout.write('block vs=1200 vp=2500 rho=2100 Qs=600 Qp=1300 z1=0.0 z2=1000\n')
    fout.write('block vs=3400 vp=6000 rho=2700 Qs=600 Qp=1300 z1=1000 z2=11000\n')
    fout.write('block vs=3700 vp=6600 rho=2900 Qs=600 Qp=1300 z1=11000 z2=21000\n')
    fout.write('block vs=4000 vp=7200 rho=3100 Qs=600 Qp=1300 z1=21000 z2=31000\n')
    fout.write('block vs=4000 vp=7200 rho=3100 Qs=600 Qp=1300 z1=31000 z2=40000\n')
    fout.write('block vs=4484.86 vp=8101.19 rho=3379.06 Qs=600 Qp=1446 z1=40000 z2=60000\n')
    fout.write('block vs=4477.15 vp=8089.07 rho=3376.88 Qs=600 Qp=1447 z1=60000 z2=80000\n')
    fout.write('block vs=4469.53 vp=8076.88 rho=3374.71 Qs=80 Qp=195 z1=80000 z2=115000\n')
    fout.write('block vs=4456.43 vp=8055.40 rho=3370.91 Qs=80 Qp=195 z1=115000 z2=150000\n')
    fout.write('block vs=4443.61 vp=8033.70 rho=3367.10 Qs=80 Qp=195 z1=150000 z2=185000\n')
    fout.write('block vs=4431.08 vp=8011.80 rho=3363.30 Qs=80 Qp=195 z1=185000\n')
    fout.write('\n')
    fout.write('prefilter fc2=%s type=lowpass passes=2 order=2\n'%(fmax_i))
    fout.write('\n')
    fout.write('topography input=rfile zmax=30e3 order=3 file=3djapan_hv=2_3_4_500m_rot35.rfile\n')
    fout.write('\n')
    fout.write('# rfile format\n')
    fout.write('rfile filename=3djapan_hv=2_3_4_500m_rot35.rfile directory=./\n')
    fout.write('\n')
    #fout.write('# Output images of the elastic model \n')
    #fout.write('image mode=mag  z=0.0   timeInterval=5 file=sub%s.%s # on the surface \n'%(str(ii+1).zfill(4),rake_str))
    #fout.write('image mode=s    z=0.0   cycle=0        file=sub%s.%s # on the surface\n'%(str(ii+1).zfill(4),rake_str))
    #fout.write('image mode=s    x=200e3 cycle=0        file=sub%s.%s # vertical cross section\n'%(str(ii+1).zfill(4),rake_str))
    #fout.write('image mode=hmax z=0     time=390       file=sub%s.%s # solution on the surface\n'%(str(ii+1).zfill(4),rake_str))
    #fout.write('\n')

    fout.write('# SRF rupture\n')
    fout.write('rupture file=ibaraki2011_srcmod_srf3d_rupt5.srf\n')

    fout.write('\n')
    
    fout.write('# GNSS Stations \n')
    for i in range(len(new_data.index)):
            st_name = str(new_data['station'][i]).zfill(4)
            lat = new_data['stlat'][i]
            lon = new_data['stlon'][i]
            elev = 0
    
            fout.write('rec lat=%e lon=%e depth=%s file=%s.%s.disp nsew=1 variables=displacement\n'\
                        %(lat,lon,elev,direc,st_name)) 
    
    fout.write('\n')
    fout.close()

for i in range(len(fmax)):
    for j in range(len(n_nodes)):
        for k in range(len(runs_type)):
            create_sw4in(fmax[i],n_nodes[j],runs_type[k],ydim,xdim,
                         maxdep,reflayer,lat0,lon0,tot_time,minvs,PPW)
    
    
#%% Create subsmission script on Talapas

simul_list = np.array(sorted(glob('sw4in2_ibaraki2011/*TAL*')))
for j in range(len(simul_list)):
    filename = simul_list[j].split('/')[1]
    
    f = simul_list[j].split('_')
    fmax_in = f[3]
    nnodes_in = f[5]
    run_type_in = f[6].split('.')[0]
    
    WallTime = '20:00:00' #'70:00:00'
    ntask_per_node = 28
    partition = 'short' #'long' #'short'

    fout=open(output_dir+'submit_'+filename[:-9]+'.srun','w')

    fout.write('#!/bin/bash\n')
    fout.write('#SBATCH --partition=%s          ### Partition\n'% partition)
    fout.write('#SBATCH --job-name=%s_%s_%s            ### Job Name\n'%(fmax_in,nnodes_in,run_type_in))
    fout.write('#SBATCH --time=%s             ### WallTime\n'% WallTime)
    fout.write('#SBATCH --nodes=%s                  ### Number of Nodes\n'% str(nnodes_in))
    fout.write('#SBATCH --ntasks-per-node=%s       ### Number of tasks (MPI processes)\n'% str(ntask_per_node))
    fout.write('#SBATCH --account=waves            ### Account used for job submission\n')
    fout.write('#SBATCH --mem-per-cpu=4G \n')
    fout.write('\n')
    
    if run_type_in == 'TAL':
        fout.write('module load sw4-proj\n')
        fout.write('\n')
        fout.write('srun sw4 %s\n'%(filename))
    elif run_type_in == 'CTAL':
        fout.write('module load singularity\n')
        fout.write('\n')
        fout.write('mpirun -np $SLURM_NTASKS singularity run --home $(pwd) ../final_sw4_singuarity/sw4_2.01_cpu sw4 %s\n'%(filename))
        #e.g., mpirun -np $SLURM_NTASKS singularity run --home $(pwd) 
        # ../final_sw4_singuarity/sw4_2.01_cpu sw4 simul_f_0.1_n_3_CTAL.sw4input
    else:
        fout.write('module load singularity\n')
        fout.write('\n')
        fout.write('mpirun -np $SLURM_NTASKS singularity run --home $(pwd) ../final_sw4_singuarity/sw4_3.0_cpu sw4 %s\n'%(filename))
        #e.g., mpirun -np $SLURM_NTASKS singularity run --home $(pwd) 
        # ../final_sw4_singuarity/sw4_3.0-beta2_cpu sw4 simul_f_0.1_n_3_CTAL.sw4input
        
    fout.write('\n')
    fout.close()

# Creating a batch submission script for all Talapas jobs

fout=open(output_dir+'batch_submit_TAL.sh','w')

fout.write('#!/bin/bash\n')
fout.write('# loop over all submission files in the directory, \n')
fout.write('# print the filename and submit the jobs to SLURM\n')
fout.write('\n')
fout.write('# Syntax: bash batch_submit_TAL.sh\n')
fout.write('\n')
fout.write('for FILE in *.srun; do\n')
fout.write('    echo ${FILE}\n')
fout.write('    sbatch ${FILE}\n')
fout.write('\n')
fout.write('done\n')
fout.write('\n')
fout.close()

# Move files to respective folders
os.system('mv sw4in2_ibaraki2011/*TAL* ' + output_dir+'talapas_sw4in/')
os.system('mv sw4in2_ibaraki2011/*QZ* ' + output_dir+'quartz_sw4in/')

# ####################################################################
end = time.time()
time_elaps = end - start
if time_elaps < 60:
    print(f'Duration: {round(time_elaps)} seconds')
else:
    print(f'Duration: {round(time_elaps/60)} minutes')

