#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  3 11:07:57 2023

@author: oluwaseunfadugba
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 13:04:37 2022

@author: oluwaseunfadugba
"""
#%% functions
def extract_source_pts(rupt):
    #Read mudpy file
    f=np.genfromtxt(rupt)
    
    lon_s = np.array([])
    lat_s = np.array([])
    depth_s = np.array([])
    
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

        #If subfault has zero rise time or zero slip
        zero_slip=False
        if slip==0:
            zero_slip=True
            #print('Zero slip at '+str(kfault))
        elif rise_time==0:
            slip=0
            zero_slip=True
            #print('Zero rise time at '+str(kfault))     

        #make rake be -180 to 180
        #if rake>180:
        #    rake=rake-360

        if zero_slip==False:
            
           lon_s = np.append(lon_s, lon)
           lat_s = np.append(lat_s, lat)
           depth_s = np.append(depth_s, depth)
           
    return lon_s,lat_s,depth_s


#%%
import pygmt, os
import pandas as pd
from shapely.geometry import Point, Polygon
import numpy as np

'''
This code plots the location map showing the GNSS locations and earthquake locations.

To use this code, change the environment to pygmt by running 
"conda activate pygmt" on a new terminal and restart Jupyter notebook.
# conda activate /Users/oluwaseunfadugba/mambaforge/envs/pygmt
'''
current_dir = '/Users/oluwaseunfadugba/Documents/Projects/Singularity_SW4/figures/map/'

os.chdir(current_dir)


#%% Extracting GNSS locations
gflist_filename = '/Users/oluwaseunfadugba/Documents/Projects/TsE_ValerieDiego/'+\
    'TsE_1D_vs_3D/1D_vs_3D_HR-GNSS_CrustalDeformation/GNSS_locations/'+\
        'gflists/ALL_events.GNSS_locs.txt'       

grid_E = [34,47,134.1125,147] # [lat1,lat2,lon1,lon2]
grid_W = [30,37.9,129,141.1] # [lat1,lat2,lon1,lon2]

# -----------------------------------------------------------
# Read in the metadata file
stadata = pd.read_csv(gflist_filename, sep='\t', header=0,\
                      names=['st_name', 'lat', 'lon', 'elev','sr','gain','unit'])

# Create a Polygon with the coordinates of the East and West 3D velocity model domains
coords_E = [(grid_E[2], grid_E[0]), (grid_E[3], grid_E[0]), 
            (grid_E[3], grid_E[1]), (grid_E[2], grid_E[1])]
coords_W = [(grid_W[2], grid_W[0]), (grid_W[3], grid_W[0]), 
            (grid_W[3], grid_W[1]), (grid_W[2], grid_W[1])]

poly_E = Polygon(coords_E)
poly_W = Polygon(coords_W)

# Initialize array
lon = []; lat = []; elev = []
_lon = []; _lat = []; _elev = []

#print(stadata)
for i in range(len(stadata.index)-1):
    
    lat_pt = float(stadata['lat'][i+1])
    lon_pt = float(stadata['lon'][i+1])
    elev_pt = float(stadata['elev'][i+1])
    
    # Check if the station is within the polygons using the within function
    p1 = Point(lon_pt, lat_pt)
    
    if p1.within(poly_E) == True or p1.within(poly_W) == True:
        lon.append(lon_pt)
        lat.append(lat_pt)
        elev.append(elev_pt)

    else:        
        _lon.append(lon_pt)
        _lat.append(lat_pt)
        _elev.append(elev_pt)
        
print(len(stadata['lon']))
print(len(lon))
print(len(_lon))


#%% PYGMT Figure (Slant rfile- All earthquakes)
# fig = pygmt.Figure()

# # Set the region for the plot to be slightly larger than the data bounds.
# region = [126,150.5,27.5,48]
 
# with pygmt.clib.Session() as session:
#     session.call_module('gmtset', 'FONT 15.3p')
#     session.call_module('gmtset', 'MAP_FRAME_TYPE fancy')
#     session.call_module('gmtset', 'MAP_FRAME_WIDTH 0.25')
    
# grid = pygmt.datasets.load_earth_relief(resolution="15s", region=region)
 
# fig.coast(region=region,projection="M15c",land="gray",water="lightblue",borders="1/0.5p",
#     shorelines="1/0.5p,black",frame="ag")

# fig.grdimage(grid=grid, projection="M15c", frame="ag",cmap="geo")
# fig.basemap(frame=["a", '+t"."'])

# #fig.colorbar(frame=["a2000", "x+lElevation", "y+lkm"],scale=1)
# fig.colorbar(frame=["x+lElevation", "y+lkm"],scale=0.001)#cmap="geo", 



# #fig.grdcontour(annotation=1000,interval=500,grid=grid,pen = "0.2p",limit=[-9000, 2000])

# # Plotting the west and east Japan 3D velocity boundaries
# east_x= [];  east_x.extend(np.linspace(134.1125,147,4)); 

# def cs(x1,x2,x3,x4,x5):
#     num = 7
#     return np.concatenate((np.linspace(x1,x2,num), 
#                           np.linspace(x2,x3,num),
#                           np.linspace(x3,x4,num),
#                           np.linspace(x4,x5,num)))

# fig.plot(x=cs(134.1125, 147, 147, 134.1125, 134.1125),y=cs(34, 34, 47, 47, 34),label='East_3DVel_Domain',pen="3p,cyan,-.")
# fig.plot(x=cs(129, 141.1, 141.1, 129, 129),y=cs(30, 30, 37.9, 37.9, 30),label='West_3DVel_Domain',pen="3p,green,-")
# fig.plot(x=cs(129, 147, 147, 129, 129),y=cs(30, 30, 47, 47, 30),label='All_3DVelPDomain',pen= "3p,blue")





# # plot ruptures
# pygmt.makecpt(cmap="viridis", series=[0,25])

# homerupt = '/Users/oluwaseunfadugba/Documents/Projects/TsE_ValerieDiego/TsE_1D_vs_3D/1D_Modeling_using_FQs_Mudpy/Running_FakeQuakes_now/'
# #fig.basemap(frame=["a", '+t"."'])

# rupt_no =["5","0","0","0"]
# simul = ["ibaraki2011_srcmod","iwate2011_zheng1","miyagi2011a_zheng1","tokachi2003_usgs"]

# for i in range(len(simul)):
#     slipfile = homerupt+simul[i] +'/ruptures/'+ simul[i]+'.00000'+rupt_no[i]+'.gmt'    
#     fig.plot(data = slipfile, color='+z',cmap = True)

# # pygmt.config(MAP_TICK_PEN="white",MAP_DEFAULT_PEN="white",MAP_GRID_PEN="white",MAP_FRAME_PEN="white",
# #              MAP_TICK_PEN_SECONDARY="white",MAP_GRID_PEN_PRIMARY="white",MAP_GRID_PEN_SECONDARY="white",
# #              MAP_TICK_PEN_PRIMARY="white")

# #pygmt.config(MAP_FRAME_PEN="white",MAP_TICK_PEN_SECONDARY="white")

# fig.colorbar(
#     cmap="viridis",
#     # Colorbar positioned at map coordinates (g) longitude/latitude 0.3/8.7,
#     # with a length/width (+w) of 4 cm by 0.5 cm, and plotted horizontally (+h)
#     position="g134./30.+w5c/0.3c+h",
#     box=True,
#     frame=["x+lSlip(m)"],
#     scale=25,
# )

# s="Slip (m)"
# #fig.colorbar(frame='af+l"'+s+'"',position="JMR+o1c/0c+w7c/0.5c",)
# #fig.colorbar(frame=["a2000", "x+lSlip(m)", "y+lkm"],scale=1)

# # Plotting the earthquake locations
# fig.plot(x=141.2653,y=36.1083,style="a0.7", color="red",pen="1p,black") ; 
# fig.text(x=141.2653,y=36.1083, text='                            Ibaraki 2011', font="16p,Helvetica-Bold,white")
# fig.plot(x=142.7815,y=39.8390,style="a0.7", color="red",pen="1p,black"); 
# fig.text(x=142.7815,y=39.8390,  text='                            Iwate 2011', font="16p,Helvetica-Bold,white")
# fig.plot(x=143.2798,y=38.3285,style="a0.7", color="red",pen="1p,black"); 
# fig.text(x=143.2798,y=38.6,  text='                              Miyagi 2011A', font="16p,Helvetica-Bold,white")
# #fig.plot(x=143.867,y=38.018,style="a0.7", color="red",pen="1p,black") ; 
# #fig.text(x=143.0,y=37.2,     text='                               N.Honshu 2012', font="16p,Helvetica-Bold,white")
# fig.plot(x=143.9040,y=41.7750,style="a0.7", color="red",pen="1p,black"); 
# fig.text(x=143.9040,y=41.7750,  text='                            Tokachi 2003', font="16p,Helvetica-Bold,white")

# fig.plot(x=lon,y=lat,style="t0.3",color="blue",pen="0.9p,black",transparency=50,label='GNSS_Stations')



# #fig.legend()
# fig.show()
# fig.savefig(current_dir+"/fig.model_setup.png")



#%% Functions for Pygmt Slant rfile for each earthquake (<=1000 km)

def cs(x1,x2,x3,x4,x5):
    num = 7
    return np.concatenate((np.linspace(x1,x2,num), 
                          np.linspace(x2,x3,num),
                          np.linspace(x3,x4,num),
                          np.linspace(x4,x5,num)))

def plot_slantrfile_sw4_geometry(lat0,lon0,xdim,ydim,evlon,evlat,rupt_5,flatfile_res_path,dist_thresh,eqname,outputfilename):
    from math import cos, sin,pi

    az = 35*pi/180
    R = np.array([[cos(az), -sin(az)],[ sin(az), cos(az)]])
    R_cl = np.array([[cos(az), sin(az)],[ -sin(az), cos(az)]])

    #
    fig = pygmt.Figure()

    # Set the region for the plot to be slightly larger than the data bounds.
    region = [126,150.5,27.5,48]
     
    with pygmt.clib.Session() as session:
        session.call_module('gmtset', 'FONT 15.3p')
        session.call_module('gmtset', 'MAP_FRAME_TYPE fancy')
        session.call_module('gmtset', 'MAP_FRAME_WIDTH 0.25')
        
    grid = pygmt.datasets.load_earth_relief(resolution="15s", region=region)
     
    fig.coast(region=region,projection="M15c",land="gray",water="lightblue",borders="1/0.5p",
        shorelines="1/0.5p,black",frame="ag")

    fig.grdimage(grid=grid, projection="M15c", frame="ag",cmap="geo")
    fig.basemap(frame=["a", '+t"."'])

    #fig.grdcontour(annotation=1000,interval=500,grid=grid,pen = "0.2p",limit=[-9000, 2000])

    # Plotting the west and east Japan 3D velocity boundaries
    east_x= [];  east_x.extend(np.linspace(134.1125,147,4)); 

    fig.plot(x=cs(134.1125, 147, 147, 134.1125, 134.1125),y=cs(34, 34, 47, 47, 34),label='East_3DVel_Domain',pen="3p,cyan,-.")
    fig.plot(x=cs(129, 141.1, 141.1, 129, 129),y=cs(30, 30, 37.9, 37.9, 30),label='West_3DVel_Domain',pen="3p,green,-")
    fig.plot(x=cs(129, 147, 147, 129, 129),y=cs(30, 30, 47, 47, 30),label='All_3DVelPDomain',pen= "3p,blue")

    #  Plotting  slant rfile
    # Slant lower left corner:
    lat0_sl =  31.8341; lon0_sl =  126.2932
    xdim_sl = 870; ydim_sl= 2100

    # Rotated geometry
    edges = np.array([[0,xdim_sl,xdim_sl, 0],[0,0,ydim_sl, ydim_sl]])
    edges_r1 = np.matmul(R_cl,edges)

    # converting slant edges to lat and lon
    edges_r1[0,:] = edges_r1[0,:]*0.0125+lon0_sl
    edges_r1[1,:] = edges_r1[1,:]*0.00833+lat0_sl
    sl = edges_r1

    fig.plot(x=cs(sl[0,0],  sl[0,1],  sl[0,2],  sl[0,3],sl[0,0]),
              y=cs(sl[1,0],  sl[1,1],  sl[1,2],  sl[1,3],sl[1,0]),label='Slant_3DVelPDomain',pen= "3p,black")
    

    # Rotated geometry
    edges = np.array([[0,xdim,xdim, 0],[0,0,ydim, ydim]])
    edges_r2 = np.matmul(R_cl,edges)

    # converting slant edges to lat and lon
    edges_r2[0,:] = edges_r2[0,:]*0.0125+lon0
    edges_r2[1,:] = edges_r2[1,:]*0.00833+lat0
    sl = edges_r2

    stadata = pd.read_csv(flatfile_res_path) 
    stadata = stadata[stadata['rupt_no']==1]
    stadata = stadata[stadata['hypdist']<=dist_thresh].reset_index()
    
    print('')
    print('Total Number of stations:',len(stadata['stlat']))
    
    #grid_E = [34,47,134.1125,147]
    # Create a Polygon with the coordinates of the East and West 3D velocity model domains
    coords_domain = [(sl[0,0], sl[1,0]), (sl[0,1], sl[1,1]), 
                    (sl[0,2], sl[1,2]), (sl[0,3], sl[1,3])]
    poly_coords_domain = Polygon(coords_domain)
    
    lon_d = []
    lat_d = []
    
    for jj in range(len(stadata['stlon'])):
        # Check if the station is within the polygons using the within function
        lon_pt = float(stadata['stlon'][jj])
        lat_pt = float(stadata['stlat'][jj])
        
        p1_d = Point(lon_pt,lat_pt)
        
        if p1_d.within(poly_coords_domain) == True:
            lon_d.append(lon_pt)
            lat_d.append(lat_pt)
        
    print('Number of stations within the domain:',len(lon_d))
    
    fig.plot(x=stadata['stlon'],y=stadata['stlat'],style="t0.3",color="grey",
             pen="0.9p,black",transparency=0,label='GNSS_Stations')

    fig.plot(x=lon_d,y=lat_d,style="t0.3",color="blue",
             pen="0.9p,black",transparency=50,label='GNSS_Stations')


    fig.plot(x=cs(sl[0,0],  sl[0,1],  sl[0,2],  sl[0,3],sl[0,0]),
              y=cs(sl[1,0],  sl[1,1],  sl[1,2],  sl[1,3],sl[1,0]),
              label='Slant_3DVelPDomain',pen= "3p,red")
    
    
    # plot ruptures
    [lon_s,lat_s,depth_s] = extract_source_pts(rupt_5); 
    fig.plot(x=lon_s,y=lat_s,style="c0.1",color="magenta")
    # Plotting the earthquake locations
    fig.plot(x=evlon,y=evlat,style="a0.7", color="red",pen="1p,black") ; 
    fig.text(x=evlon,y=evlat, text='                            '+eqname, 
             font="16p,Helvetica-Bold,white")

    
    
    #fig.plot(x=cs(130, 145.5, 145.5, 130, 130),y=cs(32, 32, 45, 45, 32),
    #label='All_3DVelPDomain',pen= "3p,blue,-")
 
    fig.show()
    fig.savefig(current_dir+"/"+outputfilename)




#%% Plotting maps using Pygmt Slant rfile for each earthquake (<=1000 km)
# Plotting slant rfile and sw4 geometry for Ibaraki2011
# Slant lower left corner:
lat0 =  34.5; lon0 = 133.1
xdim = 600; ydim= 900 #1400 1750 #1750
evlon=141.2653; evlat=36.1083
rupt_5 = '/Users/oluwaseunfadugba/Documents/Projects/TsE_ValerieDiego/TsE_1D_vs_3D/'+\
    '1D_Modeling_using_FQs_Mudpy/Running_FakeQuakes_now/ibaraki2011_srcmod/ruptures/ibaraki2011_srcmod.000005.rupt'
flatfile_res_path='/Users/oluwaseunfadugba/Documents/Projects/TsE_ValerieDiego/'+\
    'TsE_1D_vs_3D/1D_Modeling_using_FQs_Mudpy/IM_Residuals/Results_ibaraki2011_srcmod_IM_residuals/Flatfiles_IMs_ibaraki2011_srcmod.csv'    
dist_thresh = 10000
eqname = 'Ibaraki 2011'
#outputfilename= 'fig.map_GNSS_locations_slant_ibaraki2_1000km.png'
outputfilename= 'fig.model_setup.png'

plot_slantrfile_sw4_geometry(lat0,lon0,xdim,ydim,evlon,evlat,rupt_5,flatfile_res_path,dist_thresh,eqname,outputfilename)



#%% Other earthquakes
# # Plotting slant rfile and sw4 geometry for Ibaraki2011
# # Slant lower left corner:
# lat0 =  33.2; lon0 = 129.5
# xdim = 760; ydim= 1750 #1400 1750 #1750
# evlon=141.2653; evlat=36.1083
# rupt_5 = '/Users/oluwaseunfadugba/Documents/Projects/TsE_ValerieDiego/TsE_1D_vs_3D/'+\
#     '1D_Modeling_using_FQs_Mudpy/Running_FakeQuakes_now/ibaraki2011_srcmod/ruptures/ibaraki2011_srcmod.000005.rupt'
# flatfile_res_path='/Users/oluwaseunfadugba/Documents/Projects/TsE_ValerieDiego/'+\
#     'TsE_1D_vs_3D/1D_Modeling_using_FQs_Mudpy/IM_Residuals/Results_ibaraki2011_srcmod_IM_residuals/Flatfiles_IMs_ibaraki2011_srcmod.csv'    
# dist_thresh = 10000
# eqname = 'Ibaraki 2011'
# #outputfilename= 'fig.map_GNSS_locations_slant_ibaraki2_1000km.png'
# outputfilename= 'fig.map_GNSS_locations_full_ibaraki2_1000km.png'

# plot_slantrfile_sw4_geometry(lat0,lon0,xdim,ydim,evlon,evlat,rupt_5,flatfile_res_path,dist_thresh,eqname,outputfilename)



# # -----------------------------------------------------------------------
# # #  Plotting slant rfile and sw4 geometry for Iwate 
# # # Slant lower left corner:
# # lat0 =  35; lon0 = 131.4
# # xdim = 610; ydim= 1520 #1750

# # evlon=142.7815; evlat=39.8390
# # rupt_5 = '/Users/oluwaseunfadugba/Documents/Projects/TsE_ValerieDiego/TsE_1D_vs_3D/'+\
# #     '1D_Modeling_using_FQs_Mudpy/Running_FakeQuakes_now/iwate2011_zheng1/ruptures/iwate2011_zheng1.000000.rupt'
# # flatfile_res_path='/Users/oluwaseunfadugba/Documents/Projects/TsE_ValerieDiego/TsE_1D_vs_3D/'+\
# #     '1D_Modeling_using_FQs_Mudpy/IM_Residuals/Results_iwate2011_zheng1_IM_residuals/Flatfiles_IMs_iwate2011_zheng1.csv'    
# # dist_thresh = 1000
# # eqname = 'Iwate 2011'
# # outputfilename= 'fig.map_GNSS_locations_slant_iwate_1000km.png'

# # plot_slantrfile_sw4_geometry(lat0,lon0,xdim,ydim,evlon,evlat,rupt_5,flatfile_res_path,dist_thresh,eqname,outputfilename)


# # -----------------------------------------------------------------------

# # #  Plotting slant rfile and sw4 geometry for Miyagi
# # # Slant lower left corner:
# # lat0 =  34.7; lon0 = 131.
# # xdim = 700; ydim= 1520 #1750

# # evlon=143.2798; evlat=38.3285
# # rupt_5 = '/Users/oluwaseunfadugba/Documents/Projects/TsE_ValerieDiego/TsE_1D_vs_3D/'+\
# #     '1D_Modeling_using_FQs_Mudpy/Running_FakeQuakes_now/miyagi2011a_zheng1/ruptures/miyagi2011a_zheng1.000000.rupt'
# # flatfile_res_path='/Users/oluwaseunfadugba/Documents/Projects/TsE_ValerieDiego/TsE_1D_vs_3D/'+\
# #     '1D_Modeling_using_FQs_Mudpy/IM_Residuals/Results_miyagi2011a_zheng1_IM_residuals/Flatfiles_IMs_miyagi2011a_zheng1.csv'    
# # dist_thresh = 1000
# # eqname = 'Miyagi 2011A'
# # outputfilename= 'fig.map_GNSS_locations_slant_miyagi2011a_1000km.png'

# # plot_slantrfile_sw4_geometry(lat0,lon0,xdim,ydim,evlon,evlat,rupt_5,flatfile_res_path,dist_thresh,eqname,outputfilename)


# # -----------------------------------------------------------------------

# # #  Plotting slant rfile and sw4 geometry for Tokachi 2003
# # # Slant lower left corner:
# # lat0 =  38.8; lon0 = 133.6
# # xdim = 620; ydim= 1080 #1750

# # evlon=143.9040; evlat=41.7750
# # rupt_5 = '/Users/oluwaseunfadugba/Documents/Projects/TsE_ValerieDiego/TsE_1D_vs_3D/'+\
# #     '1D_Modeling_using_FQs_Mudpy/Running_FakeQuakes_now/tokachi2003_usgs/ruptures/tokachi2003_usgs.000000.rupt'
# # flatfile_res_path='/Users/oluwaseunfadugba/Documents/Projects/TsE_ValerieDiego/TsE_1D_vs_3D/'+\
# #     '1D_Modeling_using_FQs_Mudpy/IM_Residuals/Results_tokachi2003_usgs_IM_residuals/Flatfiles_IMs_tokachi2003_usgs.csv'    
# # dist_thresh = 1000
# # eqname = 'Tokachi 2003'
# # outputfilename= 'fig.map_GNSS_locations_slant_Tokachi2003_1000km.png'

# # plot_slantrfile_sw4_geometry(lat0,lon0,xdim,ydim,evlon,evlat,rupt_5,flatfile_res_path,dist_thresh,eqname,outputfilename)































# #%%  Plotting slant rfile and sw4 geometry for N.Honshu 
# # # Slant lower left corner:
# # lat0 =  35; lon0 = 131.4
# # xdim = 740; ydim= 1520 #1750

# # evlon=144.3153; evlat=37.8158
# # rupt_5 = '/Users/oluwaseunfadugba/Documents/Projects/TsE_ValerieDiego/TsE_1D_vs_3D/'+\
# #     '1D_Modeling_using_FQs_Mudpy/Running_FakeQuakes_now/n.honshu2012_zheng1/ruptures/n.honshu2012_zheng1.000009.rupt'
# # flatfile_res_path='/Users/oluwaseunfadugba/Documents/Projects/TsE_ValerieDiego/TsE_1D_vs_3D/'+\
# #     '1D_Modeling_using_FQs_Mudpy/IM_Residuals/Results_n.honshu2012_zheng1_IM_residuals/Flatfiles_IMs_n.honshu2012_zheng1.csv'    
# # dist_thresh = 1000
# # eqname = 'N.Honshu 2012'
# # outputfilename= 'fig.map_GNSS_locations_slant_nhonshu2012_1000km.png'

# # plot_slantrfile_sw4_geometry(lat0,lon0,xdim,ydim,evlon,evlat,rupt_5,flatfile_res_path,dist_thresh,eqname,outputfilename)




# #%% PYGMT Slant rfile Ibaraki only
# # fig = pygmt.Figure()

# # # Set the region for the plot to be slightly larger than the data bounds.
# # region = [126,150.5,27.5,48]
 
# # with pygmt.clib.Session() as session:
# #     session.call_module('gmtset', 'FONT 15.3p')
# #     session.call_module('gmtset', 'MAP_FRAME_TYPE fancy')
# #     session.call_module('gmtset', 'MAP_FRAME_WIDTH 0.25')
    
# # grid = pygmt.datasets.load_earth_relief(resolution="10m", region=region)
 
# # fig.coast(region=region,projection="M15c",land="gray",water="lightblue",borders="1/0.5p",
# #     shorelines="1/0.5p,black",frame="ag")

# # fig.grdimage(grid=grid, projection="M15c", frame="ag",cmap="geo")
# # fig.basemap(frame=["a", '+t"."'])

# # fig.grdcontour(annotation=1000,interval=500,grid=grid,pen = "0.2p",limit=[-9000, 2000])

# # # Plotting the west and east Japan 3D velocity boundaries

# # east_x= [];  east_x.extend(np.linspace(134.1125,147,4)); 


# # def cs(x1,x2,x3,x4,x5):
# #     num = 7
# #     return np.concatenate((np.linspace(x1,x2,num), 
# #                           np.linspace(x2,x3,num),
# #                           np.linspace(x3,x4,num),
# #                           np.linspace(x4,x5,num)))

# # from math import cos, sin,pi

# # az = 35*pi/180
# # R = np.array([[cos(az), -sin(az)],[ sin(az), cos(az)]])
# # R_cl = np.array([[cos(az), sin(az)],[ -sin(az), cos(az)]])

# # # Original geometry
# # lat0 = 27;  lon0= 126



# # # A_x = 0;    A_y = 0 # in km not px (km= px-1)
# # # B_x = 1920; B_y = 0
# # # C_x = 1920; C_y = 2641
# # # D_x = 0;    D_y = 2641
# # A_x = 0;    A_y = 0 # in km not px (km= px-1)
# # B_x = 2000; B_y = 0
# # C_x = 2000; C_y = 2761
# # D_x = 0;    D_y = 2761




# # edges = np.array([[A_x,B_x,C_x, D_x],[A_y,B_y,C_y, D_y]])

# # # Rotated geometry
# # edges_r = np.matmul(R,edges)

# # D_rx = edges_r[0,3]
# # C_ry = edges_r[1,2]

# # # Geometry of the cropped image





# # x0 = 1270 #1280
# # y0 = 820 # 760 from the top




# # xdim = 870 #858 #760
# # ydim = 2100

# # P_x = D_rx + x0
# # Q_x = D_rx + x0 + xdim
# # R_x = D_rx + x0 + xdim
# # S_x = D_rx + x0

# # P_y = C_ry - y0 - ydim
# # Q_y = C_ry - y0 - ydim 
# # R_y = C_ry - y0 
# # S_y = C_ry - y0

# # edges_sl = np.array([[P_x,Q_x,R_x, S_x],[P_y,Q_y,R_y, S_y]])

# # # Rotating the slant edges back to the original geometry
# # slant_edges = np.matmul(R_cl,edges_sl)


# # #print(slant_edges)

# # # converting slant edges to lat and lon
# # slant_edges_deg = slant_edges[:]
# # slant_edges_deg[0,:] = slant_edges[0,:]*0.0125+lon0
# # slant_edges_deg[1,:] = slant_edges[1,:]*0.00833+lat0

# # sx = slant_edges_deg[0,:]
# # sy = slant_edges_deg[1,:]

# # fig.plot(x=cs(134.1125, 147, 147, 134.1125, 134.1125),y=cs(34, 34, 47, 47, 34),label='East_3DVel_Domain',pen="3p,cyan,-.")
# # fig.plot(x=cs(129, 141.1, 141.1, 129, 129),y=cs(30, 30, 37.9, 37.9, 30),label='West_3DVel_Domain',pen="3p,green,-")
# # fig.plot(x=cs(129, 147, 147, 129, 129),y=cs(30, 30, 47, 47, 30),label='All_3DVelPDomain',pen= "3p,blue")
# # #fig.plot(x=cs(127.92701572,135.70896014,150.33515926,142.55321484,127.92701572),y=cs(32.29246027,28.66126257,42.58127709,46.21247479,32.29246027),label='Slant_3DVelPDomain',pen= "3p,red")
# # fig.plot(x=cs(sx[0],  sx[1],  sx[2],  sx[3],sx[0]),
# #          y=cs(sy[0],  sy[1],  sy[2],  sy[3],sy[0]),label='Slant_3DVelPDomain',pen= "3p,red")


# # print('')
# # print('Slant rfile corners:')
# # print(slant_edges_deg)

# # print('')
# # print('Slant lower left corner:')
# # print('lat0 = ',round(sy[0],4),'lon0 = ',round(sx[0],4))


# # # plot ruptures
# # # homerupt = '/Users/oluwaseunfadugba/Documents/Projects/TsE_ValerieDiego/TsE_1D_vs_3D/1D_Modeling_using_FQs_Mudpy/Creating_Inputs/Finite_Fault_Models/rupt_on_mesh/'
# # # [lon_s,lat_s,depth_s] = extract_source_pts(homerupt+'ibaraki2011_srcmod/ibaraki2011_srcmod_mesh.rupt'); 
# # # fig.plot(x=lon_s,y=lat_s,style="c0.1",color="magenta")

# # homerupt = '/Users/oluwaseunfadugba/Documents/Projects/TsE_ValerieDiego/TsE_1D_vs_3D/1D_Modeling_using_FQs_Mudpy/Running_FakeQuakes_now/ibaraki2011_srcmod/ruptures/ibaraki2011_srcmod.000005.rupt'
# # [lon_s,lat_s,depth_s] = extract_source_pts(homerupt); 
# # fig.plot(x=lon_s,y=lat_s,style="c0.1",color="magenta")

# # # Plotting the earthquake locations
# # fig.plot(x=141.2653,y=36.1083,style="a0.7", color="red",pen="1p,black") ; 
# # fig.text(x=141.2653,y=36.1083, text='                            Ibaraki 2011', font="16p,Helvetica-Bold,white")

# # ibk_gflist_filename = '/Users/oluwaseunfadugba/Documents/Projects/TsE_ValerieDiego/TsE_1D_vs_3D/1D_Modeling_using_FQs_Mudpy/Running_FakeQuakes_now/ibaraki2011_srcmod/ibaraki2011.gflist'
# # stadata = pd.read_csv(ibk_gflist_filename, sep='\t', header=0,\
# #                       names=['st_name', 'lon', 'lat', 'others'])
    
    
# # fig.plot(x=stadata['lon'],y=stadata['lat'],style="t0.3",color="blue",pen="0.9p,black",transparency=50,label='GNSS_Stations')

# # fig.colorbar(frame=["a2000", "x+lElevation", "y+lkm"],scale=1)
# # #fig.legend()
# # fig.show()
# # fig.savefig(current_dir+"/fig.map_GNSS_locations_slant_ibaraki2.png")




# #%% PYGMT Figure
# # fig = pygmt.Figure()

# # # Set the region for the plot to be slightly larger than the data bounds.
# # region = [127,150,29,48]
 
# # with pygmt.clib.Session() as session:
# #     session.call_module('gmtset', 'FONT 15.3p')
# #     session.call_module('gmtset', 'MAP_FRAME_TYPE fancy')
# #     session.call_module('gmtset', 'MAP_FRAME_WIDTH 0.25')
    
# # grid = pygmt.datasets.load_earth_relief(resolution="10m", region=region)
 
# # fig.coast(region=region,projection="M15c",land="gray",water="lightblue",borders="1/0.5p",
# #     shorelines="1/0.5p,black",frame="ag")

# # fig.grdimage(grid=grid, projection="M15c", frame="ag",cmap="geo")
# # fig.basemap(frame=["a", '+t"."'])

# # fig.grdcontour(annotation=1000,interval=500,grid=grid,pen = "0.2p",limit=[-9000, 2000])

# # # Plotting the west and east Japan 3D velocity boundaries

# # east_x= [];  east_x.extend(np.linspace(134.1125,147,4)); 


# # def cs(x1,x2,x3,x4,x5):
# #     num = 7
# #     return np.concatenate((np.linspace(x1,x2,num), 
# #                           np.linspace(x2,x3,num),
# #                           np.linspace(x3,x4,num),
# #                           np.linspace(x4,x5,num)))

# # fig.plot(x=cs(134.1125, 147, 147, 134.1125, 134.1125),y=cs(34, 34, 47, 47, 34),label='East_3DVel_Domain',pen="3p,cyan,-.")
# # fig.plot(x=cs(129, 141.1, 141.1, 129, 129),y=cs(30, 30, 37.9, 37.9, 30),label='West_3DVel_Domain',pen="3p,green,-")
# # fig.plot(x=cs(129, 147, 147, 129, 129),y=cs(30, 30, 47, 47, 30),label='All_3DVelPDomain',pen= "3p,blue")


# # # fig.plot(x=[134.1125, 147, 147, 134.1125, 134.1125],y=[34, 34, 47, 47, 34],label='East_3DVel_Domain',pen="3p,cyan,-")
# # # fig.plot(x=[129, 141.1, 141.1, 129, 129],y=[30, 30, 37.9, 37.9, 30],label='West_3DVel_Domain',pen="3p,green,-")
# # # fig.plot(x=[129, 147, 147, 129, 129],y=[30, 30, 47, 47, 30],label='All_3DVelPDomain',pen= "3p,blue,-")





# # # # Plot supergrid boundary
# # # lat_sup = 28;    lon_sup = 126
# # # grid_x = 2400e3; grid_xd = grid_x/120e3
# # # grid_y = 2000e3; grid_yd = grid_y/80e3
# # # h= 3000;          sgrid_xd = h*30/120e3; sgrid_yd = h*30/80e3

# # # fig.plot(x=[lon_sup, lon_sup+grid_yd, lon_sup+grid_yd, lon_sup, lon_sup],\
# # #          y=[lat_sup, lat_sup, lat_sup+grid_xd, lat_sup+grid_xd, lat_sup],pen= "3p,red,-")
    
# # # fig.plot(x=[lon_sup+sgrid_yd, lon_sup+grid_yd-sgrid_yd, lon_sup+grid_yd-sgrid_yd, lon_sup+sgrid_yd, lon_sup+sgrid_yd],\
# # #          y=[lat_sup+sgrid_xd, lat_sup+sgrid_xd, lat_sup+grid_xd-sgrid_xd, lat_sup+grid_xd-sgrid_xd, lat_sup+sgrid_xd], \
# # #          pen= "3p,red,-")    
    
# # # plot ruptures
# # homerupt = '/Users/oluwaseunfadugba/Documents/Projects/TsE_ValerieDiego/TsE_1D_vs_3D/1D_Modeling_using_FQs_Mudpy/Creating_Inputs/Finite_Fault_Models/rupt_on_mesh/'
# # [lon_s,lat_s,depth_s] = extract_source_pts(homerupt+'ibaraki2011_srcmod/ibaraki2011_srcmod_mesh.rupt'); 
# # fig.plot(x=lon_s,y=lat_s,style="c0.1",color="magenta")
# # [lon_s,lat_s,depth_s] = extract_source_pts(homerupt+'iwate2011_zheng1/iwate2011_zheng1_mesh.rupt'); 
# # fig.plot(x=lon_s,y=lat_s,style="c0.1",color="magenta")
# # [lon_s,lat_s,depth_s] = extract_source_pts(homerupt+'miyagi2011a_zheng1/miyagi2011a_zheng1_mesh.rupt'); 
# # fig.plot(x=lon_s,y=lat_s,style="c0.1",color="magenta")
# # [lon_s,lat_s,depth_s] = extract_source_pts(homerupt+'n.Honshu2012_zheng1/n.Honshu2012_zheng1_mesh.rupt'); 
# # fig.plot(x=lon_s,y=lat_s,style="c0.1",color="magenta")
# # [lon_s,lat_s,depth_s] = extract_source_pts(homerupt+'tokachi2003_usgs/tokachi2003_usgs_mesh.rupt'); 
# # fig.plot(x=lon_s,y=lat_s,style="c0.1",color="magenta")

# # # Plotting the earthquake locations
# # fig.plot(x=141.2653,y=36.1083,style="a0.7", color="red",pen="1p,black") ; 
# # fig.text(x=141.2653,y=36.1083, text='                            Ibaraki 2011', font="16p,Helvetica-Bold,white")
# # fig.plot(x=142.7815,y=39.8390,style="a0.7", color="red",pen="1p,black"); 
# # fig.text(x=142.7815,y=39.8390,  text='                            Iwate 2011', font="16p,Helvetica-Bold,white")
# # fig.plot(x=143.2798,y=38.3285,style="a0.7", color="red",pen="1p,black"); 
# # fig.text(x=143.2798,y=38.6,  text='                              Miyagi 2011A', font="16p,Helvetica-Bold,white")
# # fig.plot(x=143.867,y=38.018,style="a0.7", color="red",pen="1p,black") ; 
# # fig.text(x=143.0,y=37.2,     text='                               N.Honshu 2012', font="16p,Helvetica-Bold,white")
# # fig.plot(x=143.9040,y=41.7750,style="a0.7", color="red",pen="1p,black"); 
# # fig.text(x=143.9040,y=41.7750,  text='                            Tokachi 2003', font="16p,Helvetica-Bold,white")

# # fig.plot(x=lon,y=lat,style="t0.3",color="blue",pen="0.9p,black",transparency=50,label='GNSS_Stations')

# # fig.colorbar(frame=["a2000", "x+lElevation", "y+lkm"],scale=1)
# # #fig.legend()
# # fig.show()
# # fig.savefig(current_dir+"/fig.map_GNSS_locations.png")
