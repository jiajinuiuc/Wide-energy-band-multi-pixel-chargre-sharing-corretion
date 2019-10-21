# -*- coding: utf-8 -*-
"""
Objective： 
Full range energy spectrum correction try with 3-pixel charge sharing events
0. charge sharing events clustering on 60 keV(Am), 80.99 keV(Ba), 122 keV + 136 keV(Co)
1. full "spatial" range segmentation and calculate the projection distance in each channel
3. calculate the projection distance of each charge sharing band at each channel
4. linear interpolation of the porjection distance between each band at different channel
5. based on the linear interpolation results, do the full range charge sharing correction
@author: J. J. Zhang
Last update: August, 2019
"""

import sys
sys.path.append('C:\Jiajin\Mfile\Training_Sample_Analysis')

from charge_sharing_correction_v2 import charge_sharing_correction as CSC # class file, charge sharing correction
from charge_sharing_correction_v2 import Common_used_function as CF # class file, common-used plotting tools
from charge_sharing_correction_v2 import SG_Filter as SG # class file, S-G filters of different dimentions
from charge_sharing_correction_v2 import Map_generation as MG # class file, compensation map generation
from charge_sharing_correction_v2 import scattering_clustering as SC # class file, DBSCAN clustering


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.datasets as ds
import matplotlib.colors
from sklearn.cluster import DBSCAN
from mpl_toolkits.mplot3d import Axes3D
from functools import reduce
from scipy import signal
import heapq
import gc

%matplotlib qt5



################################### initialize the transformation parameters #############################
### Initialize basis parameters
basis_old =np.mat( [ [1,0,0],
                     [0,1,0],
                     [0,0,1] ] ) #[x, y, z]

basis_new = np.mat( [ [1/np.sqrt(6),  1/np.sqrt(2), 1/np.sqrt(3)],
                      [1/np.sqrt(6), -1/np.sqrt(2), 1/np.sqrt(3)],
                      [-2/np.sqrt(6), 0,            1/np.sqrt(3)] ] )



seg_size = 0.04
########################################################################################################



################################### Load the 60-keV charge sharing events #############################
CS_data = pd.read_csv( 'C:\Jiajin\ChaSha_2017\CSC_Data\CSC_extracted_data\Am_3Pix_new.csv' )
Energy = CS_data.iloc[:, :].values  
#Energy = Energy - 1
#CF.SaveFiles( var=Energy, var_name=['E1', 'E2', 'E3'], location='C:\Jiajin\ChaSha_2017\CSC_Data\CSC_extracted_data\Am_3Pix_new.csv' )
CS_dim = Energy.shape[1]
del(CS_data)

Energy_sum = np.sum( Energy, axis=1 ).reshape(-1,1)
Energy_E1_E2 = Energy[ np.intersect1d(np.where(Energy_sum >= 50)[0],np.where(Energy_sum <= 70)[0]) ]
Energy_E1_E2_else = np.delete( Energy, np.intersect1d(np.where(Energy_sum >= 50)[0],np.where(Energy_sum <= 70)[0]), axis =0 )

##### plot the histogram of sum_Energy within the selected ROI
CF.Histogram_lineplot(Hist=Energy_sum, Bins=300, x_lim_high=100, x_lim_low=0, color='red')

##### plot the raw scatter figures within the selected ROI
fig = plt.figure(figsize=(12, 12), facecolor='w')
ax = Axes3D(fig)
CF.Scatter3D_plot(ax=ax, x=Energy_E1_E2[:,0], y=Energy_E1_E2[:,1], z=Energy_E1_E2[:,2], elev=45, azim=45)
########################################################################################################



####################### DBSCAN Clustering and Plot the results #########################################
Energy_E1_E2_sum = np.sum( Energy_E1_E2, axis=1 ).reshape(-1,1)

##### Model Fitting I        
# Set Fitting Parameters density=28
y_hat1, y_unique1, core_indices1 = SC.Clustering(radius=1, density=28, data=Energy_E1_E2)

##### Model Fitting II    
# Set Fitting Parameters
# Plot the DBSCAN clustering results
clrs = plt.cm.Spectral(np.linspace(0, 0.5, 2*y_unique1.size))
fig = plt.figure(figsize=(12, 12), facecolor='w')
ax = Axes3D(fig)
for k, clr in zip(y_unique1, clrs):
    cur = (y_hat1 == k)
    if k == -1:
        CF.Scatter3D_plot(ax, Energy_E1_E2[cur, 0], Energy_E1_E2[cur, 1], Energy_E1_E2[cur, 2], elev=45, azim=45, color='k', marker='.')
        continue
    CF.Scatter3D_plot(ax, Energy_E1_E2[cur, 0], Energy_E1_E2[cur, 1], Energy_E1_E2[cur, 2], elev=45, azim=45, color=clr, marker='.')

ax.set_xlabel('Energy (keV)')
ax.set_ylabel('Energy (keV)')
ax.set_zlabel('Energy (keV)')

plt.xlim(0, 65)
plt.ylim(0, 65)
plt.grid(True)

###### check each DBSCAN clustering results
clrs = plt.cm.Spectral(np.linspace(0, 0.5, 2*y_unique2.size))
fig = plt.figure(figsize=(12, 12), facecolor='w')
ax = Axes3D(fig)
for k, clr in zip(y_unique2, clrs):
    cur = (y_hat1 == k)
    if k == 0:
        CF.Scatter3D_plot(ax, Energy_E1_E2[cur, 0], Energy_E1_E2[cur, 1], Energy_E1_E2[cur, 2], elev=45, azim=45, color='k', marker='.')
        continue
    
ax.set_xlabel('Energy (keV)')
ax.set_ylabel('Energy (keV)')
ax.set_zlabel('Energy (keV)')

plt.xlim(0, 65)
plt.ylim(0, 65)
plt.grid(True)
########################################################################################################



####################### Extract the cluster in the ROI #################################################
##### Reorganize the clustered scattering points
Energy = np.vstack( (Energy_E1_E2, Energy_E1_E2_else) )
Energy_sum = np.sum( Energy, axis=1 ).reshape(-1,1)

y_hat = np.array([-1]*len(Energy))
y_hat[np.where( y_hat1 == 0 ),] = 0
    
y_unique = np.unique(y_hat)

cluster_60_lab_0 = Energy[np.where( ( y_hat == 0 ) )]
cluster_60_lab_noise = Energy[np.where( ( y_hat == -1 ) )]

# Plot the DBSCAN clustering results
clrs = plt.cm.Spectral(np.linspace(0, 0.5, 2*y_unique.size))
fig = plt.figure(figsize=(12, 12), facecolor='w')
ax = Axes3D(fig)
for k in y_unique:
    cur = (y_hat == k)
    if k == -1:
        CF.Scatter3D_plot(ax, Energy[cur, 0], Energy[cur, 1], Energy[cur, 2], elev=45, azim=45, color='k', marker='.')
        continue
    if k == 0:
        CF.Scatter3D_plot(ax, Energy[cur, 0], Energy[cur, 1], Energy[cur, 2], elev=45, azim=45, color='r', marker='.')
        continue

ax.set_xlabel('Energy (keV)')
ax.set_ylabel('Energy (keV)')
ax.set_zlabel('Energy (keV)')

plt.xlim(0, 65)
plt.ylim(0, 65)
plt.grid(True)

###### Save data Figure 5 122-keV, 2-pixel, scattering + DBSCAN scattering
data_save = np.hstack( (Energy, y_hat.reshape(-1,1)) ) ### ( E1, E2, label)
CF.SaveFiles(var=data_save, var_name=['E1', 'E2', 'E3', 'label'], \
              location="C:\Jiajin\ChaSha_2017\CSC_Data\CSC_extracted_data\Figure_data\Figure_1301_Am_scatter_labeled.csv")
del(data_save)
######################################################################################################## 



#######################      CSC --- parameters measurement      #######################################
##### Initialize the CSC object
CSC_60_3pix = CSC( CS_dim=CS_dim, basis_old=basis_old, basis_new=basis_new, peak_energy=60, max_energy_range=140, seg_size=seg_size )
##### Calculate the MC-curve
wet_x_60, wet_y_60, wet_w_60, shift_w_60, seg_unit_60, Energy_rotation = CSC_60_3pix.Pix3_Measurement( CS_data_labeled = Energy )
Energy_rotation = np.hstack( ( Energy_rotation[:,0], Energy_rotation[:,1], Energy_rotation[:,2] ) )

##### Calculate the rotated (Ratio, E-sum) points
wet_x_60, wet_y_60, wet_w_60, shift_w_60, seg_unit_60, Energy_rot = CSC_60_3pix.Pix3_Measurement( CS_data_labeled = cluster_60_lab_0 )

##### Check the scattering plot and MC plot
clrs = plt.cm.Spectral(np.linspace(0, 0.5, 2*y_unique.size))
fig = plt.figure(figsize=(12, 12), facecolor='w')
ax = Axes3D(fig)
for k in y_unique:
    cur = (y_hat == k)
    if k == -1:
        CF.Scatter3D_plot(ax, Energy_rotation[cur, 0], Energy_rotation[cur, 1], Energy_rotation[cur, 2], elev=45, azim=45, color='k', marker='.')
        continue
    if k == 0:
        CF.Scatter3D_plot(ax, Energy_rotation[cur, 0], Energy_rotation[cur, 1], Energy_rotation[cur, 2], elev=45, azim=45, color='r', marker='.')
        continue

ax.set_xlabel('Energy (keV)')
ax.set_ylabel('Energy (keV)')
ax.set_zlabel('Energy (keV)')

###### Save data Figure 5 122-keV, 2-pixel, scattering + DBSCAN scattering
data_save = np.hstack( (Energy_rotation, y_hat.reshape(-1,1)) ) ### ( E1, E2, label)
CF.SaveFiles(var=data_save, var_name=['E1', 'E2', 'E3', 'label'], \
              location="C:\Jiajin\ChaSha_2017\CSC_Data\CSC_extracted_data\Figure_data\Figure_1401_Am_transformed_scatter_labeled.csv")
del(data_save)
######################################################################################################## 




########################## filter the loss plane and compensation plane ################################ 
tmp = wet_w_60
tmp = tmp.reshape(-1,1)
boundary_mean = np.mean( heapq.nlargest(10, tmp) )
del tmp
gc.collect()
wet_w_60_filtered = wet_w_60
wet_w_60_filtered[np.where(wet_w_60 == 0)] = boundary_mean
wet_w_60_filtered = SG.sg_2d(z=wet_w_60_filtered, window_size=9, order=2)
wet_w_60[np.where(wet_w_60 == boundary_mean)] = 0
wet_w_60_filtered[np.where(wet_w_60 == 0)] = 0

shift_w_60_filtered = 60 - wet_w_60_filtered
shift_w_60_filtered[np.where(shift_w_60_filtered<=0)] = 0
shift_w_60_filtered[np.where(wet_w_60_filtered==0)] = 0

# check the MC of CS band
plt.matshow(wet_w_60)
plt.matshow(wet_w_60_filtered)

Fig60_surface = plt.figure(figsize=(12, 12), facecolor='w')
ax = Axes3D(Fig60_surface)
#plt.matshow(wet_w_60)
#CF.Scatter3D_plot(ax=ax,x=wet_x_60, y=wet_y_60, z=wet_w_60, elev=45, azim=45, color='r')
CF.Scatter3D_plot(ax=ax,x=wet_x_60, y=wet_y_60, z=wet_w_60_filtered, elev=45, azim=45, color='b')

Fig60_surface = plt.figure(figsize=(12, 12), facecolor='w')
ax = Axes3D(Fig60_surface)
CF.Scatter3D_plot(ax=ax,x=wet_x_60, y=wet_y_60, z=shift_w_60_filtered, elev=45, azim=45, color='b')

###### Save data Figure 5 122-keV, 2-pixel, scattering + DBSCAN scattering
data_save = wet_x_60
CF.SaveFiles(var=data_save, var_name=None, \
              location="C:\Jiajin\ChaSha_2017\CSC_Data\CSC_extracted_data\Figure_data\Figure_1500_Am_wet_x_60.csv")

data_save = wet_y_60
CF.SaveFiles(var=data_save, var_name=None, \
              location="C:\Jiajin\ChaSha_2017\CSC_Data\CSC_extracted_data\Figure_data\Figure_1500_Am_wet_y_60.csv")

data_save = wet_w_60
CF.SaveFiles(var=data_save, var_name=None, \
              location="C:\Jiajin\ChaSha_2017\CSC_Data\CSC_extracted_data\Figure_data\Figure_1500_Am_wet_w_60.csv")

data_save = wet_w_60_filtered
CF.SaveFiles(var=data_save, var_name=None, \
              location="C:\Jiajin\ChaSha_2017\CSC_Data\CSC_extracted_data\Figure_data\Figure_1500_Am_wet_w_60_filtered.csv")

data_save = shift_w_60_filtered
CF.SaveFiles(var=data_save, var_name=None, \
              location="C:\Jiajin\ChaSha_2017\CSC_Data\CSC_extracted_data\Figure_data\Figure_1500_Am_shift_w_60_filtered.csv")
del(data_save)
######################################################################################################## 





################################### Load the 81-keV charge sharing events #############################
CS_data = pd.read_csv( 'C:\Jiajin\ChaSha_2017\CSC_Data\CSC_extracted_data\Ba_3Pix_new.csv' )
Energy = CS_data.iloc[:, :].values  
#Energy = Energy - 1
#CF.SaveFiles( var=Energy, var_name=['E1', 'E2', 'E3'], location='C:\Jiajin\ChaSha_2017\CSC_Data\CSC_extracted_data\Ba_3Pix_new.csv' )

CS_dim = Energy.shape[1]
del(CS_data)

Energy_sum = np.sum( Energy, axis=1 ).reshape(-1,1)
Energy_E1_E2 = Energy[ np.intersect1d(np.where(Energy_sum >= 60)[0],np.where(Energy_sum <= 90)[0]) ]
Energy_E1_E2_else = np.delete( Energy, np.intersect1d(np.where(Energy_sum >= 60)[0],np.where(Energy_sum <= 90)[0]), axis =0 )

##### plot the histogram of sum_Energy within the selected ROI
CF.Histogram_lineplot(Hist=Energy_sum, Bins=300, x_lim_high=100, x_lim_low=0, color='red')

##### plot the raw scatter figures within the selected ROI
fig = plt.figure(figsize=(12, 12), facecolor='w')
ax = Axes3D(fig)
CF.Scatter3D_plot(ax=ax, x=Energy_E1_E2[:,0], y=Energy_E1_E2[:,1], z=Energy_E1_E2[:,2], elev=45, azim=45)
########################################################################################################



####################### DBSCAN Clustering and Plot the results #########################################
Energy_E1_E2_sum = np.sum( Energy_E1_E2, axis=1 ).reshape(-1,1)

##### Model Fitting I        
# Set Fitting Parameters density=28
y_hat1, y_unique1, core_indices1 = SC.Clustering(radius=1, density=5, data=Energy_E1_E2)

##### Model Fitting II    
# Set Fitting Parameters
# Plot the DBSCAN clustering results
clrs = plt.cm.Spectral(np.linspace(0, 0.5, 2*y_unique1.size))
fig = plt.figure(figsize=(12, 12), facecolor='w')
ax = Axes3D(fig)
for k, clr in zip(y_unique1, clrs):
    cur = (y_hat1 == k)
    if k == -1:
        CF.Scatter3D_plot(ax, Energy_E1_E2[cur, 0], Energy_E1_E2[cur, 1], Energy_E1_E2[cur, 2], elev=45, azim=45, color='k', marker='.')
        continue
    CF.Scatter3D_plot(ax, Energy_E1_E2[cur, 0], Energy_E1_E2[cur, 1], Energy_E1_E2[cur, 2], elev=45, azim=45, color=clr, marker='.')

ax.set_xlabel('Energy (keV)')
ax.set_ylabel('Energy (keV)')
ax.set_zlabel('Energy (keV)')

plt.xlim(0, 90)
plt.ylim(0, 90)
plt.grid(True)

###### check each DBSCAN clustering results
clrs = plt.cm.Spectral(np.linspace(0, 0.5, 2*y_unique2.size))
fig = plt.figure(figsize=(12, 12), facecolor='w')
ax = Axes3D(fig)
for k, clr in zip(y_unique2, clrs):
    cur = (y_hat1 == k)
    if k == 0:
        CF.Scatter3D_plot(ax, Energy_E1_E2[cur, 0], Energy_E1_E2[cur, 1], Energy_E1_E2[cur, 2], elev=45, azim=45, color='k', marker='.')
        continue
    
ax.set_xlabel('Energy (keV)')
ax.set_ylabel('Energy (keV)')
ax.set_zlabel('Energy (keV)')

plt.xlim(0, 65)
plt.ylim(0, 65)
plt.grid(True)
########################################################################################################




####################### Extract the cluster in the ROI #################################################
##### Reorganize the clustered scattering points
Energy = np.vstack( (Energy_E1_E2, Energy_E1_E2_else) )
Energy_sum = np.sum( Energy, axis=1 ).reshape(-1,1)

y_hat = np.array([-1]*len(Energy))
y_hat[np.where( y_hat1 == 0 ),] = 0
    
y_unique = np.unique(y_hat)

cluster_81_lab_0 = Energy[np.where( ( y_hat == 0 ) )]
cluster_81_lab_noise = Energy[np.where( ( y_hat == -1 ) )]

# Plot the DBSCAN clustering results
fig = plt.figure(figsize=(12, 12), facecolor='w')
ax = Axes3D(fig)
for k in y_unique:
    cur = (y_hat == k)
    if k == -1:
        CF.Scatter3D_plot(ax, Energy[cur, 0], Energy[cur, 1], Energy[cur, 2], elev=45, azim=45, color='k', marker='.')
        continue
    if k == 0:
        CF.Scatter3D_plot(ax, Energy[cur, 0], Energy[cur, 1], Energy[cur, 2], elev=45, azim=45, color='r', marker='.')
        continue

ax.set_xlabel('Energy (keV)')
ax.set_ylabel('Energy (keV)')
ax.set_zlabel('Energy (keV)')

ax.set_xlim3d(0, 90) 
ax.set_ylim3d(0, 90)
ax.set_zlim3d(0, 90)

###### Save data
data_save = np.hstack( (Energy, y_hat.reshape(-1,1)) ) ### ( E1, E2, label)
CF.SaveFiles(var=data_save, var_name=['E1', 'E2', 'E3', 'label'], \
              location="C:\Jiajin\ChaSha_2017\CSC_Data\CSC_extracted_data\Figure_data\Figure_1600_Ba_scatter_labeled.csv")
del(data_save)
######################################################################################################## 



#######################      CSC --- parameters measurement      #######################################
##### Initialize the CSC object
CSC_81_3pix = CSC( CS_dim=CS_dim, basis_old=basis_old, basis_new=basis_new, peak_energy=81, max_energy_range=140, seg_size=seg_size )
##### Calculate the MC-curve
wet_x_81, wet_y_81, wet_w_81, shift_w_81, seg_unit_81, Energy_rotation = CSC_81_3pix.Pix3_Measurement( CS_data_labeled = Energy )
Energy_rotation = np.hstack( ( Energy_rotation[:,0], Energy_rotation[:,1], Energy_rotation[:,2] ) )

##### Calculate the rotated (Ratio, E-sum) points
wet_x_81, wet_y_81, wet_w_81, shift_w_81, seg_unit_81, Energy_rot = CSC_81_3pix.Pix3_Measurement( CS_data_labeled = cluster_81_lab_0 )

##### Check the scattering plot and MC plot
clrs = plt.cm.Spectral(np.linspace(0, 0.5, 2*y_unique.size))
fig = plt.figure(figsize=(12, 12), facecolor='w')
ax = Axes3D(fig)
for k in y_unique:
    cur = (y_hat == k)
    if k == -1:
        CF.Scatter3D_plot(ax, Energy_rotation[cur, 0], Energy_rotation[cur, 1], Energy_rotation[cur, 2], elev=45, azim=45, color='k', marker='.')
        continue
    if k == 0:
        CF.Scatter3D_plot(ax, Energy_rotation[cur, 0], Energy_rotation[cur, 1], Energy_rotation[cur, 2], elev=45, azim=45, color='r', marker='.')
        continue

ax.set_xlabel('Energy (keV)')
ax.set_ylabel('Energy (keV)')
ax.set_zlabel('Energy (keV)')

ax.set_xlim3d(-1, 1) 
ax.set_ylim3d(-1, 1)
ax.set_zlim3d(0, 90)

###### Save data Figure 5 122-keV, 2-pixel, scattering + DBSCAN scattering
data_save = np.hstack( (Energy_rotation, y_hat.reshape(-1,1)) ) ### ( E1, E2, label)
CF.SaveFiles(var=data_save, var_name=['E1', 'E2', 'E3', 'label'], \
              location="C:\Jiajin\ChaSha_2017\CSC_Data\CSC_extracted_data\Figure_data\Figure_1600_Ba_transformed_scatter_labeled.csv")
del(data_save)
######################################################################################################## 



########################## filter the loss plane and compensation plane ################################ 
tmp = wet_w_81
tmp = tmp.reshape(-1,1)
boundary_mean = np.mean( heapq.nlargest(10, tmp) )
del tmp
gc.collect()
wet_w_81_filtered = wet_w_81
wet_w_81_filtered[np.where(wet_w_81 == 0)] = boundary_mean
wet_w_81_filtered = SG.sg_2d(z=wet_w_81_filtered, window_size=9, order=2)
wet_w_81[np.where(wet_w_81 == boundary_mean)] = 0
wet_w_81_filtered[np.where(wet_w_81 == 0)] = 0

shift_w_81_filtered = 81 - wet_w_81_filtered
shift_w_81_filtered[np.where(shift_w_81_filtered<=0)] = 0
shift_w_81_filtered[np.where(wet_w_81_filtered==0)] = 0

# check the MC of CS band
plt.matshow(wet_w_81)
plt.matshow(wet_w_81_filtered)

Fig81_surface = plt.figure(figsize=(12, 12), facecolor='w')
ax = Axes3D(Fig81_surface)
#plt.matshow(wet_w_60)
#CF.Scatter3D_plot(ax=ax,x=wet_x_60, y=wet_y_60, z=wet_w_60, elev=45, azim=45, color='r')
CF.Scatter3D_plot(ax=ax,x=wet_x_81, y=wet_y_81, z=wet_w_81_filtered, elev=45, azim=45, color='b')

Fig81_surface = plt.figure(figsize=(12, 12), facecolor='w')
ax = Axes3D(Fig81_surface)
CF.Scatter3D_plot(ax=ax,x=wet_x_81, y=wet_y_81, z=shift_w_81_filtered, elev=45, azim=45, color='b')

###### Save data Figure 5 122-keV, 2-pixel, scattering + DBSCAN scattering
data_save = wet_x_81
CF.SaveFiles(var=data_save, var_name=None, \
              location="C:\Jiajin\ChaSha_2017\CSC_Data\CSC_extracted_data\Figure_data\Figure_1600_Ba_wet_x_81.csv")

data_save = wet_y_81
CF.SaveFiles(var=data_save, var_name=None, \
              location="C:\Jiajin\ChaSha_2017\CSC_Data\CSC_extracted_data\Figure_data\Figure_1600_Ba_wet_y_81.csv")

data_save = wet_w_81
CF.SaveFiles(var=data_save, var_name=None, \
              location="C:\Jiajin\ChaSha_2017\CSC_Data\CSC_extracted_data\Figure_data\Figure_1600_Ba_wet_w_81.csv")

data_save = wet_w_81_filtered
CF.SaveFiles(var=data_save, var_name=None, \
              location="C:\Jiajin\ChaSha_2017\CSC_Data\CSC_extracted_data\Figure_data\Figure_1600_Ba_wet_w_81_filtered.csv")

data_save = shift_w_81_filtered
CF.SaveFiles(var=data_save, var_name=None, \
              location="C:\Jiajin\ChaSha_2017\CSC_Data\CSC_extracted_data\Figure_data\Figure_1600_Ba_shift_w_81_filtered.csv")
del(data_save)
######################################################################################################## 





################################### Load the 122-keV charge sharing events #############################
CS_data = pd.read_csv( 'C:\Jiajin\ChaSha_2017\CSC_Data\CSC_extracted_data\Co_3Pix_new.csv' )
Energy = CS_data.iloc[:, :].values  
#Energy = Energy - 1.2
#CF.SaveFiles( var=Energy, var_name=['E1', 'E2', 'E3'], location='C:\Jiajin\ChaSha_2017\CSC_Data\CSC_extracted_data\Co_3Pix_new.csv' )

CS_dim = Energy.shape[1]
del(CS_data)

Energy_sum = np.sum( Energy, axis=1 ).reshape(-1,1)
Energy_E1_E2 = Energy[ np.intersect1d(np.where(Energy_sum >= 100)[0],np.where(Energy_sum <= 126)[0]) ]
Energy_E1_E2_else = np.delete( Energy, np.intersect1d(np.where(Energy_sum >= 100)[0],np.where(Energy_sum <= 126)[0]), axis =0 )

##### plot the histogram of sum_Energy within the selected ROI
CF.Histogram_lineplot(Hist=Energy_sum, Bins=300, x_lim_high=150, x_lim_low=0, color='red')

##### plot the raw scatter figures within the selected ROI
fig = plt.figure(figsize=(12, 12), facecolor='w')
ax = Axes3D(fig)
CF.Scatter3D_plot(ax=ax, x=Energy_E1_E2[:,0], y=Energy_E1_E2[:,1], z=Energy_E1_E2[:,2], elev=45, azim=45)
########################################################################################################



####################### DBSCAN Clustering and Plot the results #########################################
Energy_E1_E2_sum = np.sum( Energy_E1_E2, axis=1 ).reshape(-1,1)

##### Model Fitting I        
# Set Fitting Parameters density=28
y_hat1, y_unique1, core_indices1 = SC.Clustering(radius=2, density=1, data=Energy_E1_E2)

##### Model Fitting II    
# Set Fitting Parameters
# Plot the DBSCAN clustering results
clrs = plt.cm.Spectral(np.linspace(0, 0.5, 2*y_unique1.size))
fig = plt.figure(figsize=(12, 12), facecolor='w')
ax = Axes3D(fig)
for k, clr in zip(y_unique1, clrs):
    cur = (y_hat1 == k)
    if k == -1:
        CF.Scatter3D_plot(ax, Energy_E1_E2[cur, 0], Energy_E1_E2[cur, 1], Energy_E1_E2[cur, 2], elev=45, azim=45, color='k', marker='.')
        continue
    CF.Scatter3D_plot(ax, Energy_E1_E2[cur, 0], Energy_E1_E2[cur, 1], Energy_E1_E2[cur, 2], elev=45, azim=45, color=clr, marker='.')

ax.set_xlabel('Energy (keV)')
ax.set_ylabel('Energy (keV)')
ax.set_zlabel('Energy (keV)')

ax.set_xlim3d(0, 130) 
ax.set_ylim3d(0, 130)
ax.set_zlim3d(0, 130)

###### check each DBSCAN clustering results
clrs = plt.cm.Spectral(np.linspace(0, 0.5, 2*y_unique2.size))
fig = plt.figure(figsize=(12, 12), facecolor='w')
ax = Axes3D(fig)
for k, clr in zip(y_unique2, clrs):
    cur = (y_hat1 == k)
    if k == 0:
        CF.Scatter3D_plot(ax, Energy_E1_E2[cur, 0], Energy_E1_E2[cur, 1], Energy_E1_E2[cur, 2], elev=45, azim=45, color='k', marker='.')
        continue
    
ax.set_xlabel('Energy (keV)')
ax.set_ylabel('Energy (keV)')
ax.set_zlabel('Energy (keV)')

plt.xlim(0, 65)
plt.ylim(0, 65)
plt.grid(True)
########################################################################################################




####################### Extract the cluster in the ROI #################################################
##### Reorganize the clustered scattering points
Energy = np.vstack( (Energy_E1_E2, Energy_E1_E2_else) )
Energy_sum = np.sum( Energy, axis=1 ).reshape(-1,1)

y_hat = np.array([-1]*len(Energy))
y_hat[np.where( y_hat1 == 0 ),] = 0
    
y_unique = np.unique(y_hat)

cluster_122_lab_0 = Energy[np.where( ( y_hat == 0 ) )]
cluster_122_lab_noise = Energy[np.where( ( y_hat == -1 ) )]

# Plot the DBSCAN clustering results
fig = plt.figure(figsize=(12, 12), facecolor='w')
ax = Axes3D(fig)
for k in y_unique:
    cur = (y_hat == k)
    if k == -1:
        CF.Scatter3D_plot(ax, Energy[cur, 0], Energy[cur, 1], Energy[cur, 2], elev=45, azim=45, color='k', marker='.')
        continue
    if k == 0:
        CF.Scatter3D_plot(ax, Energy[cur, 0], Energy[cur, 1], Energy[cur, 2], elev=45, azim=45, color='r', marker='.')
        continue

ax.set_xlabel('Energy (keV)')
ax.set_ylabel('Energy (keV)')
ax.set_zlabel('Energy (keV)')

ax.set_xlim3d(0, 130) 
ax.set_ylim3d(0, 130)
ax.set_zlim3d(0, 130)

###### Save data
data_save = np.hstack( (Energy, y_hat.reshape(-1,1)) ) ### ( E1, E2, label)
CF.SaveFiles(var=data_save, var_name=['E1', 'E2', 'E3', 'label'], \
              location="C:\Jiajin\ChaSha_2017\CSC_Data\CSC_extracted_data\Figure_data\Figure_1600_Co_scatter_labeled.csv")
del(data_save)
######################################################################################################## 



#######################      CSC --- parameters measurement      #######################################
##### Initialize the CSC object
CSC_122_3pix = CSC( CS_dim=CS_dim, basis_old=basis_old, basis_new=basis_new, peak_energy=122, max_energy_range=140, seg_size=seg_size )
##### Calculate the MC-curve
wet_x_122, wet_y_122, wet_w_122, shift_w_122, seg_unit_122, Energy_rotation = CSC_122_3pix.Pix3_Measurement( CS_data_labeled = Energy )
Energy_rotation = np.hstack( ( Energy_rotation[:,0], Energy_rotation[:,1], Energy_rotation[:,2] ) )

##### Calculate the rotated (Ratio, E-sum) points
wet_x_122, wet_y_122, wet_w_122, shift_w_122, seg_unit_122, Energy_rot = CSC_122_3pix.Pix3_Measurement( CS_data_labeled = cluster_122_lab_0 )

##### Check the scattering plot and MC plot
clrs = plt.cm.Spectral(np.linspace(0, 0.5, 2*y_unique.size))
fig = plt.figure(figsize=(12, 12), facecolor='w')
ax = Axes3D(fig)
for k in y_unique:
    cur = (y_hat == k)
    if k == -1:
        CF.Scatter3D_plot(ax, Energy_rotation[cur, 0], Energy_rotation[cur, 1], Energy_rotation[cur, 2], elev=45, azim=45, color='k', marker='.')
        continue
    if k == 0:
        CF.Scatter3D_plot(ax, Energy_rotation[cur, 0], Energy_rotation[cur, 1], Energy_rotation[cur, 2], elev=45, azim=45, color='r', marker='.')
        continue

ax.set_xlabel('Energy (keV)')
ax.set_ylabel('Energy (keV)')
ax.set_zlabel('Energy (keV)')

ax.set_xlim3d(-1, 1) 
ax.set_ylim3d(-1, 1)
ax.set_zlim3d(0, 130)

###### Save data Figure 5 122-keV, 2-pixel, scattering + DBSCAN scattering
data_save = np.hstack( (Energy_rotation, y_hat.reshape(-1,1)) ) ### ( E1, E2, label)
CF.SaveFiles(var=data_save, var_name=['E1', 'E2', 'E3', 'label'], \
              location="C:\Jiajin\ChaSha_2017\CSC_Data\CSC_extracted_data\Figure_data\Figure_1600_Co_transformed_scatter_labeled.csv")
del(data_save)
######################################################################################################## 



########################## filter the loss plane and compensation plane ################################ 
tmp = wet_w_122
tmp = tmp.reshape(-1,1)
boundary_mean = np.mean( heapq.nlargest(10, tmp) )
del tmp
gc.collect()
wet_w_122_filtered = wet_w_122
wet_w_122_filtered[np.where(wet_w_122 == 0)] = boundary_mean
wet_w_122_filtered = SG.sg_2d(z=wet_w_122_filtered, window_size=9, order=2)
wet_w_122[np.where(wet_w_122 == boundary_mean)] = 0
wet_w_122_filtered[np.where(wet_w_122 == 0)] = 0

shift_w_122_filtered = 122 - wet_w_122_filtered
shift_w_122_filtered[np.where(shift_w_122_filtered<=0)] = 0
shift_w_122_filtered[np.where(wet_w_122_filtered==0)] = 0

# check the MC of CS band
plt.matshow(wet_w_122)
plt.matshow(wet_w_122_filtered)

Fig122_surface = plt.figure(figsize=(12, 12), facecolor='w')
ax = Axes3D(Fig122_surface)
#plt.matshow(wet_w_60)
#CF.Scatter3D_plot(ax=ax,x=wet_x_60, y=wet_y_60, z=wet_w_60, elev=45, azim=45, color='r')
CF.Scatter3D_plot(ax=ax,x=wet_x_122, y=wet_y_122, z=wet_w_122_filtered, elev=45, azim=45, color='b')

Fig122_surface = plt.figure(figsize=(12, 12), facecolor='w')
ax = Axes3D(Fig122_surface)
CF.Scatter3D_plot(ax=ax,x=wet_x_122, y=wet_y_122, z=shift_w_122_filtered, elev=45, azim=45, color='b')

###### Save data Figure 5 122-keV, 2-pixel, scattering + DBSCAN scattering
data_save = wet_x_122
CF.SaveFiles(var=data_save, var_name=None, \
              location="C:\Jiajin\ChaSha_2017\CSC_Data\CSC_extracted_data\Figure_data\Figure_1600_Co_wet_x_122.csv")

data_save = wet_y_122
CF.SaveFiles(var=data_save, var_name=None, \
              location="C:\Jiajin\ChaSha_2017\CSC_Data\CSC_extracted_data\Figure_data\Figure_1600_Co_wet_y_122.csv")

data_save = wet_w_122
CF.SaveFiles(var=data_save, var_name=None, \
              location="C:\Jiajin\ChaSha_2017\CSC_Data\CSC_extracted_data\Figure_data\Figure_1600_Co_wet_w_122.csv")

data_save = wet_w_122_filtered
CF.SaveFiles(var=data_save, var_name=None, \
              location="C:\Jiajin\ChaSha_2017\CSC_Data\CSC_extracted_data\Figure_data\Figure_1600_Co_wet_w_122_filtered.csv")

data_save = shift_w_122_filtered
CF.SaveFiles(var=data_save, var_name=None, \
              location="C:\Jiajin\ChaSha_2017\CSC_Data\CSC_extracted_data\Figure_data\Figure_1600_Co_shift_w_122_filtered.csv")
del(data_save)
######################################################################################################## 





################################################################################################################
###################################    Compensation mapping calculation    #####################################
##### Three Calibration line visulization
Fig_surface = plt.figure(figsize=(12, 12), facecolor='w')
ax = Axes3D(Fig_surface)
CF.Scatter3D_plot(ax=ax,x=wet_x_60, y=wet_y_60, z=shift_w_60_filtered, elev=45, azim=45, color='b')
CF.Scatter3D_plot(ax=ax,x=wet_x_122, y=wet_y_122, z=shift_w_122_filtered, elev=45, azim=45, color='b')
CF.Scatter3D_plot(ax=ax,x=wet_x_122, y=wet_y_122, z=shift_w_122_filtered, elev=45, azim=45, color='b')

##### Compensation map implementation
Map, R, E = MG.Pix2_MapGeneration(E1=60, E1_shift=shift_w_60_filter, E2=81, E2_shift=shift_w_81_filter, \
                            E3=122, E3_shift=shift_w_122_filter, Ratio=wet_x_60, E_range=140, dE=1)
Map[np.where(Map < 0)] = 0

##### Compensation map visualization
X = np.tile( R.reshape(-1,1), (1,len(E)) ).T
Y = np.tile( E.reshape(-1,1).T, (len(R),1) ).T
Z = Map
         
fig = plt.figure(figsize=(12, 12), facecolor='w')
ax = Axes3D(fig)
ax.view_init(elev=90,azim=270)
surf = ax.plot_surface(X, Y, Z, cmap=plt.cm.coolwarm, linewidth=0, antialiased=False)
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.suptitle('Interpolation of the charge loss compensation map',fontsize=20)
plt.xlabel('Spatial-axis',fontsize=20)
plt.ylabel('Energy-axis',fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.show()
################################################################################################################




################################################################################################################
############################## Charge sharing correction based on compensation map #############################
from charge_sharing_correction import charge_sharing_correction as CSC2

from charge_sharing_correction_v2 import charge_sharing_correction as CSC # class file, charge sharing correction
from charge_sharing_correction_v2 import Common_used_function as CF # class file, common-used plotting tools
from charge_sharing_correction_v2 import SG_Filter as SG # class file, S-G filters of different dimentions
from charge_sharing_correction_v2 import Map_generation as MG # class file, compensation map generation
from charge_sharing_correction_v2 import scattering_clustering as SC # class file, DBSCAN clustering

'''
data_map = np.vstack( (cluster_60_lab_0, cluster_60_lab_noise, cluster_81_lab_0, cluster_81_lab_noise,\
                                 cluster_122_lab_0, cluster_122_lab_noise) )

CSC_122_2pix = CSC( CS_dim=CS_dim, basis_old=basis_old, basis_new=basis_new, peak_energy=122, max_energy_range=140, seg_size=seg_size )

data_map_corrected = CSC_122_2pix.Pix2_Correction( CS_data_labeled=data_map, correction_map=Map, R=R, E=E )

CF.Scatter2D_plot(x=data_map[:,0].tolist(), y=data_map[:,1].tolist(), x_lim_left=0, x_lim_right=140,\
                  y_lim_left=0, y_lim_right=140, color='r')

CF.Scatter2D_plot(x=data_map_corrected[:,0].tolist(), y=data_map_corrected[:,1].tolist(), x_lim_left=0, x_lim_right=140,\
                  y_lim_left=0, y_lim_right=140, color='k')
'''

### 60
CSC2_60_3Pix = CSC2( CS_dim=CS_dim, basis_old=basis_old, basis_new=basis_new, peak_energy=60, max_energy_range=140, seg_size=2 )
wet_x, wet_y, wet_w, shift_w, seg_unit = CSC2_60_3Pix.Pix3_Measurement( CS_data_labeled=cluster_60_lab_0 )
cluster_60_lab_0_corrected = CSC2_60_3Pix.Pix3_Correction( seg_unit=seg_unit, shift_w=shift_w, CS_data_labeled=cluster_60_lab_0 )
cluster_60_lab_noise_corrected = CSC2_60_3Pix.Pix3_Correction( seg_unit=seg_unit, shift_w=shift_w, CS_data_labeled=cluster_60_lab_noise )

### 81
CSC2_81_3Pix = CSC2(CS_dim=CS_dim, basis_old=basis_old, basis_new=basis_new, peak_energy=81, max_energy_range=140, seg_size=2)
wet_x, wet_y, wet_w, shift_w, seg_unit = CSC2_81_3Pix.Pix3_Measurement( CS_data_labeled=cluster_81_lab_0 )
cluster_81_lab_0_corrected = CSC2_81_3Pix.Pix3_Correction( seg_unit=seg_unit, shift_w=shift_w, CS_data_labeled=cluster_81_lab_0 )
cluster_81_lab_noise_corrected = CSC2_81_3Pix.Pix3_Correction( seg_unit=seg_unit, shift_w=shift_w, CS_data_labeled=cluster_81_lab_noise )

### 122
CSC2_122_3Pix = CSC2(CS_dim=CS_dim, basis_old=basis_old, basis_new=basis_new, peak_energy=122, max_energy_range=140, seg_size=2)
wet_x, wet_y, wet_w, shift_w, seg_unit = CSC2_122_3Pix.Pix3_Measurement( CS_data_labeled=cluster_122_lab_0 )
cluster_122_lab_0_corrected = CSC2_122_3Pix.Pix3_Correction( seg_unit=seg_unit, shift_w=shift_w, CS_data_labeled=cluster_122_lab_0 )
cluster_122_lab_noise_corrected = CSC2_122_3Pix.Pix3_Correction( seg_unit=seg_unit, shift_w=shift_w, CS_data_labeled=cluster_122_lab_noise )

### noise
#noise = np.vstack( (cluster_60_lab_noise, cluster_81_lab_noise, cluster_122_lab_noise) )
#CSC_122_2pix = CSC( CS_dim=CS_dim, basis_old=basis_old, basis_new=basis_new, peak_energy=122, max_energy_range=140, seg_size=seg_size )
#noise_corrected = CSC_122_2pix.Pix2_Correction( CS_data_labeled=noise, correction_map=Map, R=R, E=E )


data = np.vstack( (cluster_60_lab_0, cluster_60_lab_noise,\
                   cluster_81_lab_0, cluster_81_lab_noise,\
                   cluster_122_lab_0, cluster_122_lab_noise) )

data_corrected = np.vstack( (cluster_60_lab_0_corrected, cluster_60_lab_noise_corrected, \
                             cluster_81_lab_0_corrected, cluster_81_lab_noise_corrected, \
                             cluster_122_lab_0_corrected, cluster_122_lab_noise_corrected) )

Energy_sum = np.sum(data, axis=1).reshape(-1,1)
CF.Histogram_lineplot(Hist=Energy_sum, Bins=300, x_lim_low=0, x_lim_high=200, color='blue')

Energy_sum_corrected = np.sum(data_corrected, axis=1).reshape(-1,1)
CF.Histogram_lineplot(Hist=Energy_sum_corrected, Bins=300, x_lim_low=0, x_lim_high=200, color='red')
plt.ylim(0, 20000)

### save the data and corrected data
CF.SaveFiles(var=data, var_name=['E1', 'E2', 'E3'], location='C:\Jiajin\ChaSha_2017\CSC_Data\CSC_extracted_data\Figure_data\Figure_18_raw.csv')
CF.SaveFiles(var=data_corrected, var_name=['E1', 'E2', 'E3'], location='C:\Jiajin\ChaSha_2017\CSC_Data\CSC_extracted_data\Figure_data\Figure_18_corrected.csv')

### save separately
# raw
data = np.vstack( ( cluster_60_lab_0, cluster_60_lab_noise ) )
CF.SaveFiles(var=data, var_name=['E1', 'E2', 'E3'], location='C:\Jiajin\ChaSha_2017\CSC_Data\CSC_extracted_data\spectra_data\Am_3pix_raw.csv')

data = np.vstack( ( cluster_81_lab_0, cluster_81_lab_noise ) )
CF.SaveFiles(var=data, var_name=['E1', 'E2', 'E3'], location='C:\Jiajin\ChaSha_2017\CSC_Data\CSC_extracted_data\spectra_data\Ba_3pix_raw.csv')

data = np.vstack( ( cluster_122_lab_0, cluster_122_lab_noise ) )
CF.SaveFiles(var=data, var_name=['E1', 'E2', 'E3'], location='C:\Jiajin\ChaSha_2017\CSC_Data\CSC_extracted_data\spectra_data\Co_3pix_raw.csv')

# corrected
data = np.vstack( ( cluster_60_lab_0_corrected, cluster_60_lab_noise_corrected ) )
CF.SaveFiles(var=data, var_name=['E1', 'E2', 'E3'], location='C:\Jiajin\ChaSha_2017\CSC_Data\CSC_extracted_data\spectra_data\Am_3pix_corrected.csv')

data = np.vstack( ( cluster_81_lab_0_corrected, cluster_81_lab_noise_corrected ) )
CF.SaveFiles(var=data, var_name=['E1', 'E2', 'E3'], location='C:\Jiajin\ChaSha_2017\CSC_Data\CSC_extracted_data\spectra_data\Ba_3pix_corrected.csv')

data = np.vstack( ( cluster_122_lab_0_corrected, cluster_122_lab_noise_corrected ) )
CF.SaveFiles(var=data, var_name=['E1', 'E2', 'E3'], location='C:\Jiajin\ChaSha_2017\CSC_Data\CSC_extracted_data\spectra_data\Co_3pix_corrected.csv')



