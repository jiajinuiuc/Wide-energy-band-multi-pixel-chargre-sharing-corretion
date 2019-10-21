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
Last update: May, 2019
"""
import sys
sys.path.append('C:\Jiajin\Mfile\Training_Sample_Analysis')

from charge_sharing_correction import charge_sharing_correction as CSC
from charge_sharing_correction import SG_Filter as SG
from charge_sharing_correction import Common_used_function as CF

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.datasets as ds
import matplotlib.colors
from sklearn.cluster import DBSCAN
from mpl_toolkits.mplot3d import Axes3D
from functools import reduce
from scipy import signal

%matplotlib qt5

################################### Load the 3-pixel charge sharing events #############################
CS_data = pd.read_csv( 'C:\Jiajin\Mfile\Training_Sample_Analysis\Pix3_80Sharing.csv' )
Energy = CS_data.iloc[:, :].values  
CS_dim = Energy.shape[1]

### Initialize basis parameters
basis_old =np.mat( [ [1,0,0],
                     [0,1,0],
                     [0,0,1] ] ) #[x, y, z]

basis_new = np.mat( [ [1/np.sqrt(6),  1/np.sqrt(2), 1/np.sqrt(3)],
                      [1/np.sqrt(6), -1/np.sqrt(2), 1/np.sqrt(3)],
                      [-2/np.sqrt(6), 0,            1/np.sqrt(3)] ] )

### Initialized energy range segmentation
energy_seg = np.array( [20, 50, 76, 100, 127, 140] )
energy_ind = 0
########################################################################################################





########################################################################################################
################################### Extract 36 keV events ##############################################
Energy_sum = np.sum( Energy, axis=1 ).reshape(-1,1)
Energy_E1_E2 = Energy[ np.intersect1d(np.where(Energy_sum >= energy_seg[energy_ind])[0],\
                                      np.where(Energy_sum <= energy_seg[energy_ind+1])[0]) ]
energy_ind += 1
Energy_sum = np.sum( Energy_E1_E2, axis=1 ).reshape(-1,1)
# scatter plot and histogram
fig = plt.figure(figsize=(12, 12), facecolor='w'); ax = Axes3D(fig);
CF.Scatter3D_plot(ax=ax, x=Energy_E1_E2[:,0], y=Energy_E1_E2[:,1], z=Energy_E1_E2[:,2], elev=0, azim=45,\
                   x_lim=55, y_lim=55, color='k')
CF.Histogram_lineplot(Hist=Energy_sum, Bins=800, x_lim_low=20, x_lim_high=140, color='blue')
########################################################################################################

####################### DBSCAN Clustering and Plot the results #########################################
##### Model Fitting I       
# Set Fitting Parameters
eps, min_samples = (2, 65) # 122 keV, high density CS events
model1 = DBSCAN( eps=eps, min_samples=min_samples )
model1.fit( Energy_E1_E2 )
y_hat1 = model1.labels_

core_indices1 = np.zeros_like(y_hat1, dtype=bool) # create zero/boolean array with the same length
core_indices1[model1.core_sample_indices_] = True # 核样本的目录 < (label != 0)

y_unique1 = np.unique(y_hat1) # extract different Labels
n_clusters1 = y_unique1.size - (1 if -1 in y_hat1 else 0)
print(y_unique1, 'clustering number is :', n_clusters1)

# Plot the DBSCAN clustering results
fig = plt.figure(figsize=(12, 12), facecolor='w')
ax = Axes3D(fig)
elev=45; azim=45
for k in y_unique1:
    cur = (y_hat1 == k)
    if k == -1:
        CF.Scatter3D_plot(ax=ax, x=Energy_E1_E2[cur,0], y=Energy_E1_E2[cur,1],\
                           z=Energy_E1_E2[cur,2], elev=elev, azim=azim, x_lim=55, y_lim=55, color = 'k')
        continue
    CF.Scatter3D_plot(ax=ax, x=Energy_E1_E2[cur,0], y=Energy_E1_E2[cur,1],\
                           z=Energy_E1_E2[cur,2], elev=elev, azim=azim, x_lim=55, y_lim=55, color = 'r')
    CF.Scatter3D_plot(ax=ax, x=Energy_E1_E2[cur & core_indices1,0], y=Energy_E1_E2[cur & core_indices1,1],\
                           z=Energy_E1_E2[cur & core_indices1,2], elev=elev, azim=azim, x_lim=55, y_lim=55, color = 'r')

###### check each DBSCAN clustering results
fig = plt.figure(figsize=(12, 12), facecolor='w')
ax = Axes3D(fig)
for k in y_unique1:
    cur = (y_hat1 == k)
    if k == 0:
        CSC.Scatter3D_plot(ax=ax, x=Energy_E1_E2[cur,0], y=Energy_E1_E2[cur,1],\
                           z=Energy_E1_E2[cur,2], elev=45, azim=45, x_lim=55, y_lim=55, color = 'k')
        continue
########################################################################################################

####################### Extract the cluster in the ROI #################################################
##### Reorganize the clustered scattering points
y_hat = np.array([-1]*len(y_hat1))
y_hat[np.where(y_hat1 != -1),] = 0
y_unique = np.unique(y_hat)
cluster_lab_0 = Energy_E1_E2[np.where( ( y_hat == 0 ) )]
###### check each DBSCAN clustering results
fig = plt.figure(figsize=(12, 12), facecolor='w')
ax = Axes3D(fig)
for k in y_unique:
    cur = (y_hat == k)
    if k == -1:
        CSC.Scatter3D_plot(ax=ax, x=Energy_E1_E2[cur,0], y=Energy_E1_E2[cur,1],\
                           z=Energy_E1_E2[cur,2], elev=45, azim=45, x_lim=55, y_lim=55, color = 'k')
        continue
    if k == 0:
        CSC.Scatter3D_plot(ax=ax, x=Energy_E1_E2[cur,0], y=Energy_E1_E2[cur,1],\
                           z=Energy_E1_E2[cur,2], elev=45, azim=45, x_lim=55, y_lim=55, color = 'b')
        continue
########################################################################################################  

####################### "Rot -> MC Shifting -> Rot" CSC function #######################################
##### Initialize the CSC object, charge sharing band correction
seg_size = 4
CSC_36_3pix = CSC( CS_dim=CS_dim, basis_old=basis_old, basis_new=basis_new, peak_energy=36, max_energy_range=140, seg_size=seg_size )
wet_x_36, wet_y_36, wet_w_36, shift_w_36, seg_unit_36 = CSC_36_3pix.Pix3_Measurement( CS_data_labeled = cluster_lab_0 )
Energy36_corrected = CSC_36_3pix.Pix3_Correction(seg_unit=seg_unit_36, shift_w=shift_w_36, CS_data_labeled=Energy_E1_E2)

# check the scattering plot and MC plot
Fig36 = plt.figure(figsize=(12, 12), facecolor='w')
ax = Axes3D(Fig36)
CF.Scatter3D_plot(ax=ax, x=Energy36_corrected[:,0], y=Energy36_corrected[:,1], z=Energy36_corrected[:,2],\
                   elev=elev, azim=azim, x_lim=55, y_lim=55, color='red')

# check the histogram
Energy_sum = np.sum(Energy_E1_E2, axis=1).reshape(-1,1)
CF.Histogram_barplot(Hist=Energy_sum, Bins=80, x_lim_low=20, x_lim_high=140)
Energy_sum = np.sum(Energy36_corrected, axis=1).reshape(-1,1)
CF.Histogram_barplot(Hist=Energy_sum, Bins=80, x_lim_low=20, x_lim_high=140)

# check the MC of CS band
Fig36_surface = plt.figure(figsize=(12, 12), facecolor='w')
ax = Axes3D(Fig36_surface)
CF.Surface3D_plot(ax=ax, x=wet_x_36, y=wet_y_36, z=wet_w_36, elev=45, azim=45)
###############################################################################################################



########################################################################################################
################################### Extract 60 keV events ##############################################
# energy_ind = 1
Energy_sum = np.sum( Energy, axis=1 ).reshape(-1,1)
Energy_E1_E2 = Energy[ np.intersect1d(np.where(Energy_sum >= energy_seg[energy_ind])[0],\
                                      np.where(Energy_sum <= energy_seg[energy_ind+1])[0]) ]
energy_ind += 1
Energy_sum = np.sum( Energy_E1_E2, axis=1 ).reshape(-1,1)
# scatter plot and histogram
fig = plt.figure(figsize=(12, 12), facecolor='w'); ax = Axes3D(fig);
CF.Scatter3D_plot(ax=ax, x=Energy_E1_E2[:,0], y=Energy_E1_E2[:,1], z=Energy_E1_E2[:,2], elev=45, azim=45,\
                   x_lim=62, y_lim=62, color='k')
CF.Histogram_lineplot(Hist=Energy_sum, Bins=800, x_lim_low=20, x_lim_high=140, color='blue')
########################################################################################################

####################### DBSCAN Clustering and Plot the results #########################################
##### Model Fitting I       
# Set Fitting Parameters
eps, min_samples = (2,75) # 122 keV, high density CS events
model1 = DBSCAN( eps=eps, min_samples=min_samples )
model1.fit( Energy_E1_E2 )
y_hat1 = model1.labels_

core_indices1 = np.zeros_like(y_hat1, dtype=bool) # create zero/boolean array with the same length
core_indices1[model1.core_sample_indices_] = True # 核样本的目录 < (label != 0)

y_unique1 = np.unique(y_hat1) # extract different Labels
n_clusters1 = y_unique1.size - (1 if -1 in y_hat1 else 0)
print(y_unique1, 'clustering number is :', n_clusters1)

# Plot the DBSCAN clustering results
fig = plt.figure(figsize=(12, 12), facecolor='w')
ax = Axes3D(fig)
elev=45; azim=45
for k in y_unique1:
    cur = (y_hat1 == k)
    if k == -1:
        CF.Scatter3D_plot(ax=ax, x=Energy_E1_E2[cur,0], y=Energy_E1_E2[cur,1],\
                           z=Energy_E1_E2[cur,2], elev=elev, azim=azim, x_lim=62, y_lim=62, color = 'k')
        continue
    CF.Scatter3D_plot(ax=ax, x=Energy_E1_E2[cur,0], y=Energy_E1_E2[cur,1],\
                           z=Energy_E1_E2[cur,2], elev=elev, azim=azim, x_lim=62, y_lim=62, color = 'r')
    CF.Scatter3D_plot(ax=ax, x=Energy_E1_E2[cur & core_indices1,0], y=Energy_E1_E2[cur & core_indices1,1],\
                           z=Energy_E1_E2[cur & core_indices1,2], elev=elev, azim=azim, x_lim=62, y_lim=62, color = 'r')

###### check each DBSCAN clustering results
fig = plt.figure(figsize=(12, 12), facecolor='w')
ax = Axes3D(fig)
for k in y_unique1:
    cur = (y_hat1 == k)
    if k == 0:
        CF.Scatter3D_plot(ax=ax, x=Energy_E1_E2[cur,0], y=Energy_E1_E2[cur,1],\
                           z=Energy_E1_E2[cur,2], elev=45, azim=45, x_lim=62, y_lim=62, color = 'k')
        continue
########################################################################################################

####################### Extract the cluster in the ROI #################################################
##### Reorganize the clustered scattering points
y_hat = np.array([-1]*len(y_hat1))
y_hat[np.where(y_hat1 != -1),] = 0
y_unique = np.unique(y_hat)
cluster_lab_0 = Energy_E1_E2[np.where( ( y_hat == 0 ) )]
###### check each DBSCAN clustering results
fig = plt.figure(figsize=(12, 12), facecolor='w')
ax = Axes3D(fig)
for k in y_unique:
    cur = (y_hat == k)
    if k == -1:
        CF.Scatter3D_plot(ax=ax, x=Energy_E1_E2[cur,0], y=Energy_E1_E2[cur,1],\
                           z=Energy_E1_E2[cur,2], elev=45, azim=45, x_lim=55, y_lim=55, color = 'k')
        continue
    if k == 0:
        CF.Scatter3D_plot(ax=ax, x=Energy_E1_E2[cur,0], y=Energy_E1_E2[cur,1],\
                           z=Energy_E1_E2[cur,2], elev=45, azim=45, x_lim=55, y_lim=55, color = 'b')
        continue
########################################################################################################  

####################### "Rot -> MC Shifting -> Rot" CSC function #######################################
##### Initialize the CSC object, charge sharing band correction
seg_size = 4
CSC_60_3pix = CSC( CS_dim=CS_dim, basis_old=basis_old, basis_new=basis_new, peak_energy=60, max_energy_range=140, seg_size=seg_size )
wet_x_60, wet_y_60, wet_w_60, shift_w_60, seg_unit_60 = CSC_60_3pix.Pix3_Measurement( CS_data_labeled = cluster_lab_0 )
Energy60_corrected = CSC_60_3pix.Pix3_Correction(seg_unit=seg_unit_60, shift_w=shift_w_60, CS_data_labeled=Energy_E1_E2)

# check the scattering plot and MC plot
Fig60 = plt.figure(figsize=(12, 12), facecolor='w')
ax = Axes3D(Fig60)
CF.Scatter3D_plot(ax=ax, x=Energy60_corrected[:,0], y=Energy60_corrected[:,1], z=Energy60_corrected[:,2],\
                   elev=elev, azim=azim, x_lim=55, y_lim=55, color='red')

# check the histogram
Energy_sum = np.sum(Energy_E1_E2, axis=1).reshape(-1,1)
CF.Histogram_barplot(Hist=Energy_sum, Bins=80, x_lim_low=20, x_lim_high=140)
Energy_sum = np.sum(Energy60_corrected, axis=1).reshape(-1,1)
CF.Histogram_barplot(Hist=Energy_sum, Bins=80, x_lim_low=20, x_lim_high=140)

# check the MC of CS band
Fig60_surface = plt.figure(figsize=(12, 12), facecolor='w')
ax = Axes3D(Fig60_surface)
CF.Surface3D_plot(ax=ax, x=wet_x_60, y=wet_y_60, z=wet_w_60, elev=45, azim=45)
###############################################################################################################



########################################################################################################
################################### Extract 81 keV events ##############################################
Energy_sum = np.sum( Energy, axis=1 ).reshape(-1,1)
Energy_E1_E2 = Energy[ np.intersect1d(np.where(Energy_sum >= energy_seg[energy_ind])[0],\
                                      np.where(Energy_sum <= energy_seg[energy_ind+1])[0]) ]
energy_ind += 1
Energy_sum = np.sum( Energy_E1_E2, axis=1 ).reshape(-1,1)
# scatter plot and histogram
fig = plt.figure(figsize=(12, 12), facecolor='w'); ax = Axes3D(fig);
CF.Scatter3D_plot(ax=ax, x=Energy_E1_E2[:,0], y=Energy_E1_E2[:,1], z=Energy_E1_E2[:,2], elev=45, azim=45,\
                   x_lim=100, y_lim=100, color='k')
CF.Histogram_lineplot(Hist=Energy_sum, Bins=800, x_lim_low=20, x_lim_high=80, color='blue')
########################################################################################################

####################### DBSCAN Clustering and Plot the results #########################################
##### Model Fitting I       
# Set Fitting Parameters
eps, min_samples = (1, 2) 
model1 = DBSCAN( eps=eps, min_samples=min_samples )
model1.fit( Energy_E1_E2 )
y_hat1 = model1.labels_

core_indices1 = np.zeros_like(y_hat1, dtype=bool) # create zero/boolean array with the same length
core_indices1[model1.core_sample_indices_] = True # 核样本的目录 < (label != 0)

y_unique1 = np.unique(y_hat1) # extract different Labels
n_clusters1 = y_unique1.size - (1 if -1 in y_hat1 else 0)
print(y_unique1, 'clustering number is :', n_clusters1)

# Plot the DBSCAN clustering results
fig = plt.figure(figsize=(12, 12), facecolor='w')
ax = Axes3D(fig)
elev=45; azim=45
for k in y_unique1:
    cur = (y_hat1 == k)
    if k == -1:
        CF.Scatter3D_plot(ax=ax, x=Energy_E1_E2[cur,0], y=Energy_E1_E2[cur,1],\
                           z=Energy_E1_E2[cur,2], elev=elev, azim=azim, x_lim=100, y_lim=100, color = 'k')
        continue
    CF.Scatter3D_plot(ax=ax, x=Energy_E1_E2[cur,0], y=Energy_E1_E2[cur,1],\
                           z=Energy_E1_E2[cur,2], elev=elev, azim=azim, x_lim=100, y_lim=100, color = 'r')
    CF.Scatter3D_plot(ax=ax, x=Energy_E1_E2[cur & core_indices1,0], y=Energy_E1_E2[cur & core_indices1,1],\
                           z=Energy_E1_E2[cur & core_indices1,2], elev=elev, azim=azim, x_lim=100, y_lim=100, color = 'r')

###### check each DBSCAN clustering results
fig = plt.figure(figsize=(12, 12), facecolor='w')
ax = Axes3D(fig)
for k in y_unique1:
    cur = (y_hat1 == k)
    if k == -1:
        CF.Scatter3D_plot(ax=ax, x=Energy_E1_E2[cur,0], y=Energy_E1_E2[cur,1],\
                           z=Energy_E1_E2[cur,2], elev=45, azim=45, x_lim=55, y_lim=55, color = 'k')
        continue
########################################################################################################

####################### Extract the cluster in the ROI #################################################
##### Reorganize the clustered scattering points
y_hat = np.array([-1]*len(y_hat1))
y_hat[np.where(y_hat1 != -1),] = 0
y_unique = np.unique(y_hat)
cluster_lab_0 = Energy_E1_E2[np.where( ( y_hat == 0 ) )]
###### check each DBSCAN clustering results
fig = plt.figure(figsize=(12, 12), facecolor='w')
ax = Axes3D(fig)
for k in y_unique:
    cur = (y_hat == k)
    if k == -1:
        CF.Scatter3D_plot(ax=ax, x=Energy_E1_E2[cur,0], y=Energy_E1_E2[cur,1],\
                           z=Energy_E1_E2[cur,2], elev=45, azim=45, x_lim=140, y_lim=140, color = 'k')
        continue
    if k == 0:
        CF.Scatter3D_plot(ax=ax, x=Energy_E1_E2[cur,0], y=Energy_E1_E2[cur,1],\
                           z=Energy_E1_E2[cur,2], elev=45, azim=45, x_lim=140, y_lim=140, color = 'b')
        continue
########################################################################################################  

####################### "Rot -> MC Shifting -> Rot" CSC function #######################################
##### Initialize the CSC object, charge sharing band correction
seg_size = 4
CSC_81_3pix = CSC( CS_dim=CS_dim, basis_old=basis_old, basis_new=basis_new, peak_energy=81, max_energy_range=140, seg_size=seg_size )
wet_x_81, wet_y_81, wet_w_81, shift_w_81, seg_unit_81 = CSC_81_3pix.Pix3_Measurement( CS_data_labeled = cluster_lab_0 )
Energy81_corrected = CSC_81_3pix.Pix3_Correction(seg_unit=seg_unit_81, shift_w=shift_w_81, CS_data_labeled=Energy_E1_E2)

# check the scattering plot and MC plot
Fig81 = plt.figure(figsize=(12, 12), facecolor='w')
ax = Axes3D(Fig81)
CF.Scatter3D_plot(ax=ax, x=Energy81_corrected[:,0], y=Energy81_corrected[:,1], z=Energy81_corrected[:,2],\
                   elev=elev, azim=azim, x_lim=100, y_lim=100, color='red')

# check the histogram
Energy_sum = np.sum(Energy_E1_E2, axis=1).reshape(-1,1)
CF.Histogram_barplot(Hist=Energy_sum, Bins=80, x_lim_low=20, x_lim_high=140)
Energy_sum = np.sum(Energy81_corrected, axis=1).reshape(-1,1)
CF.Histogram_barplot(Hist=Energy_sum, Bins=80, x_lim_low=20, x_lim_high=140)

# check the MC of CS band
Fig81_surface = plt.figure(figsize=(12, 12), facecolor='w')
ax = Axes3D(Fig81_surface)
CF.Surface3D_plot(ax=ax, x=wet_x_81, y=wet_y_81, z=wet_w_81, elev=45, azim=45)
###############################################################################################################




###############################################################################################################
################################### Extract 122 keV events ####################################################
Energy_sum = np.sum( Energy, axis=1 ).reshape(-1,1)
Energy_E1_E2 = Energy[ np.intersect1d(np.where(Energy_sum >= energy_seg[energy_ind])[0],\
                                      np.where(Energy_sum <= energy_seg[energy_ind+1])[0]) ]
energy_ind += 1
Energy_sum = np.sum( Energy_E1_E2, axis=1 ).reshape(-1,1)
# scatter plot and histogram
fig = plt.figure(figsize=(12, 12), facecolor='w'); ax = Axes3D(fig);
CF.Scatter3D_plot(ax=ax, x=Energy_E1_E2[:,0], y=Energy_E1_E2[:,1], z=Energy_E1_E2[:,2], elev=45, azim=45,\
                   x_lim=130, y_lim=130, color='k')
CF.Histogram_lineplot(Hist=Energy_sum, Bins=800, x_lim_low=20, x_lim_high=140, color='blue')
###############################################################################################################

####################### DBSCAN Clustering and Plot the results ################################################
##### Model Fitting I       
# Set Fitting Parameters
eps, min_samples = (2, 6) # 122 keV, high density CS events

model1 = DBSCAN( eps=eps, min_samples=min_samples )
model1.fit( Energy_E1_E2 )
y_hat1 = model1.labels_

core_indices1 = np.zeros_like(y_hat1, dtype=bool) # create zero/boolean array with the same length
core_indices1[model1.core_sample_indices_] = True # 核样本的目录 < (label != 0)

y_unique1 = np.unique(y_hat1) # extract different Labels
n_clusters1 = y_unique1.size - (1 if -1 in y_hat1 else 0)
print(y_unique1, 'clustering number is :', n_clusters1)

# Plot the DBSCAN clustering results
fig = plt.figure(figsize=(12, 12), facecolor='w')
ax = Axes3D(fig)
elev=45; azim=45
for k in y_unique1:
    cur = (y_hat1 == k)
    if k == -1:
        CF.Scatter3D_plot(ax=ax, x=Energy_E1_E2[cur,0], y=Energy_E1_E2[cur,1],\
                           z=Energy_E1_E2[cur,2], elev=elev, azim=azim, x_lim=140, y_lim=140, color = 'k')
        continue
    CF.Scatter3D_plot(ax=ax, x=Energy_E1_E2[cur,0], y=Energy_E1_E2[cur,1],\
                           z=Energy_E1_E2[cur,2], elev=elev, azim=azim, x_lim=140, y_lim=140, color = 'r')
    CF.Scatter3D_plot(ax=ax, x=Energy_E1_E2[cur & core_indices1,0], y=Energy_E1_E2[cur & core_indices1,1],\
                           z=Energy_E1_E2[cur & core_indices1,2], elev=elev, azim=azim, x_lim=140, y_lim=140, color = 'r')

###### check each DBSCAN clustering results
fig = plt.figure(figsize=(12, 12), facecolor='w')
ax = Axes3D(fig)
for k in y_unique1:
    cur = (y_hat1 == k)
    if k == -1:
        CF.Scatter3D_plot(ax=ax, x=Energy_E1_E2[cur,0], y=Energy_E1_E2[cur,1],\
                           z=Energy_E1_E2[cur,2], elev=45, azim=45, x_lim=140, y_lim=140, color = 'k')
        continue
################################################################################################################

####################### Extract the cluster in the ROI #########################################################
##### Reorganize the clustered scattering points
y_hat = np.array([-1]*len(y_hat1))
y_hat[np.where(y_hat1 != -1),] = 0
y_unique = np.unique(y_hat)
cluster_lab_0 = Energy_E1_E2[np.where( ( y_hat == 0 ) )]

###### check each DBSCAN clustering results
fig = plt.figure(figsize=(12, 12), facecolor='w')
ax = Axes3D(fig)
for k in y_unique:
    cur = (y_hat == k)
    if k == -1:
        CF.Scatter3D_plot(ax=ax, x=Energy_E1_E2[cur,0], y=Energy_E1_E2[cur,1],\
                           z=Energy_E1_E2[cur,2], elev=45, azim=45, x_lim=140, y_lim=140, color = 'k')
        continue
    if k == 0:
        CF.Scatter3D_plot(ax=ax, x=Energy_E1_E2[cur,0], y=Energy_E1_E2[cur,1],\
                           z=Energy_E1_E2[cur,2], elev=45, azim=45, x_lim=140, y_lim=140, color = 'b')
        continue
################################################################################################################

####################### "Rot -> MC Shifting -> Rot" CSC function ###############################################
##### Initialize the CSC object, charge sharing band correction
seg_size = 4
CSC_122_3pix = CSC( CS_dim=CS_dim, basis_old=basis_old, basis_new=basis_new, peak_energy=122, max_energy_range=140, seg_size=seg_size )
wet_x_122, wet_y_122, wet_w_122, shift_w_122, seg_unit_122 = CSC_122_3pix.Pix3_Measurement( CS_data_labeled = cluster_lab_0 )
Energy122_corrected = CSC_122_3pix.Pix3_Correction(seg_unit=seg_unit_122, shift_w=shift_w_122, CS_data_labeled=Energy_E1_E2)

# check the scattering plot and MC plot
Fig122 = plt.figure(figsize=(12, 12), facecolor='w')
ax = Axes3D(Fig122)
CF.Scatter3D_plot(ax=ax, x=Energy122_corrected[:,0], y=Energy122_corrected[:,1], z=Energy122_corrected[:,2],\
                   elev=elev, azim=azim, x_lim=140, y_lim=140, color='red')

# check the histogram
Energy_sum = np.sum(Energy_E1_E2, axis=1).reshape(-1,1)
CF.Histogram_barplot(Hist=Energy_sum, Bins=80, x_lim_low=20, x_lim_high=140)
Energy_sum = np.sum(Energy122_corrected, axis=1).reshape(-1,1)
CF.Histogram_barplot(Hist=Energy_sum, Bins=80, x_lim_low=20, x_lim_high=140)

# check the MC of CS band
Fig122_surface = plt.figure(figsize=(12, 12), facecolor='w')
ax = Axes3D(Fig122_surface)
CF.Surface3D_plot(ax=ax, x=wet_x, y=wet_y, z=wet_w, elev=45, azim=45)
###############################################################################################################




###############################################################################################################
################################### Extract 136 keV events ####################################################
Energy_sum = np.sum( Energy, axis=1 ).reshape(-1,1)
Energy_E1_E2 = Energy[ np.intersect1d(np.where(Energy_sum >= energy_seg[energy_ind])[0],\
                                      np.where(Energy_sum <= energy_seg[energy_ind+1])[0]) ]
energy_ind += 1
Energy_sum = np.sum( Energy_E1_E2, axis=1 ).reshape(-1,1)
# scatter plot and histogram
fig = plt.figure(figsize=(12, 12), facecolor='w'); ax = Axes3D(fig);
CF.Scatter3D_plot(ax=ax, x=Energy_E1_E2[:,0], y=Energy_E1_E2[:,1], z=Energy_E1_E2[:,2], elev=45, azim=45,\
                   x_lim=130, y_lim=130, color='k')
CF.Histogram_lineplot(Hist=Energy_sum, Bins=800, x_lim_low=20, x_lim_high=140, color='blue')
###############################################################################################################

####################### DBSCAN Clustering and Plot the results ################################################
##### Model Fitting I       
# Set Fitting Parameters
eps, min_samples = (2, 4) # 122 keV, high density CS events

model1 = DBSCAN( eps=eps, min_samples=min_samples )
model1.fit( Energy_E1_E2 )
y_hat1 = model1.labels_

core_indices1 = np.zeros_like(y_hat1, dtype=bool) # create zero/boolean array with the same length
core_indices1[model1.core_sample_indices_] = True # 核样本的目录 < (label != 0)

y_unique1 = np.unique(y_hat1) # extract different Labels
n_clusters1 = y_unique1.size - (1 if -1 in y_hat1 else 0)
print(y_unique1, 'clustering number is :', n_clusters1)

# Plot the DBSCAN clustering results
fig = plt.figure(figsize=(12, 12), facecolor='w')
ax = Axes3D(fig)
elev=45; azim=45
for k in y_unique1:
    cur = (y_hat1 == k)
    if k == -1:
        CF.Scatter3D_plot(ax=ax, x=Energy_E1_E2[cur,0], y=Energy_E1_E2[cur,1],\
                           z=Energy_E1_E2[cur,2], elev=elev, azim=azim, x_lim=140, y_lim=140, color = 'k')
        continue
    CF.Scatter3D_plot(ax=ax, x=Energy_E1_E2[cur,0], y=Energy_E1_E2[cur,1],\
                           z=Energy_E1_E2[cur,2], elev=elev, azim=azim, x_lim=140, y_lim=140, color = 'r')
    CF.Scatter3D_plot(ax=ax, x=Energy_E1_E2[cur & core_indices1,0], y=Energy_E1_E2[cur & core_indices1,1],\
                           z=Energy_E1_E2[cur & core_indices1,2], elev=elev, azim=azim, x_lim=140, y_lim=140, color = 'r')

###### check each DBSCAN clustering results
fig = plt.figure(figsize=(12, 12), facecolor='w')
ax = Axes3D(fig)
for k in y_unique1:
    cur = (y_hat1 == k)
    if k == -1:
        CF.Scatter3D_plot(ax=ax, x=Energy_E1_E2[cur,0], y=Energy_E1_E2[cur,1],\
                           z=Energy_E1_E2[cur,2], elev=45, azim=45, x_lim=140, y_lim=140, color = 'k')
        continue
################################################################################################################

####################### Extract the cluster in the ROI #########################################################
##### Reorganize the clustered scattering points
y_hat = np.array([-1]*len(y_hat1))
y_hat[np.where(y_hat1 != -1),] = 0
y_unique = np.unique(y_hat)
cluster_lab_0 = Energy_E1_E2[np.where( ( y_hat == 0 ) )]

###### check each DBSCAN clustering results
fig = plt.figure(figsize=(12, 12), facecolor='w')
ax = Axes3D(fig)
for k in y_unique:
    cur = (y_hat == k)
    if k == -1:
        CF.Scatter3D_plot(ax=ax, x=Energy_E1_E2[cur,0], y=Energy_E1_E2[cur,1],\
                           z=Energy_E1_E2[cur,2], elev=45, azim=45, x_lim=140, y_lim=140, color = 'k')
        continue
    if k == 0:
        CF.Scatter3D_plot(ax=ax, x=Energy_E1_E2[cur,0], y=Energy_E1_E2[cur,1],\
                           z=Energy_E1_E2[cur,2], elev=45, azim=45, x_lim=140, y_lim=140, color = 'b')
        continue
################################################################################################################

####################### "Rot -> MC Shifting -> Rot" CSC function ###############################################
##### Initialize the CSC object, charge sharing band correction
seg_size = 4
CSC_136_3pix = CSC( CS_dim=CS_dim, basis_old=basis_old, basis_new=basis_new, peak_energy=136, max_energy_range=140, seg_size=seg_size )
wet_x_136, wet_y_136, wet_w_136, shift_w_136, seg_unit_136 = CSC_136_3pix.Pix3_Measurement( CS_data_labeled = cluster_lab_0 )
Energy136_corrected = CSC_136_3pix.Pix3_Correction(seg_unit=seg_unit_136, shift_w=shift_w_136, CS_data_labeled=Energy_E1_E2)

# check the scattering plot and MC plot
Fig136 = plt.figure(figsize=(12, 12), facecolor='w')
ax = Axes3D(Fig136)
CF.Scatter3D_plot(ax=ax, x=Energy136_corrected[:,0], y=Energy136_corrected[:,1], z=Energy136_corrected[:,2],\
                   elev=elev, azim=azim, x_lim=140, y_lim=140, color='red')

# check the histogram
Energy_sum = np.sum(Energy_E1_E2, axis=1).reshape(-1,1)
CF.Histogram_barplot(Hist=Energy_sum, Bins=80, x_lim_low=20, x_lim_high=140)
Energy_sum = np.sum(Energy136_corrected, axis=1).reshape(-1,1)
CF.Histogram_barplot(Hist=Energy_sum, Bins=80, x_lim_low=20, x_lim_high=140)

# check the MC of CS band
Fig122_surface = plt.figure(figsize=(12, 12), facecolor='w')
ax = Axes3D(Fig122_surface)
CF.Surface3D_plot(ax=ax, x=wet_x_136, y=wet_y_136, z=wet_w_136, elev=45, azim=45)
##############################################################################################################



########################### Finally check the energy spectra #################################################
Energy_sum = np.sum(Energy, axis = 1).reshape(-1,1)
Energy_corrected = np.vstack((Energy36_corrected, Energy60_corrected, Energy81_corrected, Energy122_corrected, Energy136_corrected))
Energy_corrected_sum = np.sum(Energy_corrected, axis = 1).reshape(-1,1)
CF.Histogram_lineplot(Hist=Energy_corrected_sum, Bins=900, x_lim_low=20, x_lim_high=150,color='red')
CF.Histogram_lineplot(Hist=Energy_sum, Bins=900, x_lim_low=20, x_lim_high=150,color='blue')

##### save the corrected events
CF.SaveFiles(var=Energy_corrected, dim=3, var_name='d', \
              location="C:\Jiajin\Mfile\Training_Sample_Analysis\FullRangeCorrected_3pix.csv")

##### Finally filter the loss plane and compensation plane 
import heapq
import gc

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
plt.matshow(wet_w_60)
plt.matshow(wet_w_60_filtered)

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
plt.matshow(wet_w_81)
plt.matshow(wet_w_81_filtered)

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
plt.matshow(wet_w_122)
plt.matshow(wet_w_122_filtered)



plt.matshow(shift_w_60)
plt.matshow(shift_w_81)
plt.matshow(shift_w_122)

### Surface plot1
Fig122_surface = plt.figure(figsize=(12, 12), facecolor='w')
ax = Axes3D(Fig122_surface)
CF.Surface3D_plot(ax=ax, x=wet_x_136, y=wet_y_136, z=wet_w_60_filtered, elev=45, azim=45)

Fig122_surface = plt.figure(figsize=(12, 12), facecolor='w')
ax = Axes3D(Fig122_surface)
CF.Surface3D_plot(ax=ax, x=wet_x_122, y=wet_y_122, z=wet_w_81_filtered, elev=45, azim=45)

Fig122_surface = plt.figure(figsize=(12, 12), facecolor='w')
ax = Axes3D(Fig122_surface)
CF.Surface3D_plot(ax=ax, x=wet_x_122, y=wet_y_122, z=wet_w_122_filtered, elev=45, azim=45)


### Surface plot2
Fig122_surface = plt.figure(figsize=(12, 12), facecolor='w')
ax = Axes3D(Fig122_surface)
CF.Surface3D_plot(ax=ax, x=wet_x_136, y=wet_y_136, z=wet_w_60, elev=45, azim=45)

Fig122_surface = plt.figure(figsize=(12, 12), facecolor='w')
ax = Axes3D(Fig122_surface)
CF.Surface3D_plot(ax=ax, x=wet_x_122, y=wet_y_122, z=wet_w_81, elev=45, azim=45)

Fig122_surface = plt.figure(figsize=(12, 12), facecolor='w')
ax = Axes3D(Fig122_surface)
CF.Surface3D_plot(ax=ax, x=wet_x_122, y=wet_y_122, z=wet_w_122, elev=45, azim=45)


### Scatter plot1
Fig122_surface = plt.figure(figsize=(12, 12), facecolor='w')
ax = Axes3D(Fig122_surface)
CF.Scatter3D_plot(ax=ax, x=wet_x_122, y=wet_y_122,\
                   z=wet_w_122, elev=45, azim=45, color = 'r')

Fig122_surface = plt.figure(figsize=(12, 12), facecolor='w')
ax = Axes3D(Fig122_surface)
CF.Scatter3D_plot(ax=ax, x=wet_x_122, y=wet_y_122,\
                   z=wet_w_122_filtered, elev=45, azim=45, color = 'b')

###
Fig122_surface = plt.figure(figsize=(12, 12), facecolor='w')
ax = Axes3D(Fig122_surface)
CF.Scatter3D_plot(ax=ax, x=wet_x_122, y=wet_y_122,\
                   z=wet_w_81, elev=45, azim=45, color = 'r')

Fig122_surface = plt.figure(figsize=(12, 12), facecolor='w')
ax = Axes3D(Fig122_surface)
CF.Scatter3D_plot(ax=ax, x=wet_x_122, y=wet_y_122,\
                   z=wet_w_81_filtered, elev=45, azim=45, color = 'b')

###
Fig122_surface = plt.figure(figsize=(12, 12), facecolor='w')
ax = Axes3D(Fig122_surface)
CF.Scatter3D_plot(ax=ax, x=wet_x_122, y=wet_y_122,\
                   z=wet_w_60, elev=45, azim=45, color = 'r')

Fig122_surface = plt.figure(figsize=(12, 12), facecolor='w')
ax = Axes3D(Fig122_surface)
CF.Scatter3D_plot(ax=ax, x=wet_x_122, y=wet_y_122,\
                   z=wet_w_60_filtered, elev=45, azim=45, color = 'b')



### Scatter plot2
Fig122_surface = plt.figure(figsize=(12, 12), facecolor='w')
ax = Axes3D(Fig122_surface)
CF.Scatter3D_plot(ax=ax, x=wet_x_122, y=wet_y_122,\
                   z=wet_w_60_filtered, elev=20, azim=45, color = 'r')
CF.Scatter3D_plot(ax=ax, x=wet_x_122, y=wet_y_122,\
                   z=wet_w_81_filtered, elev=20, azim=45, color = 'b')
CF.Scatter3D_plot(ax=ax, x=wet_x_122, y=wet_y_122,\
                   z=wet_w_122_filtered, elev=20, azim=45, color = 'k')

###
Fig122_surface = plt.figure(figsize=(12, 12), facecolor='w')
ax = Axes3D(Fig122_surface)
CF.Scatter3D_plot(ax=ax, x=wet_x_122, y=wet_y_122,\
                   z=wet_w_60, elev=20, azim=45, color = 'r')
CF.Scatter3D_plot(ax=ax, x=wet_x_122, y=wet_y_122,\
                   z=wet_w_81, elev=20, azim=45, color = 'b')
CF.Scatter3D_plot(ax=ax, x=wet_x_122, y=wet_y_122,\
                   z=wet_w_122, elev=20, azim=45, color = 'k')

