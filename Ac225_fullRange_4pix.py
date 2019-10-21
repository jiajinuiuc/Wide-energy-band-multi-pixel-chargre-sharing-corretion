# -*- coding: utf-8 -*-
"""
fllRangeCorrection_v1
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
CS_data = pd.read_csv( 'C:\Jiajin\ChaSha_2017\CSC_Data\JHU_data\Ac225_Pix4_Sharing.csv' )
Energy = CS_data.iloc[:, :].values  
CS_dim = Energy.shape[1]

### Initialize basis parameters
basis_old =np.mat( [ [1,0,0,0],
                     [0,1,0,0],
                     [0,0,1,0],
                     [0,0,0,1] ] ) #[x, y, z]

basis_new = np.mat( [ [ 1/np.sqrt(4),   0,              1/np.sqrt(2),  1/np.sqrt(4) ],
                      [ 1/np.sqrt(4),   0,             -1/np.sqrt(2),  1/np.sqrt(4) ],
                      [-1/np.sqrt(4),   1/np.sqrt(2),   0,             1/np.sqrt(4) ],
                      [-1/np.sqrt(4),  -1/np.sqrt(2),   0,             1/np.sqrt(4) ] ] ) 

########################################################################################################





########################################################################################################
################################### Extract 61 keV events ##############################################

Energy_tmp = Energy[1:1000000, :]
Energy_sum_tmp = np.sum( Energy_tmp, axis=1 ).reshape(-1,1)
Energy_E1_E2 = Energy_tmp[ np.intersect1d(np.where(Energy_sum_tmp >= 54)[0], np.where(Energy_sum_tmp <= 65.3)[0]) ]
Energy_sum_tmp = np.sum( Energy_E1_E2, axis=1 ).reshape(-1,1)

# scatter plot and histogram
fig = plt.figure(figsize=(12, 12), facecolor='w'); ax = Axes3D(fig);
CF.Scatter3D_plot(ax=ax, x=Energy_E1_E2[:,0], y=Energy_E1_E2[:,1], z=Energy_E1_E2[:,2], elev=0, azim=45, color='k', marker='.')
CF.Histogram_lineplot(Hist=Energy_sum_tmp, Bins=800, x_lim_low=20, x_lim_high=140, color='blue')

########################################################################################################

####################### DBSCAN Clustering and Plot the results #########################################
##### Model Fitting I
# Set Fitting Parameters
eps, min_samples = (2, 15) # 122 keV, high density CS events
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
                           z=Energy_E1_E2[cur,2], elev=elev, azim=azim, color = 'k')
        continue
    CF.Scatter3D_plot(ax=ax, x=Energy_E1_E2[cur,0], y=Energy_E1_E2[cur,1],\
                           z=Energy_E1_E2[cur,2], elev=elev, azim=azim, color = 'r')
    CF.Scatter3D_plot(ax=ax, x=Energy_E1_E2[cur & core_indices1,0], y=Energy_E1_E2[cur & core_indices1,1],\
                           z=Energy_E1_E2[cur & core_indices1,2], elev=elev, azim=azim, color = 'r')

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
        CF.Scatter3D_plot(ax=ax, x=Energy_E1_E2[cur,0], y=Energy_E1_E2[cur,1],\
                           z=Energy_E1_E2[cur,2], elev=45, azim=45, color = 'k')
        continue
    if k == 0:
        CF.Scatter3D_plot(ax=ax, x=Energy_E1_E2[cur,0], y=Energy_E1_E2[cur,1],\
                           z=Energy_E1_E2[cur,2], elev=45, azim=45, color = 'b')
        continue
########################################################################################################  


####################### "Rot -> MC Shifting -> Rot" CSC function #######################################
##### Initialize the CSC object, charge sharing band correction
seg_size = 6
CSC_61_4pix = CSC( CS_dim=CS_dim, basis_old=basis_old, basis_new=basis_new, peak_energy=61, max_energy_range=600, seg_size=seg_size )
wet_x_61, wet_y_61, wet_z_61, wet_w_61, shift_w_61, seg_unit_61 = CSC_61_4pix.Pix4_Measurement( CS_data_labeled = cluster_lab_0 )
Energy61_corrected = CSC_61_4pix.Pix4_Correction(seg_unit=seg_unit_61, shift_w=shift_w_61, CS_data_labeled=Energy_E1_E2)

# check the scattering plot and MC plot
Fig36 = plt.figure(figsize=(12, 12), facecolor='w')
ax = Axes3D(Fig36)
CF.Scatter3D_plot(ax=ax, x=Energy61_corrected[:,0], y=Energy61_corrected[:,1], z=Energy61_corrected[:,2],\
                   elev=elev, azim=azim, color='red')

# check the histogram
Energy_sum = np.sum(Energy_E1_E2, axis=1).reshape(-1,1)
CF.Histogram_barplot(Hist=Energy_sum, Bins=80, x_lim_low=20, x_lim_high=140)
Energy_sum = np.sum(Energy61_corrected, axis=1).reshape(-1,1)
CF.Histogram_barplot(Hist=Energy_sum, Bins=80, x_lim_low=20, x_lim_high=140)

# check the MC of CS band
Fig36_surface = plt.figure(figsize=(12, 12), facecolor='w')
ax = Axes3D(Fig36_surface)
CF.Surface3D_plot(ax=ax, x=wet_x_61, y=wet_y_61, z=wet_w_61, elev=45, azim=45)
###############################################################################################################





########################################################################################################
################################### Extract 225-keV events ##############################################

Energy_tmp = Energy[1:12700000,:]
Energy_sum_tmp = np.sum( Energy_tmp, axis=1 ).reshape(-1,1)
Energy_E1_E2 = Energy_tmp[ np.intersect1d(np.where(Energy_sum_tmp >= 195)[0],np.where(Energy_sum_tmp <= 230)[0]) ]
Energy_sum_tmp = np.sum( Energy_E1_E2, axis=1 ).reshape(-1,1)

# scatter plot and histogram
fig = plt.figure(figsize=(12, 12), facecolor='w'); ax = Axes3D(fig);
CF.Scatter3D_plot(ax=ax, x=Energy_E1_E2[:,0], y=Energy_E1_E2[:,1], z=Energy_E1_E2[:,2], elev=0, azim=45, color='k', marker='.')
CF.Histogram_lineplot(Hist=Energy_sum_tmp, Bins=800, x_lim_low=20, x_lim_high=440, color='blue')

########################################################################################################

####################### DBSCAN Clustering and Plot the results #########################################
##### Model Fitting I
# Set Fitting Parameters
eps, min_samples =  (2.5, 3) # 122 keV, high density CS events
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
                           z=Energy_E1_E2[cur,2], elev=elev, azim=azim, color = 'k')
        continue
    CF.Scatter3D_plot(ax=ax, x=Energy_E1_E2[cur,0], y=Energy_E1_E2[cur,1],\
                           z=Energy_E1_E2[cur,2], elev=elev, azim=azim, color = 'r')
    CF.Scatter3D_plot(ax=ax, x=Energy_E1_E2[cur & core_indices1,0], y=Energy_E1_E2[cur & core_indices1,1],\
                           z=Energy_E1_E2[cur & core_indices1,2], elev=elev, azim=azim, color = 'r')

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
        CF.Scatter3D_plot(ax=ax, x=Energy_E1_E2[cur,0], y=Energy_E1_E2[cur,1],\
                           z=Energy_E1_E2[cur,2], elev=45, azim=45, color = 'k')
        continue
    if k == 0:
        CF.Scatter3D_plot(ax=ax, x=Energy_E1_E2[cur,0], y=Energy_E1_E2[cur,1],\
                           z=Energy_E1_E2[cur,2], elev=45, azim=45, color = 'b')
        continue
########################################################################################################  


####################### "Rot -> MC Shifting -> Rot" CSC function #######################################
##### Initialize the CSC object, charge sharing band correction
seg_size = 6
CSC_225_4pix = CSC( CS_dim=CS_dim, basis_old=basis_old, basis_new=basis_new, peak_energy=225, max_energy_range=600, seg_size=seg_size )
wet_x_225, wet_y_225, wet_z_225, wet_w_225, shift_w_225, seg_unit_225 = CSC_225_4pix.Pix4_Measurement( CS_data_labeled = cluster_lab_0 )
Energy225_corrected = CSC_225_4pix.Pix4_Correction(seg_unit=seg_unit_225, shift_w=shift_w_225, CS_data_labeled=Energy_E1_E2)

# check the scattering plot and MC plot
Fig36 = plt.figure(figsize=(12, 12), facecolor='w')
ax = Axes3D(Fig36)
CF.Scatter3D_plot(ax=ax, x=Energy225_corrected[:,0], y=Energy225_corrected[:,1], z=Energy225_corrected[:,2],\
                   elev=elev, azim=azim, color='red')

# check the histogram
Energy_sum = np.sum(Energy_E1_E2, axis=1).reshape(-1,1)
CF.Histogram_barplot(Hist=Energy_sum, Bins=80, x_lim_low=20, x_lim_high=440)
Energy_sum = np.sum(Energy225_corrected, axis=1).reshape(-1,1)
CF.Histogram_barplot(Hist=Energy_sum, Bins=80, x_lim_low=20, x_lim_high=440)

###############################################################################################################






###############################################################################################################
################################### Extract 440-keV events ####################################################

Energy_tmp = Energy[1:12700000,:]
Energy_sum_tmp = np.sum( Energy_tmp, axis=1 ).reshape(-1,1)
Energy_E1_E2 = Energy_tmp[ np.intersect1d(np.where(Energy_sum_tmp >= 400)[0],np.where(Energy_sum_tmp <= 460)[0]) ]
Energy_sum_tmp = np.sum( Energy_E1_E2, axis=1 ).reshape(-1,1)

# scatter plot and histogram
fig = plt.figure(figsize=(12, 12), facecolor='w'); ax = Axes3D(fig);
CF.Scatter3D_plot(ax=ax, x=Energy_E1_E2[:,0], y=Energy_E1_E2[:,1], z=Energy_E1_E2[:,2], elev=45, azim=45,color='k')
CF.Histogram_lineplot(Hist=Energy_sum_tmp, Bins=800, x_lim_low=20, x_lim_high=500, color='blue')
###############################################################################################################

####################### DBSCAN Clustering and Plot the results ################################################
##### Model Fitting I       
# Set Fitting Parameters
eps, min_samples = (8, 2) # 122 keV, high density CS events

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
                           z=Energy_E1_E2[cur,2], elev=elev, azim=azim, color = 'k')
        continue
    CF.Scatter3D_plot(ax=ax, x=Energy_E1_E2[cur,0], y=Energy_E1_E2[cur,1],\
                           z=Energy_E1_E2[cur,2], elev=elev, azim=azim, color = 'r')
    CF.Scatter3D_plot(ax=ax, x=Energy_E1_E2[cur & core_indices1,0], y=Energy_E1_E2[cur & core_indices1,1],\
                           z=Energy_E1_E2[cur & core_indices1,2], elev=elev, azim=azim, color = 'r')

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
seg_size = 6
CSC_440_4pix = CSC( CS_dim=CS_dim, basis_old=basis_old, basis_new=basis_new, peak_energy=440, max_energy_range=600, seg_size=seg_size )
wet_x_440, wet_y_440, wet_z_440, wet_w_440, shift_w_440, seg_unit_440 = CSC_440_4pix.Pix4_Measurement( CS_data_labeled = cluster_lab_0 )
Energy440_corrected = CSC_440_4pix.Pix4_Correction(seg_unit=seg_unit_440, shift_w=shift_w_440, CS_data_labeled=Energy_E1_E2)

# check the scattering plot and MC plot
Fig136 = plt.figure(figsize=(12, 12), facecolor='w')
ax = Axes3D(Fig136)
CF.Scatter3D_plot(ax=ax, x=Energy136_corrected[:,0], y=Energy136_corrected[:,1], z=Energy136_corrected[:,2],\
                   elev=elev, azim=azim, x_lim=140, y_lim=140, color='red')

# check the histogram
Energy_sum = np.sum(Energy_E1_E2, axis=1).reshape(-1,1)
CF.Histogram_barplot(Hist=Energy_sum, Bins=180, x_lim_low=400, x_lim_high=460)
Energy_sum = np.sum(Energy440_corrected, axis=1).reshape(-1,1)
CF.Histogram_barplot(Hist=Energy_sum, Bins=180, x_lim_low=400, x_lim_high=460)

# check the MC of CS band
Fig122_surface = plt.figure(figsize=(12, 12), facecolor='w')
ax = Axes3D(Fig122_surface)
CF.Surface3D_plot(ax=ax, x=wet_x_136, y=wet_y_136, z=wet_w_136, elev=45, azim=45)
##############################################################################################################




###############################################################################################################
##################################### Full range correction ###################################################

Energy_sum = np.sum(Energy, axis=1).reshape(-1,1)
Energy_else1 = Energy[ np.intersect1d( np.where(Energy_sum < 54)[0], np.where(Energy_sum > 25)  ) ]
Energy_else2 = Energy[ np.intersect1d( np.where(Energy_sum > 65.3)[0], np.where(Energy_sum < 195)[0] ), ]
Energy_else3 = Energy[ np.intersect1d( np.where(Energy_sum > 230)[0], np.where(Energy_sum < 400)[0] ), ]
Energy_else4 = Energy[ np.where(Energy_sum > 460)[0], ]

Energy_corrected = np.vstack((Energy_else1, Energy_else2, Energy_else3, Energy_else4,\
                              Energy61_corrected, Energy225_corrected, Energy440_corrected))

CF.Histogram_barplot(Hist=Energy_sum, Bins=380, x_lim_low=20, x_lim_high=460)
Energy_sum = np.sum(Energy_corrected, axis=1).reshape(-1,1)
CF.Histogram_barplot(Hist=Energy_sum, Bins=380, x_lim_low=20, x_lim_high=460)

var = Energy_corrected
dataframe = pd.DataFrame( var )
dataframe.columns = ['E1','E2', 'E3', 'E4']
dataframe.to_csv('C:\Jiajin\ChaSha_2017\CSC_Data\JHU_data\Ac225_Pix4_Sharing_Corrected.csv', index=False, sep=',')



