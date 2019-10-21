# -*- coding: utf-8 -*-
"""
fllRangeCorrection_v0
Objective： 
Full range energy spectrum correction， try with 2-pixel charge sharing events
0. charge sharing events clustering on different peaks from Ac, Bi, Fr, Pb, Tl
1. full "spatial" range segmentation and calculate the projection distance in each channel
3. calculate the projection distance of each charge sharing band at each channel
4. linear interpolation of the porjection distance between each band at different channel
5. based on the linear interpolation results, do the full range charge sharing correction
Version 0
@author: J. J. Zhang
Last update: June, 2019
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

###############################################################################################################
###### Initialization #########################################################################################
CS_data = pd.read_csv( 'C:\Jiajin\ChaSha_2017\CSC_Data\JHU_data\Ac225_Pix2_Sharing.csv' )
Energy = CS_data.iloc[:, :].values   #取第一行
CS_dim = Energy.shape[1]

Energy_sum = np.sum( Energy, axis=1 ).reshape(-1,1)
##### plot the histogram of sum_Energy in full energy range
CF.Histogram_lineplot(Hist=Energy_sum, Bins=1000, x_lim_high=600, x_lim_low=20, color='red')

basis_old =np.mat( [ [1,0],
                     [0,1] ] ) #[x, y]
basis_new = np.mat( [ [  1/np.sqrt(2),  1/np.sqrt(2) ],
                      [ -1/np.sqrt(2),  1/np.sqrt(2) ] ] )



################################################################################################################
################################### Load the 61-keV charge sharing events ######################################

Energy_tmp = Energy[1:350000,:]
Energy_sum_tmp = np.sum( Energy_tmp, axis=1 ).reshape(-1,1)
Energy_E1_E2 = Energy_tmp[ np.intersect1d(np.where(Energy_sum_tmp >= 54)[0],np.where(Energy_sum_tmp <= 65.3)[0]) ]
Energy_sum_tmp = np.sum( Energy_E1_E2, axis=1 ).reshape(-1,1)

##### plot the histogram of sum_Energy within the selected ROI
CF.Histogram_lineplot(Hist=Energy_sum_tmp, Bins=300, x_lim_high=200, x_lim_low=20, color='red')

##### plot the raw scatter figures within the selected ROI
CF.Scatter2D_plot(x=Energy_E1_E2[:,0],y=Energy_E1_E2[:,1],x_lim_left=0,x_lim_right=70,y_lim_left=0,y_lim_right=70)
########################################################################################################


####################### DBSCAN Clustering and Plot the results #########################################
##### Model Fitting I       
# Set Fitting Parameters
eps, min_samples = (1, 103) # 122 keV, high density CS events
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
for k in y_unique1:
    cur = (y_hat1 == k)
    if k == -1:
        CF.Scatter2D_plot(x=Energy_E1_E2[cur,0], y=Energy_E1_E2[cur,1], x_lim_left=0,x_lim_right=70,\
                          y_lim_left=0,y_lim_right=70, color = 'k')
        continue
    CF.Scatter2D_plot(x=Energy_E1_E2[cur,0], y=Energy_E1_E2[cur,1],x_lim_left=0,x_lim_right=70,\
                      y_lim_left=0,y_lim_right=70, color = 'r')
    CF.Scatter2D_plot(x=Energy_E1_E2[cur & core_indices1,0], y=Energy_E1_E2[cur & core_indices1,1],x_lim_left=0,x_lim_right=70,\
                      y_lim_left=0,y_lim_right=70, color = 'r')

###### check each DBSCAN clustering results
fig = plt.figure(figsize=(12, 12), facecolor='w')
for k in y_unique1:
    cur = (y_hat1 == k)
    if k == 0:
        CF.Scatter2D_plot(x=Energy_E1_E2[cur,0], y=Energy_E1_E2[cur,1], x_lim_left=0,x_lim_right=70,\
                          y_lim_left=0,y_lim_right=70, color = 'k')
        continue
########################################################################################################
 
    
####################### Extract the cluster in the ROI #################################################
##### Reorganize the clustered scattering points
y_hat = np.array([-1]*len(y_hat1))
y_hat[np.where( y_hat1 != -1 ),] = 0
y_hat[reduce( np.intersect1d, (np.where(y_hat == 0),np.where(Energy_sum_tmp <= 55)) ), ] = -1

y_unique = np.unique(y_hat)
cluster_lab_0 = Energy_E1_E2[np.where( ( y_hat == 0 ) )]

###### check each DBSCAN clustering results
plt.figure(figsize=(12, 12), facecolor='w')
plt.grid('on')
for k in y_unique:
    cur = (y_hat == k)
    if k == -1:
        plt.scatter(Energy_E1_E2[cur, 0], Energy_E1_E2[cur, 1], s=2, c='k') # non-clustered points
        continue
    if k == 0:
        plt.scatter(Energy_E1_E2[cur, 0], Energy_E1_E2[cur, 1], s=7, c='b') # non-clustered points
        continue
#    plt.scatter(Energy_sub[cur, 0], Energy_sub[cur, 1], s=5, c=clr, edgecolors='k')
#    plt.scatter(Energy_sub[cur & core_indices2, 0], Energy_sub[cur & core_indices2, 1], s=10, c=clr, \
#                            marker='o', edgecolors='k')
plt.xlabel('Energy (keV)',fontsize=20)
plt.ylabel('Energy (keV)',fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlim(0, 70)
plt.ylim(0, 70)
plt.grid(True)
######################################################################################################## 
        

####################### "Rot -> MC Shifting -> Rot" CSC function #######################################
##### Initialize the CSC object, charge sharing band correction
seg_size = 2
CSC_61_2pix = CSC( CS_dim=CS_dim, basis_old=basis_old, basis_new=basis_new, peak_energy=61.7, max_energy_range=600, seg_size=seg_size )
wet_x_61, wet_w_61, shift_w_61, seg_unit_61 = CSC_61_2pix.Pix2_Measurement( CS_data_labeled = cluster_lab_0 )
Energy61_corrected = CSC_61_2pix.Pix2_Correction(seg_unit=seg_unit_61, shift_w=shift_w_61, CS_data_labeled = Energy_E1_E2)

# check the scattering plot and MC plot
plt.figure(figsize=(12, 12), facecolor='w')
CF.Scatter2D_plot(x=Energy61_corrected[:,0].tolist(),y=Energy61_corrected[:,1].tolist(),x_lim_left=0,x_lim_right=70,y_lim_left=0,y_lim_right=70)

# check the histogram
Energy_sum = np.sum(Energy_E1_E2, axis=1).reshape(-1,1)
CF.Histogram_barplot(Hist=Energy_sum, Bins=80, x_lim_low=20, x_lim_high=140)
Energy_sum = np.sum(Energy61_corrected, axis=1).reshape(-1,1)
CF.Histogram_barplot(Hist=Energy_sum, Bins=80, x_lim_low=20, x_lim_high=140)

# check the MC of CS band
###############################################################################################################






################################################################################################################
################################### Load the 78.9-keV charge sharing events ####################################

Energy_tmp = Energy[1:100000,:]
Energy_sum_tmp = np.sum( Energy_tmp, axis=1 ).reshape(-1,1)
Energy_E1_E2 = Energy_tmp[ np.intersect1d(np.where(Energy_sum_tmp >= 65.3)[0],np.where(Energy_sum_tmp <= 85)[0]) ]
Energy_sum_tmp = np.sum( Energy_E1_E2, axis=1 ).reshape(-1,1)

##### plot the histogram of sum_Energy within the selected ROI
CF.Histogram_lineplot(Hist=Energy_sum_tmp, Bins=300, x_lim_high=200, x_lim_low=20, color='red')

##### plot the raw scatter figures within the selected ROI
CF.Scatter2D_plot(x=Energy_E1_E2[:,0],y=Energy_E1_E2[:,1],x_lim_left=0,x_lim_right=86,y_lim_left=0,y_lim_right=86)
########################################################################################################


####################### DBSCAN Clustering and Plot the results #########################################
##### Model Fitting I       
# Set Fitting Parameters
eps, min_samples = (1, 63) # 122 keV, high density CS events
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
for k in y_unique1:
    cur = (y_hat1 == k)
    if k == -1:
        CF.Scatter2D_plot(x=Energy_E1_E2[cur,0], y=Energy_E1_E2[cur,1], x_lim_left=0,x_lim_right=83,\
                          y_lim_left=0,y_lim_right=83, color = 'k')
        continue
    CF.Scatter2D_plot(x=Energy_E1_E2[cur,0], y=Energy_E1_E2[cur,1],x_lim_left=0,x_lim_right=83,\
                      y_lim_left=0,y_lim_right=83, color = 'r')
    CF.Scatter2D_plot(x=Energy_E1_E2[cur & core_indices1,0], y=Energy_E1_E2[cur & core_indices1,1],x_lim_left=0,x_lim_right=83,\
                      y_lim_left=0,y_lim_right=83, color = 'r')

###### check each DBSCAN clustering results
fig = plt.figure(figsize=(12, 12), facecolor='w')
for k in y_unique1:
    cur = (y_hat1 == k)
    if k == 0:
        CF.Scatter2D_plot(x=Energy_E1_E2[cur,0], y=Energy_E1_E2[cur,1], x_lim_left=0,x_lim_right=70,\
                          y_lim_left=0,y_lim_right=70, color = 'k')
        continue
########################################################################################################
 

####################### Extract the cluster in the ROI #################################################
##### Reorganize the clustered scattering points
y_hat = np.array([-1]*len(y_hat1))
y_hat[np.where( y_hat1 != -1 ),] = 0
y_hat[reduce( np.intersect1d, (np.where(y_hat == 0),np.where(Energy_sum_tmp <= 70)) ), ] = -1

y_unique = np.unique(y_hat)
cluster_lab_0 = Energy_E1_E2[np.where( ( y_hat == 0 ) )]

###### check each DBSCAN clustering results
plt.figure(figsize=(12, 12), facecolor='w')
plt.grid('on')
for k in y_unique:
    cur = (y_hat == k)
    if k == -1:
        plt.scatter(Energy_E1_E2[cur, 0], Energy_E1_E2[cur, 1], s=2, c='k') # non-clustered points
        continue
    if k == 0:
        plt.scatter(Energy_E1_E2[cur, 0], Energy_E1_E2[cur, 1], s=7, c='b') # non-clustered points
        continue
#    plt.scatter(Energy_sub[cur, 0], Energy_sub[cur, 1], s=5, c=clr, edgecolors='k')
#    plt.scatter(Energy_sub[cur & core_indices2, 0], Energy_sub[cur & core_indices2, 1], s=10, c=clr, \
#                            marker='o', edgecolors='k')
plt.xlabel('Energy (keV)',fontsize=20)
plt.ylabel('Energy (keV)',fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlim(0, 83)
plt.ylim(0, 83)
plt.grid(True)
######################################################################################################## 
        

####################### "Rot -> MC Shifting -> Rot" CSC function #######################################
##### Initialize the CSC object, charge sharing band correction
seg_size = 2
CSC_79_2pix = CSC( CS_dim=CS_dim, basis_old=basis_old, basis_new=basis_new, peak_energy=78.9, max_energy_range=600, seg_size=seg_size )
wet_x_79, wet_w_79, shift_w_79, seg_unit_79 = CSC_79_2pix.Pix2_Measurement( CS_data_labeled = cluster_lab_0 )
Energy79_corrected = CSC_79_2pix.Pix2_Correction(seg_unit=seg_unit_79, shift_w=shift_w_79, CS_data_labeled = Energy_E1_E2)

# check the scattering plot and MC plot
plt.figure(figsize=(12, 12), facecolor='w')
CF.Scatter2D_plot(x=Energy79_corrected[:,0].tolist(),y=Energy79_corrected[:,1].tolist(),x_lim_left=0,x_lim_right=83,y_lim_left=0,y_lim_right=83)

# check the histogram
Energy_sum = np.sum(Energy_E1_E2, axis=1).reshape(-1,1)
CF.Histogram_barplot(Hist=Energy_sum, Bins=80, x_lim_low=20, x_lim_high=140)
Energy_sum = np.sum(Energy79_corrected, axis=1).reshape(-1,1)
CF.Histogram_barplot(Hist=Energy_sum, Bins=80, x_lim_low=20, x_lim_high=140)

# check the MC of CS band
###############################################################################################################





################################################################################################################
################################### Load the 100-keV charge sharing events ####################################

Energy_tmp = Energy[1:700000,:]
Energy_sum_tmp = np.sum( Energy_tmp, axis=1 ).reshape(-1,1)
Energy_E1_E2 = Energy_tmp[ np.intersect1d(np.where(Energy_sum_tmp >= 90)[0],np.where(Energy_sum_tmp <= 107)[0]) ]
Energy_sum_tmp = np.sum( Energy_E1_E2, axis=1 ).reshape(-1,1)

##### plot the histogram of sum_Energy within the selected ROI
CF.Histogram_lineplot(Hist=Energy_sum_tmp, Bins=300, x_lim_high=200, x_lim_low=20, color='red')

##### plot the raw scatter figures within the selected ROI
CF.Scatter2D_plot(x=Energy_E1_E2[:,0],y=Energy_E1_E2[:,1],x_lim_left=0,x_lim_right=107,y_lim_left=0,y_lim_right=107)
########################################################################################################


####################### DBSCAN Clustering and Plot the results #########################################
##### Model Fitting I       
# Set Fitting Parameters
eps, min_samples = (1, 133) # 122 keV, high density CS events
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
for k in y_unique1:
    cur = (y_hat1 == k)
    if k == -1:
        CF.Scatter2D_plot(x=Energy_E1_E2[cur,0], y=Energy_E1_E2[cur,1], x_lim_left=0,x_lim_right=107,\
                          y_lim_left=0,y_lim_right=107, color = 'k')
        continue
    CF.Scatter2D_plot(x=Energy_E1_E2[cur,0], y=Energy_E1_E2[cur,1],x_lim_left=0,x_lim_right=107,\
                      y_lim_left=0,y_lim_right=107, color = 'r')
    CF.Scatter2D_plot(x=Energy_E1_E2[cur & core_indices1,0], y=Energy_E1_E2[cur & core_indices1,1],x_lim_left=0,x_lim_right=107,\
                      y_lim_left=0,y_lim_right=107, color = 'r')

###### check each DBSCAN clustering results
fig = plt.figure(figsize=(12, 12), facecolor='w')
for k in y_unique1:
    cur = (y_hat1 == k)
    if k == 0:
        CF.Scatter2D_plot(x=Energy_E1_E2[cur,0], y=Energy_E1_E2[cur,1], x_lim_left=0,x_lim_right=70,\
                          y_lim_left=0,y_lim_right=70, color = 'k')
        continue
########################################################################################################


####################### Extract the cluster in the ROI #################################################
##### Reorganize the clustered scattering points
y_hat = np.array([-1]*len(y_hat1))
y_hat[np.where( y_hat1 != -1 ),] = 0

y_unique = np.unique(y_hat)
cluster_lab_0 = Energy_E1_E2[np.where( ( y_hat == 0 ) )]

###### check each DBSCAN clustering results
plt.figure(figsize=(12, 12), facecolor='w')
plt.grid('on')
for k in y_unique:
    cur = (y_hat == k)
    if k == -1:
        plt.scatter(Energy_E1_E2[cur, 0], Energy_E1_E2[cur, 1], s=2, c='k') # non-clustered points
        continue
    if k == 0:
        plt.scatter(Energy_E1_E2[cur, 0], Energy_E1_E2[cur, 1], s=7, c='b') # non-clustered points
        continue
#    plt.scatter(Energy_sub[cur, 0], Energy_sub[cur, 1], s=5, c=clr, edgecolors='k')
#    plt.scatter(Energy_sub[cur & core_indices2, 0], Energy_sub[cur & core_indices2, 1], s=10, c=clr, \
#                            marker='o', edgecolors='k')
plt.xlabel('Energy (keV)',fontsize=20)
plt.ylabel('Energy (keV)',fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlim(0, 107)
plt.ylim(0, 107)
plt.grid(True)
######################################################################################################## 
        

####################### "Rot -> MC Shifting -> Rot" CSC function #######################################
##### Initialize the CSC object, charge sharing band correction
seg_size = 2
CSC_100_2pix = CSC( CS_dim=CS_dim, basis_old=basis_old, basis_new=basis_new, peak_energy=96, max_energy_range=600, seg_size=seg_size )
wet_x_100, wet_w_100, shift_w_100, seg_unit_100 = CSC_100_2pix.Pix2_Measurement( CS_data_labeled = cluster_lab_0 )
Energy100_corrected = CSC_100_2pix.Pix2_Correction(seg_unit=seg_unit_100, shift_w=shift_w_100, CS_data_labeled = Energy_E1_E2)

# check the scattering plot and MC plot
plt.figure(figsize=(12, 12), facecolor='w')
CF.Scatter2D_plot(x=Energy100_corrected[:,0].tolist(),y=Energy100_corrected[:,1].tolist(),x_lim_left=0,x_lim_right=107,\
                  y_lim_left=0,y_lim_right=107)

# check the histogram
Energy_sum = np.sum(Energy_E1_E2, axis=1).reshape(-1,1)
CF.Histogram_barplot(Hist=Energy_sum, Bins=80, x_lim_low=20, x_lim_high=140)
Energy_sum = np.sum(Energy100_corrected, axis=1).reshape(-1,1)
CF.Histogram_barplot(Hist=Energy_sum, Bins=80, x_lim_low=20, x_lim_high=140)

# check the MC of CS band
###############################################################################################################




################################################################################################################
################################### Load the 117.3-keV charge sharing events ####################################

Energy_tmp = Energy[1:5700000,:]
Energy_sum_tmp = np.sum( Energy_tmp, axis=1 ).reshape(-1,1)
Energy_E1_E2 = Energy_tmp[ np.intersect1d(np.where(Energy_sum_tmp >= 100)[0],np.where(Energy_sum_tmp <= 130)[0]) ]
Energy_sum_tmp = np.sum( Energy_E1_E2, axis=1 ).reshape(-1,1)

##### plot the histogram of sum_Energy within the selected ROI
CF.Histogram_lineplot(Hist=Energy_sum_tmp, Bins=300, x_lim_high=200, x_lim_low=20, color='red')

##### plot the raw scatter figures within the selected ROI
CF.Scatter2D_plot(x=Energy_E1_E2[:,0],y=Energy_E1_E2[:,1],x_lim_left=0,x_lim_right=135,y_lim_left=0,y_lim_right=135)
########################################################################################################


####################### DBSCAN Clustering and Plot the results #########################################
##### Model Fitting I       
# Set Fitting Parameters
eps, min_samples = (1, 113) # 122 keV, high density CS events
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
for k in y_unique1:
    cur = (y_hat1 == k)
    if k == -1:
        CF.Scatter2D_plot(x=Energy_E1_E2[cur,0], y=Energy_E1_E2[cur,1], x_lim_left=0,x_lim_right=135,\
                          y_lim_left=0,y_lim_right=135, color = 'k')
        continue
    CF.Scatter2D_plot(x=Energy_E1_E2[cur,0], y=Energy_E1_E2[cur,1],x_lim_left=0,x_lim_right=135,\
                      y_lim_left=0,y_lim_right=135, color = 'r')
    CF.Scatter2D_plot(x=Energy_E1_E2[cur & core_indices1,0], y=Energy_E1_E2[cur & core_indices1,1],x_lim_left=0,x_lim_right=135,\
                      y_lim_left=0,y_lim_right=135, color = 'r')

###### check each DBSCAN clustering results
fig = plt.figure(figsize=(12, 12), facecolor='w')
for k in y_unique1:
    cur = (y_hat1 == k)
    if k == 0:
        CF.Scatter2D_plot(x=Energy_E1_E2[cur,0], y=Energy_E1_E2[cur,1], x_lim_left=0,x_lim_right=70,\
                          y_lim_left=0,y_lim_right=70, color = 'k')
        continue
########################################################################################################


####################### Extract the cluster in the ROI #################################################
##### Reorganize the clustered scattering points
y_hat = np.array([-1]*len(y_hat1))
y_hat[np.where( y_hat1 != -1 ),] = 0
y_hat[reduce( np.intersect1d, (np.where(y_hat == 0),np.where(Energy_E1_E2[:,0] >= 20),\
                               np.where(Energy_E1_E2[:,0] <= 110), np.where(Energy_sum_tmp >= 120)) ), ] = -1

y_unique = np.unique(y_hat)
cluster_lab_0 = Energy_E1_E2[np.where( ( y_hat == 0 ) )]

###### check each DBSCAN clustering results
plt.figure(figsize=(12, 12), facecolor='w')
plt.grid('on')
for k in y_unique:
    cur = (y_hat == k)
    if k == -1:
        plt.scatter(Energy_E1_E2[cur, 0], Energy_E1_E2[cur, 1], s=2, c='k') # non-clustered points
        continue
    if k == 0:
        plt.scatter(Energy_E1_E2[cur, 0], Energy_E1_E2[cur, 1], s=7, c='b') # non-clustered points
        continue
#    plt.scatter(Energy_sub[cur, 0], Energy_sub[cur, 1], s=5, c=clr, edgecolors='k')
#    plt.scatter(Energy_sub[cur & core_indices2, 0], Energy_sub[cur & core_indices2, 1], s=10, c=clr, \
#                            marker='o', edgecolors='k')
plt.xlabel('Energy (keV)',fontsize=20)
plt.ylabel('Energy (keV)',fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlim(0, 135)
plt.ylim(0, 135)
plt.grid(True)
######################################################################################################## 
        

####################### "Rot -> MC Shifting -> Rot" CSC function #######################################
##### Initialize the CSC object, charge sharing band correction
seg_size = 2
CSC_117_2pix = CSC( CS_dim=CS_dim, basis_old=basis_old, basis_new=basis_new, peak_energy=117.3, max_energy_range=600, seg_size=seg_size )
wet_x_117, wet_w_117, shift_w_117, seg_unit_117 = CSC_117_2pix.Pix2_Measurement( CS_data_labeled = cluster_lab_0 )
Energy117_corrected = CSC_117_2pix.Pix2_Correction(seg_unit=seg_unit_117, shift_w=shift_w_117, CS_data_labeled = Energy_E1_E2)

# check the scattering plot and MC plot
plt.figure(figsize=(12, 12), facecolor='w')
CF.Scatter2D_plot(x=Energy117_corrected[:,0].tolist(),y=Energy117_corrected[:,1].tolist(),x_lim_left=0,x_lim_right=135,\
                  y_lim_left=0,y_lim_right=135)

# check the histogram
Energy_sum = np.sum(Energy_E1_E2, axis=1).reshape(-1,1)
CF.Histogram_barplot(Hist=Energy_sum, Bins=80, x_lim_low=20, x_lim_high=140)
Energy_sum = np.sum(Energy117_corrected, axis=1).reshape(-1,1)
CF.Histogram_barplot(Hist=Energy_sum, Bins=80, x_lim_low=20, x_lim_high=140)

# check the MC of CS band
###############################################################################################################





################################################################################################################
################################### Load the 150-keV charge sharing events ####################################

Energy_tmp = Energy[1:12700000,:]
Energy_sum_tmp = np.sum( Energy_tmp, axis=1 ).reshape(-1,1)
Energy_E1_E2 = Energy_tmp[ np.intersect1d(np.where(Energy_sum_tmp >= 139)[0],np.where(Energy_sum_tmp <= 160)[0]) ]
Energy_sum_tmp = np.sum( Energy_E1_E2, axis=1 ).reshape(-1,1)

##### plot the histogram of sum_Energy within the selected ROI
CF.Histogram_lineplot(Hist=Energy_sum_tmp, Bins=300, x_lim_high=200, x_lim_low=20, color='red')

##### plot the raw scatter figures within the selected ROI
CF.Scatter2D_plot(x=Energy_E1_E2[:,0],y=Energy_E1_E2[:,1],x_lim_left=0,x_lim_right=160,y_lim_left=0,y_lim_right=160)
########################################################################################################


####################### DBSCAN Clustering and Plot the results #########################################
##### Model Fitting I       
# Set Fitting Parameters
eps, min_samples = (1, 43) # 122 keV, high density CS events
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
for k in y_unique1:
    cur = (y_hat1 == k)
    if k == -1:
        CF.Scatter2D_plot(x=Energy_E1_E2[cur,0], y=Energy_E1_E2[cur,1], x_lim_left=0,x_lim_right=135,\
                          y_lim_left=0,y_lim_right=135, color = 'k')
        continue
    CF.Scatter2D_plot(x=Energy_E1_E2[cur,0], y=Energy_E1_E2[cur,1],x_lim_left=0,x_lim_right=135,\
                      y_lim_left=0,y_lim_right=135, color = 'r')
    CF.Scatter2D_plot(x=Energy_E1_E2[cur & core_indices1,0], y=Energy_E1_E2[cur & core_indices1,1],x_lim_left=0,x_lim_right=160,\
                      y_lim_left=0,y_lim_right=160, color = 'r')

###### check each DBSCAN clustering results
fig = plt.figure(figsize=(12, 12), facecolor='w')
for k in y_unique1:
    cur = (y_hat1 == k)
    if k == 0:
        CF.Scatter2D_plot(x=Energy_E1_E2[cur,0], y=Energy_E1_E2[cur,1], x_lim_left=0,x_lim_right=70,\
                          y_lim_left=0,y_lim_right=70, color = 'k')
        continue
########################################################################################################


####################### Extract the cluster in the ROI #################################################
##### Reorganize the clustered scattering points
y_hat = np.array([-1]*len(y_hat1))
y_hat[np.where( y_hat1 != -1 ),] = 0

y_unique = np.unique(y_hat)
cluster_lab_0 = Energy_E1_E2[np.where( ( y_hat == 0 ) )]

###### check each DBSCAN clustering results
plt.figure(figsize=(12, 12), facecolor='w')
plt.grid('on')
for k in y_unique:
    cur = (y_hat == k)
    if k == -1:
        plt.scatter(Energy_E1_E2[cur, 0], Energy_E1_E2[cur, 1], s=2, c='k') # non-clustered points
        continue
    if k == 0:
        plt.scatter(Energy_E1_E2[cur, 0], Energy_E1_E2[cur, 1], s=7, c='b') # non-clustered points
        continue
#    plt.scatter(Energy_sub[cur, 0], Energy_sub[cur, 1], s=5, c=clr, edgecolors='k')
#    plt.scatter(Energy_sub[cur & core_indices2, 0], Energy_sub[cur & core_indices2, 1], s=10, c=clr, \
#                            marker='o', edgecolors='k')
plt.xlabel('Energy (keV)',fontsize=20)
plt.ylabel('Energy (keV)',fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlim(0, 160)
plt.ylim(0, 160)
plt.grid(True)
######################################################################################################## 
        

####################### "Rot -> MC Shifting -> Rot" CSC function #######################################
##### Initialize the CSC object, charge sharing band correction
seg_size = 2
CSC_150_2pix = CSC( CS_dim=CS_dim, basis_old=basis_old, basis_new=basis_new, peak_energy=150, max_energy_range=600, seg_size=seg_size )
wet_x_150, wet_w_150, shift_w_150, seg_unit_150 = CSC_150_2pix.Pix2_Measurement( CS_data_labeled = cluster_lab_0 )
Energy150_corrected = CSC_150_2pix.Pix2_Correction(seg_unit=seg_unit_150, shift_w=shift_w_150, CS_data_labeled = Energy_E1_E2)

# check the scattering plot and MC plot
plt.figure(figsize=(12, 12), facecolor='w')
CF.Scatter2D_plot(x=Energy117_corrected[:,0].tolist(),y=Energy117_corrected[:,1].tolist(),x_lim_left=0,x_lim_right=135,\
                  y_lim_left=0,y_lim_right=160)

# check the histogram
Energy_sum = np.sum(Energy_E1_E2, axis=1).reshape(-1,1)
CF.Histogram_barplot(Hist=Energy_sum, Bins=120, x_lim_low=70, x_lim_high=200)
Energy_sum = np.sum(Energy150_corrected, axis=1).reshape(-1,1)
CF.Histogram_barplot(Hist=Energy_sum, Bins=120, x_lim_low=70, x_lim_high=200)

# check the MC of CS band
###############################################################################################################





################################################################################################################
################################### Load the 225-keV charge sharing events ####################################

Energy_tmp = Energy[1:12700000,:]
Energy_sum_tmp = np.sum( Energy_tmp, axis=1 ).reshape(-1,1)
Energy_E1_E2 = Energy_tmp[ np.intersect1d(np.where(Energy_sum_tmp >= 195)[0],np.where(Energy_sum_tmp <= 230)[0]) ]
Energy_sum_tmp = np.sum( Energy_E1_E2, axis=1 ).reshape(-1,1)

##### plot the histogram of sum_Energy within the selected ROI
CF.Histogram_lineplot(Hist=Energy_sum_tmp, Bins=300, x_lim_high=250, x_lim_low=20, color='red')

##### plot the raw scatter figures within the selected ROI
CF.Scatter2D_plot(x=Energy_E1_E2[:,0],y=Energy_E1_E2[:,1],x_lim_left=0,x_lim_right=230,y_lim_left=0,y_lim_right=230)
########################################################################################################


####################### DBSCAN Clustering and Plot the results #########################################
##### Model Fitting I       
# Set Fitting Parameters
eps, min_samples = (1, 16) # 122 keV, high density CS events
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
for k in y_unique1:
    cur = (y_hat1 == k)
    if k == -1:
        CF.Scatter2D_plot(x=Energy_E1_E2[cur,0], y=Energy_E1_E2[cur,1], x_lim_left=0,x_lim_right=230,\
                          y_lim_left=0,y_lim_right=230, color = 'k')
        continue
    CF.Scatter2D_plot(x=Energy_E1_E2[cur,0], y=Energy_E1_E2[cur,1],x_lim_left=0,x_lim_right=230,\
                      y_lim_left=0,y_lim_right=230, color = 'r')
    CF.Scatter2D_plot(x=Energy_E1_E2[cur & core_indices1,0], y=Energy_E1_E2[cur & core_indices1,1],x_lim_left=0,x_lim_right=230,\
                      y_lim_left=0,y_lim_right=230, color = 'r')

###### check each DBSCAN clustering results
fig = plt.figure(figsize=(12, 12), facecolor='w')
for k in y_unique1:
    cur = (y_hat1 == k)
    if k == 0:
        CF.Scatter2D_plot(x=Energy_E1_E2[cur,0], y=Energy_E1_E2[cur,1], x_lim_left=0,x_lim_right=70,\
                          y_lim_left=0,y_lim_right=70, color = 'k')
        continue
########################################################################################################


####################### Extract the cluster in the ROI #################################################
##### Reorganize the clustered scattering points
y_hat = np.array([-1]*len(y_hat1))
y_hat[np.where( y_hat1 != -1 ),] = 0
y_hat[reduce( np.intersect1d, (np.where(y_hat == 0),np.where(Energy_E1_E2[:,0] <= 50),np.where(Energy_sum_tmp <= 200)) ), ] = -1
y_hat[reduce( np.intersect1d, (np.where(y_hat == 0),np.where(Energy_E1_E2[:,0] >= 150), np.where(Energy_sum_tmp <= 200)) ), ] = -1
    
y_unique = np.unique(y_hat)
cluster_lab_0 = Energy_E1_E2[np.where( ( y_hat == 0 ) )]

###### check each DBSCAN clustering results
plt.figure(figsize=(12, 12), facecolor='w')
plt.grid('on')
for k in y_unique:
    cur = (y_hat == k)
    if k == -1:
        plt.scatter(Energy_E1_E2[cur, 0], Energy_E1_E2[cur, 1], s=2, c='k') # non-clustered points
        continue
    if k == 0:
        plt.scatter(Energy_E1_E2[cur, 0], Energy_E1_E2[cur, 1], s=7, c='b') # non-clustered points
        continue
#    plt.scatter(Energy_sub[cur, 0], Energy_sub[cur, 1], s=5, c=clr, edgecolors='k')
#    plt.scatter(Energy_sub[cur & core_indices2, 0], Energy_sub[cur & core_indices2, 1], s=10, c=clr, \
#                            marker='o', edgecolors='k')
plt.xlabel('Energy (keV)',fontsize=20)
plt.ylabel('Energy (keV)',fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlim(0, 250)
plt.ylim(0, 250)
plt.grid(True)
######################################################################################################## 
        

####################### "Rot -> MC Shifting -> Rot" CSC function #######################################
##### Initialize the CSC object, charge sharing band correction
seg_size = 2
CSC_225_2pix = CSC( CS_dim=CS_dim, basis_old=basis_old, basis_new=basis_new, peak_energy=218, max_energy_range=600, seg_size=seg_size )
wet_x_225, wet_w_225, shift_w_225, seg_unit_225 = CSC_225_2pix.Pix2_Measurement( CS_data_labeled = cluster_lab_0 )
Energy225_corrected = CSC_225_2pix.Pix2_Correction(seg_unit=seg_unit_225, shift_w=shift_w_225, CS_data_labeled = Energy_E1_E2)

# check the scattering plot and MC plot
plt.figure(figsize=(12, 12), facecolor='w')
CF.Scatter2D_plot(x=Energy225_corrected[:,0].tolist(),y=Energy225_corrected[:,1].tolist(),x_lim_left=0,x_lim_right=260,\
                  y_lim_left=0,y_lim_right=260)

# check the histogram
Energy_sum = np.sum(Energy_E1_E2, axis=1).reshape(-1,1)
CF.Histogram_barplot(Hist=Energy_sum, Bins=120, x_lim_low=200, x_lim_high=280)
Energy_sum = np.sum(Energy225_corrected, axis=1).reshape(-1,1)
CF.Histogram_barplot(Hist=Energy_sum, Bins=120, x_lim_low=200, x_lim_high=280)

# check the MC of CS band
###############################################################################################################




################################################################################################################
################################### Load the 440-keV charge sharing events ####################################

Energy_tmp = Energy[1:12700000,:]
Energy_sum_tmp = np.sum( Energy_tmp, axis=1 ).reshape(-1,1)
Energy_E1_E2 = Energy_tmp[ np.intersect1d(np.where(Energy_sum_tmp >= 400)[0],np.where(Energy_sum_tmp <= 460)[0]) ]
Energy_sum_tmp = np.sum( Energy_E1_E2, axis=1 ).reshape(-1,1)

##### plot the histogram of sum_Energy within the selected ROI
CF.Histogram_lineplot(Hist=Energy_sum_tmp, Bins=1000, x_lim_high=500, x_lim_low=20, color='red')

##### plot the raw scatter figures within the selected ROI
CF.Scatter2D_plot(x=Energy_E1_E2[:,0],y=Energy_E1_E2[:,1],x_lim_left=0,x_lim_right=500,y_lim_left=0,y_lim_right=500)
########################################################################################################


####################### DBSCAN Clustering and Plot the results #########################################
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
for k in y_unique1:
    cur = (y_hat1 == k)
    if k == -1:
        CF.Scatter2D_plot(x=Energy_E1_E2[cur,0], y=Energy_E1_E2[cur,1], x_lim_left=0,x_lim_right=500,\
                          y_lim_left=0,y_lim_right=500, color = 'k')
        continue
    CF.Scatter2D_plot(x=Energy_E1_E2[cur,0], y=Energy_E1_E2[cur,1],x_lim_left=0,x_lim_right=500,\
                      y_lim_left=0,y_lim_right=500, color = 'r')
    CF.Scatter2D_plot(x=Energy_E1_E2[cur & core_indices1,0], y=Energy_E1_E2[cur & core_indices1,1],x_lim_left=0,x_lim_right=500,\
                      y_lim_left=0,y_lim_right=500, color = 'r')

###### check each DBSCAN clustering results
fig = plt.figure(figsize=(12, 12), facecolor='w')
for k in y_unique1:
    cur = (y_hat1 == k)
    if k == 0:
        CF.Scatter2D_plot(x=Energy_E1_E2[cur,0], y=Energy_E1_E2[cur,1], x_lim_left=0,x_lim_right=70,\
                          y_lim_left=0,y_lim_right=70, color = 'k')
        continue
########################################################################################################


####################### Extract the cluster in the ROI #################################################
##### Reorganize the clustered scattering points
y_hat = np.array([-1]*len(y_hat1))
y_hat[np.where( y_hat1 != -1 ),] = 0
y_hat[reduce( np.intersect1d, (np.where(y_hat == 0),np.where(Energy_sum_tmp <= 430)) ), ] = -1
    
y_unique = np.unique(y_hat)
cluster_lab_0 = Energy_E1_E2[np.where( ( y_hat == 0 ) )]

###### check each DBSCAN clustering results
plt.figure(figsize=(12, 12), facecolor='w')
plt.grid('on')
for k in y_unique:
    cur = (y_hat == k)
    if k == -1:
        plt.scatter(Energy_E1_E2[cur, 0], Energy_E1_E2[cur, 1], s=2, c='k') # non-clustered points
        continue
    if k == 0:
        plt.scatter(Energy_E1_E2[cur, 0], Energy_E1_E2[cur, 1], s=7, c='b') # non-clustered points
        continue
#    plt.scatter(Energy_sub[cur, 0], Energy_sub[cur, 1], s=5, c=clr, edgecolors='k')
#    plt.scatter(Energy_sub[cur & core_indices2, 0], Energy_sub[cur & core_indices2, 1], s=10, c=clr, \
#                            marker='o', edgecolors='k')
plt.xlabel('Energy (keV)',fontsize=20)
plt.ylabel('Energy (keV)',fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlim(0, 500)
plt.ylim(0, 500)
plt.grid(True)
######################################################################################################## 
        

####################### "Rot -> MC Shifting -> Rot" CSC function #######################################
##### Initialize the CSC object, charge sharing band correction
seg_size = 2
CSC_440_2pix = CSC( CS_dim=CS_dim, basis_old=basis_old, basis_new=basis_new, peak_energy=440, max_energy_range=600, seg_size=seg_size )
wet_x_440, wet_w_440, shift_w_440, seg_unit_440 = CSC_440_2pix.Pix2_Measurement( CS_data_labeled = cluster_lab_0 )
Energy440_corrected = CSC_440_2pix.Pix2_Correction(seg_unit=seg_unit_440, shift_w=shift_w_440, CS_data_labeled = Energy_E1_E2)

# check the scattering plot and MC plot
plt.figure(figsize=(12, 12), facecolor='w')
CF.Scatter2D_plot(x=Energy440_corrected[:,0].tolist(),y=Energy440_corrected[:,1].tolist(),x_lim_left=0,x_lim_right=500,\
                  y_lim_left=0,y_lim_right=500)

# check the histogram
Energy_sum = np.sum(Energy_E1_E2, axis=1).reshape(-1,1)
CF.Histogram_barplot(Hist=Energy_sum, Bins=120, x_lim_low=400, x_lim_high=480)
Energy_sum = np.sum(Energy440_corrected, axis=1).reshape(-1,1)
CF.Histogram_barplot(Hist=Energy_sum, Bins=120, x_lim_low=400, x_lim_high=480)

# check the MC of CS band
###############################################################################################################




################################################################################################################
################################### Load the 465-keV charge sharing events ####################################

Energy_tmp = Energy[1:12700000,:]
Energy_sum_tmp = np.sum( Energy_tmp, axis=1 ).reshape(-1,1)
Energy_E1_E2 = Energy_tmp[ np.intersect1d(np.where(Energy_sum_tmp >= 460)[0],np.where(Energy_sum_tmp <= 500)[0]) ]
Energy_sum_tmp = np.sum( Energy_E1_E2, axis=1 ).reshape(-1,1)

##### plot the histogram of sum_Energy within the selected ROI
CF.Histogram_lineplot(Hist=Energy_sum_tmp, Bins=1000, x_lim_high=500, x_lim_low=20, color='red')

##### plot the raw scatter figures within the selected ROI
CF.Scatter2D_plot(x=Energy_E1_E2[:,0],y=Energy_E1_E2[:,1],x_lim_left=0,x_lim_right=500,y_lim_left=0,y_lim_right=500)
########################################################################################################


####################### DBSCAN Clustering and Plot the results #########################################
##### Model Fitting I       
# Set Fitting Parameters
eps, min_samples = (2, 2) # 122 keV, high density CS events
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
for k in y_unique1:
    cur = (y_hat1 == k)
    if k == -1:
        CF.Scatter2D_plot(x=Energy_E1_E2[cur,0], y=Energy_E1_E2[cur,1], x_lim_left=0,x_lim_right=500,\
                          y_lim_left=0,y_lim_right=500, color = 'k')
        continue
    CF.Scatter2D_plot(x=Energy_E1_E2[cur,0], y=Energy_E1_E2[cur,1],x_lim_left=0,x_lim_right=500,\
                      y_lim_left=0,y_lim_right=500, color = 'r')
    CF.Scatter2D_plot(x=Energy_E1_E2[cur & core_indices1,0], y=Energy_E1_E2[cur & core_indices1,1],x_lim_left=0,x_lim_right=500,\
                      y_lim_left=0,y_lim_right=500, color = 'r')

###### check each DBSCAN clustering results
fig = plt.figure(figsize=(12, 12), facecolor='w')
for k in y_unique1:
    cur = (y_hat1 == k)
    if k == 0:
        CF.Scatter2D_plot(x=Energy_E1_E2[cur,0], y=Energy_E1_E2[cur,1], x_lim_left=0,x_lim_right=70,\
                          y_lim_left=0,y_lim_right=70, color = 'k')
        continue
########################################################################################################


####################### Extract the cluster in the ROI #################################################
##### Reorganize the clustered scattering points
y_hat = np.array([-1]*len(y_hat1))
y_hat[np.where( y_hat1 != -1 ),] = 0
    
y_unique = np.unique(y_hat)
cluster_lab_0 = Energy_E1_E2[np.where( ( y_hat == 0 ) )]

###### check each DBSCAN clustering results
plt.figure(figsize=(12, 12), facecolor='w')
plt.grid('on')
for k in y_unique:
    cur = (y_hat == k)
    if k == -1:
        plt.scatter(Energy_E1_E2[cur, 0], Energy_E1_E2[cur, 1], s=2, c='k') # non-clustered points
        continue
    if k == 0:
        plt.scatter(Energy_E1_E2[cur, 0], Energy_E1_E2[cur, 1], s=7, c='b') # non-clustered points
        continue
#    plt.scatter(Energy_sub[cur, 0], Energy_sub[cur, 1], s=5, c=clr, edgecolors='k')
#    plt.scatter(Energy_sub[cur & core_indices2, 0], Energy_sub[cur & core_indices2, 1], s=10, c=clr, \
#                            marker='o', edgecolors='k')
plt.xlabel('Energy (keV)',fontsize=20)
plt.ylabel('Energy (keV)',fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlim(0, 500)
plt.ylim(0, 500)
plt.grid(True)
######################################################################################################## 
        

####################### "Rot -> MC Shifting -> Rot" CSC function #######################################
##### Initialize the CSC object, charge sharing band correction
seg_size = 2
CSC_465_2pix = CSC( CS_dim=CS_dim, basis_old=basis_old, basis_new=basis_new, peak_energy=465, max_energy_range=600, seg_size=seg_size )
wet_x_465, wet_w_465, shift_w_465, seg_unit_465 = CSC_465_2pix.Pix2_Measurement( CS_data_labeled = cluster_lab_0 )
Energy465_corrected = CSC_465_2pix.Pix2_Correction(seg_unit=seg_unit_465, shift_w=shift_w_465, CS_data_labeled = Energy_E1_E2)

# check the scattering plot and MC plot
plt.figure(figsize=(12, 12), facecolor='w')
CF.Scatter2D_plot(x=Energy465_corrected[:,0].tolist(),y=Energy465_corrected[:,1].tolist(),x_lim_left=0,x_lim_right=500,\
                  y_lim_left=0,y_lim_right=500)

# check the histogram
Energy_sum = np.sum(Energy_E1_E2, axis=1).reshape(-1,1)
CF.Histogram_barplot(Hist=Energy_sum, Bins=120, x_lim_low=420, x_lim_high=490)
Energy_sum = np.sum(Energy465_corrected, axis=1).reshape(-1,1)
CF.Histogram_barplot(Hist=Energy_sum, Bins=120, x_lim_low=420, x_lim_high=490)

# check the MC of CS band
###############################################################################################################






###############################################################################################################
##### Full range correction ###################################################################################\
Energy_sum = np.sum(Energy, axis=1).reshape(-1,1)

Energy_E1_E2 = Energy[ np.intersect1d(np.where(Energy_sum >= 20)[0],np.where(Energy_sum <= 75)[0]) ]
Energy61_corrected = CSC_61_2pix.Pix2_Correction(seg_unit=seg_unit_61, shift_w=shift_w_61, CS_data_labeled = Energy_E1_E2)

Energy_E1_E2 = Energy[ np.intersect1d(np.where(Energy_sum > 75)[0],np.where(Energy_sum <= 95)[0]) ]
Energy79_corrected = CSC_79_2pix.Pix2_Correction(seg_unit=seg_unit_79, shift_w=shift_w_79, CS_data_labeled = Energy_E1_E2)

Energy_E1_E2 = Energy[ np.intersect1d(np.where(Energy_sum > 95)[0],np.where(Energy_sum <= 110)[0]) ]
Energy100_corrected = CSC_100_2pix.Pix2_Correction(seg_unit=seg_unit_100, shift_w=shift_w_100, CS_data_labeled = Energy_E1_E2)

Energy_E1_E2 = Energy[ np.intersect1d(np.where(Energy_sum > 110)[0],np.where(Energy_sum <= 130)[0]) ]
Energy117_corrected = CSC_117_2pix.Pix2_Correction(seg_unit=seg_unit_117, shift_w=shift_w_117, CS_data_labeled = Energy_E1_E2)

Energy_E1_E2 = Energy[ np.intersect1d(np.where(Energy_sum > 130)[0],np.where(Energy_sum <= 200)[0]) ]
Energy150_corrected = CSC_150_2pix.Pix2_Correction(seg_unit=seg_unit_150, shift_w=shift_w_150, CS_data_labeled = Energy_E1_E2)

Energy_E1_E2 = Energy[ np.intersect1d(np.where(Energy_sum > 200)[0],np.where(Energy_sum <= 250)[0]) ]
Energy225_corrected = CSC_225_2pix.Pix2_Correction(seg_unit=seg_unit_225, shift_w=shift_w_225, CS_data_labeled = Energy_E1_E2)

Energy_E1_E2 = Energy[ np.intersect1d(np.where(Energy_sum > 250)[0],np.where(Energy_sum <= 460)[0]) ]
Energy440_corrected = CSC_440_2pix.Pix2_Correction(seg_unit=seg_unit_440, shift_w=shift_w_440, CS_data_labeled = Energy_E1_E2)

Energy_E1_E2 = Energy[ np.intersect1d(np.where(Energy_sum >= 460)[0],np.where(Energy_sum <= 500)[0]) ]
Energy465_corrected = CSC_465_2pix.Pix2_Correction(seg_unit=seg_unit_465, shift_w=shift_w_465, CS_data_labeled = Energy_E1_E2)

Energy_corrected = np.vstack((Energy61_corrected, Energy79_corrected, Energy100_corrected, Energy117_corrected,\
                              Energy150_corrected, Energy225_corrected, Energy440_corrected, Energy465_corrected))
Energy_sum_corrected = np.sum(Energy_corrected, axis=1).reshape(-1,1)


### Save the corrected files
var = Energy_corrected
dataframe = pd.DataFrame( var )
dataframe.columns = ['E1','E2']
dataframe.to_csv('C:\Jiajin\ChaSha_2017\CSC_Data\JHU_data\Ac225_Pix2_Sharing_Corrected.csv', index=False, sep=',')


##### plot spectrum histogram
## raw
x_lim_low = 20
x_lim_high = 600
Bins = 1000
Hist2 = Energy_sum
color = 'blue'

bins = np.linspace( start=x_lim_low, stop=x_lim_high, num=Bins )
hist2, bin_edges = np.histogram( Hist2, bins )

hist2[np.where(bin_edges[:-1] > 180)[0],] = hist2[np.where(bin_edges[:-1] > 180)[0],] * 5
hist2[np.where(bin_edges[:-1] > 400)[0],] = hist2[np.where(bin_edges[:-1] > 400)[0],] * 10

plt.plot( bin_edges[:-1], hist2, color=color, linewidth=3, alpha=0.7 )
plt.xlim(x_lim_low, x_lim_high)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Energy (keV)',fontsize=20)
plt.ylabel('Counts per bin size',fontsize=20)
plt.grid('on')
# plt.legend('before correction')


## correction
x_lim_low = 20
x_lim_high = 600
Bins = 1000
Hist1 = Energy_sum_corrected
color = 'red'

bins = np.linspace( start=x_lim_low, stop=x_lim_high, num=Bins )
hist1, bin_edges = np.histogram( Hist1, bins )

hist1[np.where(bin_edges[:-1] > 180)[0],] = hist1[np.where(bin_edges[:-1] > 180)[0],] * 5
hist1[np.where(bin_edges[:-1] > 400)[0],] = hist1[np.where(bin_edges[:-1] > 400)[0],] * 10

plt.plot( bin_edges[:-1], hist1, color=color, linewidth=3, alpha=0.7 )
plt.xlim(x_lim_low, x_lim_high)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Energy (keV)',fontsize=20)
plt.ylabel('Counts per bin size',fontsize=20)
plt.grid('on')

plt.legend('after correction')


hist2[np.where(bin_edges[:-1] > 180)[0],] = hist2[np.where(bin_edges[:-1] > 180)[0],] / 5
hist2[np.where(bin_edges[:-1] > 400)[0],] = hist2[np.where(bin_edges[:-1] > 400)[0],] / 10
hist1[np.where(bin_edges[:-1] > 180)[0],] = hist1[np.where(bin_edges[:-1] > 180)[0],] / 5
hist1[np.where(bin_edges[:-1] > 400)[0],] = hist1[np.where(bin_edges[:-1] > 400)[0],] / 10

bin_edges = bin_edges[:-1]
var = np.vstack((bin_edges, hist1, hist2))
dataframe = pd.DataFrame( var.T )
dataframe.columns = ['Energy bin','hist_noCorr','hist_Corr']
dataframe.to_csv('E:\JHU\det_2\JHU\Point_Source\Results\histogramCorr_bins.csv', index=False, sep=',')


