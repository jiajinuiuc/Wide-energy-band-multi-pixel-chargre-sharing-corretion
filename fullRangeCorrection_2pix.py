# -*- coding: utf-8 -*-
"""
Objective： 
Full range energy spectrum correction， try with 2-pixel charge sharing events
0. charge sharing events clustering on 60 keV(Am), 80.99 keV(Ba), 122 keV（Co）
1. full "spatial" range segmentation and calculate the projection distance in each channel
3. calculate the projection distance of each charge sharing band at each channel
4. linear interpolation of the porjection distance between each band at different channel
5. based on the linear interpolation results, do the full range charge sharing correction
Version 0
@author: J. J. Zhang
Last update: May, 2019
"""
import sys
sys.path.append('C:\Jiajin\Mfile\Training_Sample_Analysis')

from charge_sharing_correction import charge_sharing_correction as CSC # class file, charge sharing correction
from charge_sharing_correction import Common_used_function as CF # class file, common-used plotting tools
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

################################################################################################################
################################### Load the 60-keV charge sharing events ######################################
CS_data = pd.read_csv( 'C:\Jiajin\ChaSha_2017\CSC_Data\CSC_extracted_data\Am_2Pix.csv' )
Energy = CS_data.iloc[:, :].values   #取第一行
CS_dim = Energy.shape[1]

Energy_sum = np.sum( Energy, axis=1 ).reshape(-1,1)
Energy_E1_E2_sum = Energy[ np.intersect1d(np.where(Energy_sum >= 50)[0],np.where(Energy_sum <= 70)[0]) ]
Energy_E1_E2_else = np.delete( Energy, np.intersect1d(np.where(Energy_sum >= 50)[0],np.where(Energy_sum <= 70)[0]), axis =0 )

# Energy_sum = np.sum( Energy_E1_E2_sum, axis=1 ).reshape(-1,1)
# Energy_E1_E2_sum = np.hstack( (Energy_E1_E2_sum, Energy_sum) )

##### plot the histogram of sum_Energy within the selected ROI
CF.Histogram_lineplot(Hist=Energy_sum, Bins=300, x_lim_high=200, x_lim_low=0, color='red')

##### plot the raw scatter figures within the selected ROI
CF.Scatter2D_plot(x=Energy_E1_E2_sum[:,0],y=Energy_E1_E2_sum[:,1],x_lim_left=0, x_lim_right=65, y_lim_left=0, y_lim_right=65)
########################################################################################################


####################### DBSCAN Clustering and Plot the results #########################################
Energy_sub = np.vstack( ( Energy_E1_E2_sum[:,0], Energy_E1_E2_sum[:,1] ) ).T

##### Model Fitting I        
# Set Fitting Parameters
eps, min_samples = (1, 30) # 122 keV, total events

model1 = DBSCAN( eps=eps, min_samples=min_samples )
model1.fit( Energy_sub )
y_hat1 = model1.labels_

core_indices1 = np.zeros_like(y_hat1, dtype=bool) # create zero/boolean array with the same length
core_indices1[model1.core_sample_indices_] = True # 核样本的目录 < (label != 0)

y_unique1 = np.unique(y_hat1) # extract different Labels
n_clusters1 = y_unique1.size - (1 if -1 in y_hat1 else 0)
print(y_unique1, 'clustering number is :', n_clusters1)

# Plot the DBSCAN clustering results
clrs = plt.cm.Spectral(np.linspace(0, 0.8, 2*y_unique1.size))
#clrs = ['k', 'r', 'b', 'g', 'y']
plt.figure(figsize=(12, 12), facecolor='w')
plt.grid('on')
for k, clr in zip(y_unique1, clrs):
    cur = (y_hat1 == k)
    if k == -1:
        plt.scatter(Energy_sub[cur, 0], Energy_sub[cur, 1], s=5, c='k') # non-clustered points
        continue
    plt.scatter(Energy_sub[cur, 0], Energy_sub[cur, 1], s=5, c=clr, edgecolors='k')
    plt.scatter(Energy_sub[cur & core_indices1, 0], Energy_sub[cur & core_indices1, 1], s=10, c=clr, \
                            marker='o', edgecolors='k')
plt.xlabel('Energy (keV)',fontsize=20)
plt.ylabel('Energy (keV)',fontsize=20)
plt.xlim(0, 70)
plt.ylim(0, 70)
plt.grid(True)


##### Model Fitting II    
# Set Fitting Parameters
eps, min_samples = (1, 46) # 122 keV, total events

model2 = DBSCAN( eps=eps, min_samples=min_samples )
model2.fit( Energy_sub )
y_hat2 = model2.labels_

core_indices2 = np.zeros_like(y_hat2, dtype=bool) # create zero/boolean array with the same length
core_indices2[model2.core_sample_indices_] = True # 核样本的目录 < (label != 0)

y_unique2 = np.unique(y_hat2) # extract different Labels
n_clusters2 = y_unique2.size - (1 if -1 in y_hat2 else 0)
print(y_unique2, 'clustering number is :', n_clusters2)

# Plot the DBSCAN clustering results
clrs = plt.cm.Spectral(np.linspace(0, 0.8, 2*y_unique2.size))
plt.figure(figsize=(12, 12), facecolor='w')
plt.grid('on')
for k, clr in zip(y_unique2, clrs):
    cur = (y_hat2 == k)
    if k == -1:
        plt.scatter(Energy_sub[cur, 0], Energy_sub[cur, 1], s=5, c='k') # non-clustered points
        continue
    plt.scatter(Energy_sub[cur, 0], Energy_sub[cur, 1], s=5, c=clr, edgecolors='k')
    plt.scatter(Energy_sub[cur & core_indices2, 0], Energy_sub[cur & core_indices2, 1], s=10, c=clr, \
                            marker='o', edgecolors='k')
plt.xlabel('Energy (keV)',fontsize=20)
plt.ylabel('Energy (keV)',fontsize=20)
plt.xlim(0, 70)
plt.ylim(0, 70)
plt.grid(True)

###### check each DBSCAN clustering results
clrs = plt.cm.Spectral(np.linspace(0, 0.8, 2*y_unique1.size))
plt.figure(figsize=(12, 12), facecolor='w')
plt.grid('on')
for k, clr in zip(y_unique1, clrs):
    cur = (y_hat1 == k)
    if k == 5:
        plt.scatter(Energy_sub[cur, 0], Energy_sub[cur, 1], s=5, c='k') # non-clustered points
        continue
#    plt.scatter(Energy_sub[cur, 0], Energy_sub[cur, 1], s=5, c=clr, edgecolors='k')
#    plt.scatter(Energy_sub[cur & core_indices2, 0], Energy_sub[cur & core_indices2, 1], s=10, c=clr, \
#                            marker='o', edgecolors='k')
plt.xlabel('Energy (keV)',fontsize=20)
plt.ylabel('Energy (keV)',fontsize=20)
plt.xlim(0, 70)
plt.ylim(0, 70)
plt.grid(True)
########################################################################################################


####################### Extract the cluster in the ROI #################################################
##### Reorganize the clustered scattering points
Energy = np.vstack( (Energy_sub, Energy_E1_E2_else) )
Energy_E1_E2_sum = np.sum( Energy_sub, axis=1 ).reshape(-1,1)
y_hat = np.array([-1]*len(Energy))
y_hat[np.where( y_hat1 != -1 ),] = 0
y_hat[reduce( np.intersect1d, (np.where(y_hat2 == 2), np.where(Energy_E1_E2_sum >= 56)) ), ] = 1
y_hat[reduce( np.intersect1d, (np.where(y_hat2 == 3), np.where(Energy_E1_E2_sum >= 56)) ), ] = 2
y_unique = np.unique(y_hat)

cluster_lab_0 = Energy[np.where( ( y_hat == 0 ) )]
cluster_lab_1 = Energy[np.where( ( y_hat == 1 ) )]
cluster_lab_2 = Energy[np.where( ( y_hat == 2 ) )]
cluster_lab_noise = Energy[np.where( ( y_hat == -1 ) )]

###### check each DBSCAN clustering results
plt.figure(figsize=(12, 12), facecolor='w')
plt.grid('on')
for k in y_unique:
    cur = (y_hat == k)
    if k == -1:
        plt.scatter(Energy[cur, 0], Energy[cur, 1], s=2, c='k') # non-clustered points
        continue
    if k == 0:
        plt.scatter(Energy[cur, 0], Energy[cur, 1], s=7, c='b') # non-clustered points
        continue
    if k == 1:
        plt.scatter(Energy[cur, 0], Energy[cur, 1], s=7, c='r') # non-clustered points
        continue
    if k == 2:
        plt.scatter(Energy[cur, 0], Energy[cur, 1], s=7, c='y') # non-clustered points
        continue
plt.xlabel('Energy (keV)',fontsize=20)
plt.ylabel('Energy (keV)',fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlim(0, 70)
plt.ylim(0, 70)
plt.grid(True)

###### Save data Figure 5 122-keV, 2-pixel, scattering + DBSCAN scattering
data_save = np.hstack( (Energy, y_hat.reshape(-1,1)) ) ### ( E1, E2, label)
CF.SaveFiles(var=data_save, var_name=['E1', 'E2', 'label'], \
              location="C:\Jiajin\ChaSha_2017\CSC_Data\CSC_extracted_data\Figure_data\Figure_5.csv")
del(data_save)
######################################################################################################## 
        

####################### "Rot -> MC Shifting -> Rot" CSC function #######################################
basis_old =np.mat( [ [1,0],
                     [0,1] ] ) #[x, y]
basis_new = np.mat( [ [  1/np.sqrt(2),  1/np.sqrt(2) ],
                      [ -1/np.sqrt(2),  1/np.sqrt(2) ] ] )
seg_size = 2
##### Initialize the CSC object, charge sharing band correction
CSC_60_2pix = CSC( CS_dim=CS_dim, basis_old=basis_old, basis_new=basis_new, peak_energy=60, max_energy_range=140, seg_size=seg_size )
wet_x_60, wet_y_60, wet_w_60, shift_w_60, seg_unit_60 = CSC_60_2pix.Pix2_Measurement( CS_data_labeled = cluster_lab_0 )
Energy60_corrected = CSC_60_2pix.Pix2_Correction(seg_unit=seg_unit_60, shift_w=shift_w_60, CS_data_labeled=Energy)
# check the scattering plot and MC plot
CF.Scatter2D_plot(x=cluster_lab_0_corrected[:,0].tolist(), y=cluster_lab_0_corrected[:,1].tolist(), x_lim=70, y_lim=70 )
CF.Line_plot(x=wet_x, y=wet_w_60, color='blue', x_label='Spatial-axis', y_label='Energy-axis')
CF.Line_plot(x=wet_x, y=shift_w_60, color='red', x_label='Spatial-axis', y_label='Energy-axis')

##### fluorescence events correction
wet_fluo, wet_fluo, wet_fluo, shift_fluo, seg_unit_60 = CSC_60_2pix.Pix2_Measurement( CS_data_labeled = cluster_lab_1 )
cluster_lab_1_corrected = CSC_60_2pix.Pix2_Correction(seg_unit=seg_unit_60, shift_w=shift_fluo, CS_data_labeled=Energy)
wet_fluo, wet_fluo, wet_fluo, shift_fluo, seg_unit_60 = CSC_60_2pix.Pix2_Measurement( CS_data_labeled = cluster_lab_1 )
cluster_lab_2_corrected, wet_x, wet_w, shift_w, seg_unit = CSC_60_2pix.Pix2_Correction( CS_data_labeled = cluster_lab_2 )
##### plot the local energy spectrum before and after CS correction
Energy_sub_corrected = np.vstack( (cluster_lab_noise, cluster_lab_0_corrected, cluster_lab_1_corrected, cluster_lab_2_corrected) )
CSC.Scatter2D_plot(x=Energy_sub_corrected[:,0].tolist(), y=Energy_sub_corrected[:,1].tolist(),  x_lim_left=0, x_lim_right=70, \
                   y_lim_left=0, y_lim_right=70 )

CSC.Histogram_barplot(Hist=Energy_sum, Bins=100, x_lim_low=20, x_lim_high=140)
CSC.Histogram_barplot(Hist=np.sum( Energy_sub_corrected, axis=1 ).reshape(-1,1), Bins=100, x_lim_low=20, x_lim_high=140)
################################################################################################################





################################################################################################################
################################### Load the 81-keV charge sharing events ######################################
CS_data = pd.read_csv( 'C:\Jiajin\Mfile\Training_Sample_Analysis\Pix2_80Sharing.csv' )
Energy = CS_data.iloc[:, :].values   #取第一行
CS_dim = Energy.shape[1]

Energy_sum = np.sum( Energy, axis=1 ).reshape(-1,1)
Energy_E1_E2_sum = Energy[ np.intersect1d(np.where(Energy_sum >= 70)[0],np.where(Energy_sum <= 90)[0]) ]
Energy_E1_E2_else = np.delete( Energy, np.intersect1d(np.where(Energy_sum >= 70)[0],np.where(Energy_sum <= 90)[0]), axis =0 )

Energy_sum = np.sum( Energy_E1_E2_sum, axis=1 ).reshape(-1,1)
Energy_E1_E2_sum = np.hstack( (Energy_E1_E2_sum, Energy_sum) )

##### plot the histogram of sum_Energy within the selected ROI
CSC.Histogram_barplot(Hist=Energy_sum,Bins=100,x_lim_low=20,x_lim_high=150)

##### plot the raw scatter figures within the selected ROI
CSC.Scatter2D_plot(x=Energy_E1_E2_sum[:,0],y=Energy_E1_E2_sum[:,1],x_lim_left=0, x_lim_right=100, \
                   y_lim_left=0, y_lim_right=100)
########################################################################################################


####################### DBSCAN Clustering and Plot the results #########################################
Energy_sub = np.vstack( ( Energy_E1_E2_sum[:,0], Energy_E1_E2_sum[:,1] ) ).T

##### Model Fitting I        
# Set Fitting Parameters
eps, min_samples = (1, 150) # 122 keV, high density CS events
# eps, min_samples = (1, 100) # 122 keV, high density CS events

model1 = DBSCAN( eps=eps, min_samples=min_samples )
model1.fit( Energy_sub )
y_hat1 = model1.labels_

core_indices1 = np.zeros_like(y_hat1, dtype=bool) # create zero/boolean array with the same length
core_indices1[model1.core_sample_indices_] = True # 核样本的目录 < (label != 0)

y_unique1 = np.unique(y_hat1) # extract different Labels
n_clusters1 = y_unique1.size - (1 if -1 in y_hat1 else 0)
print(y_unique1, 'clustering number is :', n_clusters1)

# Plot the DBSCAN clustering results
clrs = plt.cm.Spectral(np.linspace(0, 0.8, 2*y_unique1.size))
#clrs = ['k', 'r', 'b', 'g', 'y']
plt.figure(figsize=(12, 12), facecolor='w')
plt.grid('on')
for k, clr in zip(y_unique1, clrs):
    cur = (y_hat1 == k)
    if k == -1:
        plt.scatter(Energy_sub[cur, 0], Energy_sub[cur, 1], s=5, c='k') # non-clustered points
        continue
    plt.scatter(Energy_sub[cur, 0], Energy_sub[cur, 1], s=5, c=clr, edgecolors='k')
    plt.scatter(Energy_sub[cur & core_indices1, 0], Energy_sub[cur & core_indices1, 1], s=10, c=clr, \
                            marker='o', edgecolors='k')
plt.xlabel('Energy (keV)',fontsize=20)
plt.ylabel('Energy (keV)',fontsize=20)
plt.xlim(0, 90)
plt.ylim(0, 90)
plt.grid(True)

###### check each DBSCAN clustering results
clrs = plt.cm.Spectral(np.linspace(0, 0.8, 2*y_unique1.size))
plt.figure(figsize=(12, 12), facecolor='w')
plt.grid('on')
for k, clr in zip(y_unique1, clrs):
    cur = (y_hat1 == k)
    if k == 5:
        plt.scatter(Energy_sub[cur, 0], Energy_sub[cur, 1], s=5, c='k') # non-clustered points
        continue
#    plt.scatter(Energy_sub[cur, 0], Energy_sub[cur, 1], s=5, c=clr, edgecolors='k')
#    plt.scatter(Energy_sub[cur & core_indices2, 0], Energy_sub[cur & core_indices2, 1], s=10, c=clr, \
#                            marker='o', edgecolors='k')
plt.xlabel('Energy (keV)',fontsize=20)
plt.ylabel('Energy (keV)',fontsize=20)
plt.xlim(0, 90)
plt.ylim(0, 90)
plt.grid(True)
########################################################################################################
##### Model Fitting II
# Set Fitting Parameters
eps, min_samples = (1, 200) # 122 keV, all possible CS events
model2 = DBSCAN( eps=eps, min_samples=min_samples )
model2.fit( Energy_sub )
y_hat2 = model2.labels_

core_indices2 = np.zeros_like(y_hat2, dtype=bool) # create zero/boolean array with the same length
core_indices2[model2.core_sample_indices_] = True # 核样本的目录 < (label != 0)
 
y_unique2 = np.unique(y_hat2) # extract different Labels
n_clusters2 = y_unique2.size - (1 if -1 in y_hat2 else 0)
print(y_unique2, 'clustering number is :', n_clusters2)

# Plot the DBSCAN clustering results
clrs = plt.cm.Spectral(np.linspace(0, 0.8, 2*y_unique2.size))
plt.figure(figsize=(12, 12), facecolor='w')
plt.grid('on')
for k, clr in zip(y_unique2, clrs):
    cur = (y_hat2 == k)
    if k == -1:
        plt.scatter(Energy_sub[cur, 0], Energy_sub[cur, 1], s=5, c='k') # non-clustered points
        continue
    plt.scatter(Energy_sub[cur, 0], Energy_sub[cur, 1], s=10, c=clr, edgecolors='k')
    plt.scatter(Energy_sub[cur & core_indices2, 0], Energy_sub[cur & core_indices2, 1], s=10, c=clr, \
                            marker='o', edgecolors='k')
plt.xlabel('Energy (keV)',fontsize=20)
plt.ylabel('Energy (keV)',fontsize=20)
plt.xlim(0, 140)
plt.ylim(0, 140)
plt.grid(True)
###### check each DBSCAN clustering results
clrs = plt.cm.Spectral(np.linspace(0, 0.8, 2*y_unique2.size))
plt.figure(figsize=(12, 12), facecolor='w')
plt.grid('on')
for k, clr in zip(y_unique2, clrs):
    cur = (y_hat2 == k)
    if k == 3:
        plt.scatter(Energy_sub[cur, 0], Energy_sub[cur, 1], s=5, c='k') # non-clustered points
        continue
#    plt.scatter(Energy_sub[cur, 0], Energy_sub[cur, 1], s=5, c=clr, edgecolors='k')
#    plt.scatter(Energy_sub[cur & core_indices2, 0], Energy_sub[cur & core_indices2, 1], s=10, c=clr, \
#                            marker='o', edgecolors='k')
plt.xlabel('Energy (keV)',fontsize=20)
plt.ylabel('Energy (keV)',fontsize=20)
plt.xlim(0, 90)
plt.ylim(0, 90)
plt.grid(True)
########################################################################################################
##### Model Fitting III
# Set Fitting Parameters
eps, min_samples = (1, 100) # 122 keV, all possible CS events
model3 = DBSCAN( eps=eps, min_samples=min_samples )
model3.fit( Energy_sub )
y_hat3 = model3.labels_

core_indices3 = np.zeros_like(y_hat3, dtype=bool) # create zero/boolean array with the same length
core_indices3[model3.core_sample_indices_] = True # 核样本的目录 < (label != 0)
 
y_unique3 = np.unique(y_hat3) # extract different Labels
n_clusters3 = y_unique3.size - (1 if -1 in y_hat3 else 0)
print(y_unique3, 'clustering number is :', n_clusters3)

# Plot the DBSCAN clustering results
clrs = plt.cm.Spectral(np.linspace(0, 0.8, 2*y_unique3.size))
plt.figure(figsize=(12, 12), facecolor='w')
plt.grid('on')
for k, clr in zip(y_unique3, clrs):
    cur = (y_hat3 == k)
    if k == -1:
        plt.scatter(Energy_sub[cur, 0], Energy_sub[cur, 1], s=5, c='k') # non-clustered points
        continue
    plt.scatter(Energy_sub[cur, 0], Energy_sub[cur, 1], s=10, c=clr, edgecolors='k')
    plt.scatter(Energy_sub[cur & core_indices3, 0], Energy_sub[cur & core_indices3, 1], s=10, c=clr, \
                            marker='o', edgecolors='k')
plt.xlabel('Energy (keV)',fontsize=20)
plt.ylabel('Energy (keV)',fontsize=20)
plt.xlim(0, 90)
plt.ylim(0, 90)
plt.grid(True)
###### check each DBSCAN clustering results
clrs = plt.cm.Spectral(np.linspace(0, 0.8, 2*y_unique3.size))
plt.figure(figsize=(12, 12), facecolor='w')
plt.grid('on')
for k, clr in zip(y_unique3, clrs):
    cur = (y_hat3 == k)
    if k == 3:
        plt.scatter(Energy_sub[cur, 0], Energy_sub[cur, 1], s=5, c='k') # non-clustered points
        continue
#    plt.scatter(Energy_sub[cur, 0], Energy_sub[cur, 1], s=5, c=clr, edgecolors='k')
#    plt.scatter(Energy_sub[cur & core_indices2, 0], Energy_sub[cur & core_indices2, 1], s=10, c=clr, \
#                            marker='o', edgecolors='k')
plt.xlabel('Energy (keV)',fontsize=20)
plt.ylabel('Energy (keV)',fontsize=20)
plt.xlim(0, 90)
plt.ylim(0, 90)
plt.grid(True)
########################################################################################################



####################### Extract the cluster in the ROI #################################################
##### Reorganize the clustered scattering points
y_hat = np.array([-1]*len(y_hat1))
y_hat[np.where( (y_hat1 != -1) ),] = 0
y_hat[reduce( np.intersect1d, (np.where(y_hat == -1),np.where(Energy_E1_E2_sum[:,0] < 50), \
                               np.where(Energy_E1_E2_sum[:,1] < 50),\
                               np.where(y_hat3 != -1)) ), ] = 0
y_hat[reduce( np.intersect1d, (np.where(y_hat == 0),np.where(Energy_E1_E2_sum[:,0] >= 20), \
                               np.where(Energy_E1_E2_sum[:,1] >= 20),np.where(Energy_sum >= 80)) ), ] = -1

y_hat[np.where(y_hat2 == 1),] = 1
y_hat[np.where(y_hat2 == 3),] = 2

y_hat[reduce( np.intersect1d, (np.where(y_hat == 0),np.where(Energy_E1_E2_sum[:,0] <= 9),np.where(Energy_sum <= 75)) ), ] = -1
y_hat[reduce( np.intersect1d, (np.where(y_hat == 0),np.where(Energy_E1_E2_sum[:,1] <= 9),np.where(Energy_sum <= 75)) ), ] = -1

y_unique = np.unique(y_hat)
cluster_lab_0 = Energy_sub[np.where( ( y_hat == 0 ) )]
cluster_lab_1 = Energy_sub[np.where( ( y_hat == 1 ) )]
cluster_lab_2 = Energy_sub[np.where( ( y_hat == 2 ) )]
cluster_lab_noise = Energy_sub[np.where( ( y_hat == -1 ) )]

###### check each DBSCAN clustering results
plt.figure(figsize=(12, 12), facecolor='w')
plt.grid('on')
for k in y_unique1:
    cur = (y_hat == k)
    if k == -1:
        plt.scatter(Energy_sub[cur, 0], Energy_sub[cur, 1], s=.5, c='k') # non-clustered points
        continue
    if k == 0:
        plt.scatter(Energy_sub[cur, 0], Energy_sub[cur, 1], s=4, c='b') # non-clustered points
        continue
    if k == 1:
        plt.scatter(Energy_sub[cur, 0], Energy_sub[cur, 1], s=4, c='r') # non-clustered points
        continue
    if k == 2:
        plt.scatter(Energy_sub[cur, 0], Energy_sub[cur, 1], s=4, c='y') # non-clustered points
        continue
#    plt.scatter(Energy_sub[cur, 0], Energy_sub[cur, 1], s=5, c=clr, edgecolors='k')
#    plt.scatter(Energy_sub[cur & core_indices2, 0], Energy_sub[cur & core_indices2, 1], s=10, c=clr, \
#                            marker='o', edgecolors='k')
plt.xlabel('Energy (keV)',fontsize=20)
plt.ylabel('Energy (keV)',fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlim(0, 90)
plt.ylim(0, 90)
plt.grid(True)
######################################################################################################## 
        

####################### "Rot -> MC Shifting -> Rot" CSC function #######################################
basis_old =np.mat( [ [1,0],
                     [0,1] ] ) #[x, y]
basis_new = np.mat( [ [  1/np.sqrt(2),  1/np.sqrt(2) ],
                      [ -1/np.sqrt(2),  1/np.sqrt(2) ] ] )
##### Initialize the CSC object, charge sharing band correction
CSC_81_2pix = CSC( CS_dim=CS_dim, basis_old=basis_old, basis_new=basis_new, peak_energy=81, max_energy_range=140, seg_size=2 )
cluster_lab_0_corrected, wet_x, wet_w_81, shift_w_81, seg_unit = CSC_81_2pix.Pix2_Correction( CS_data_labeled = cluster_lab_0 )
# check the scattering plot and MC plot
CSC.Scatter2D_plot(x=cluster_lab_0_corrected[:,0].tolist(), y=cluster_lab_0_corrected[:,1].tolist(), x_lim_left=0, x_lim_right=90, \
                   y_lim_left=0, y_lim_right=90 )
CSC.Line_plot(x=wet_x, y=wet_w_81, color='blue', x_label='Spatial-axis', y_label='Energy-axis')
CSC.Line_plot(x=wet_x, y=shift_w_81, color='red', x_label='Spatial-axis', y_label='Energy-axis')
##### fluorescence events correction
cluster_lab_1_corrected, wet_x, wet_w, shift_w, seg_unit = CSC_81_2pix.Pix2_Correction( CS_data_labeled = cluster_lab_1 )
cluster_lab_2_corrected, wet_x, wet_w, shift_w, seg_unit = CSC_81_2pix.Pix2_Correction( CS_data_labeled = cluster_lab_2 )
##### plot the local energy spectrum before and after CS correction
Energy_sub_corrected = np.vstack( (cluster_lab_noise, cluster_lab_0_corrected, cluster_lab_1_corrected, cluster_lab_2_corrected) )
CSC.Scatter2D_plot(x=Energy_sub_corrected[:,0].tolist(), y=Energy_sub_corrected[:,1].tolist(), x_lim_left=0, x_lim_right=90, \
                   y_lim_left=0, y_lim_right=90 )

CSC.Histogram_plot(Hist=Energy_sum, Bins=100, x_lim_low=20, x_lim_high=140)
CSC.Histogram_plot(Hist=np.sum( Energy_sub_corrected, axis=1 ).reshape(-1,1), Bins=100, x_lim_low=20, x_lim_high=140)
#################################################################################################################





#################################################################################################################
################################### Load the 122-keV charge sharing events ######################################
CS_data = pd.read_csv( 'C:\Jiajin\Mfile\Training_Sample_Analysis\Pix2Sharing.csv' )
Energy = CS_data.iloc[:, :].values   #取第一行
CS_dim = Energy.shape[1]

Energy_sum = np.sum( Energy, axis=1 ).reshape(-1,1)
Energy_E1_E2_sum = Energy[ np.intersect1d(np.where(Energy_sum >= 105)[0],np.where(Energy_sum <= 130)[0]) ]
Energy_E1_E2_else = np.delete( Energy, np.intersect1d(np.where(Energy_sum >= 105)[0],np.where(Energy_sum <= 130)[0]), axis =0 )

Energy_sum = np.sum( Energy_E1_E2_sum, axis=1 ).reshape(-1,1)
Energy_E1_E2_sum = np.hstack( (Energy_E1_E2_sum, Energy_sum) )

##### plot the histogram of sum_Energy within the selected ROI
CSC.Histogram_barplot(Hist=Energy_sum,Bins=100,x_lim_low=20,x_lim_high=140)

##### plot the raw scatter figures within the selected ROI
CSC.Scatter2D_plot(x=Energy_E1_E2_sum[:,0],y=Energy_E1_E2_sum[:,1],x_lim_left=0, x_lim_right=140, \
                   y_lim_left=0, y_lim_right=140)
########################################################################################################




####################### DBSCAN Clustering and Plot the results #########################################
Energy_sub = np.vstack( ( Energy_E1_E2_sum[:,0], Energy_E1_E2_sum[:,1] ) ).T

##### Model Fitting I        
# Set Fitting Parameters
eps, min_samples = (1, 30) # 122 keV, high density CS events
# eps, min_samples = (1, 100) # 122 keV, high density CS events

model1 = DBSCAN( eps=eps, min_samples=min_samples )
model1.fit( Energy_sub )
y_hat1 = model1.labels_

core_indices1 = np.zeros_like(y_hat1, dtype=bool) # create zero/boolean array with the same length
core_indices1[model1.core_sample_indices_] = True # 核样本的目录 < (label != 0)

y_unique1 = np.unique(y_hat1) # extract different Labels
n_clusters1 = y_unique1.size - (1 if -1 in y_hat1 else 0)
print(y_unique1, 'clustering number is :', n_clusters1)

# Plot the DBSCAN clustering results
clrs = plt.cm.Spectral(np.linspace(0, 0.8, 2*y_unique1.size))
#clrs = ['k', 'r', 'b', 'g', 'y']
plt.figure(figsize=(12, 12), facecolor='w')
plt.grid('on')
for k, clr in zip(y_unique1, clrs):
    cur = (y_hat1 == k)
    if k == -1:
        plt.scatter(Energy_sub[cur, 0], Energy_sub[cur, 1], s=5, c='k') # non-clustered points
        continue
    plt.scatter(Energy_sub[cur, 0], Energy_sub[cur, 1], s=5, c=clr, edgecolors='k')
    plt.scatter(Energy_sub[cur & core_indices1, 0], Energy_sub[cur & core_indices1, 1], s=10, c=clr, \
                            marker='o', edgecolors='k')
plt.xlabel('Energy (keV)',fontsize=20)
plt.ylabel('Energy (keV)',fontsize=20)
plt.xlim(0, 140)
plt.ylim(0, 140)
plt.grid(True)
########################################################################################################
##### Model Fitting II
# Set Fitting Parameters
eps, min_samples = (1, 12) # 122 keV, all possible CS events
model2 = DBSCAN( eps=eps, min_samples=min_samples )
model2.fit( Energy_sub )
y_hat2 = model2.labels_

core_indices2 = np.zeros_like(y_hat2, dtype=bool) # create zero/boolean array with the same length
core_indices2[model2.core_sample_indices_] = True # 核样本的目录 < (label != 0)
 
y_unique2 = np.unique(y_hat2) # extract different Labels
n_clusters2 = y_unique2.size - (1 if -1 in y_hat2 else 0)
print(y_unique2, 'clustering number is :', n_clusters2)

# Plot the DBSCAN clustering results
clrs = plt.cm.Spectral(np.linspace(0, 0.8, 2*y_unique2.size))
plt.figure(figsize=(12, 12), facecolor='w')
plt.grid('on')
for k, clr in zip(y_unique2, clrs):
    cur = (y_hat2 == k)
    if k == -1:
        plt.scatter(Energy_sub[cur, 0], Energy_sub[cur, 1], s=5, c='k') # non-clustered points
        continue
    plt.scatter(Energy_sub[cur, 0], Energy_sub[cur, 1], s=10, c=clr, edgecolors='k')
    plt.scatter(Energy_sub[cur & core_indices2, 0], Energy_sub[cur & core_indices2, 1], s=10, c=clr, \
                            marker='o', edgecolors='k')
plt.xlabel('Energy (keV)',fontsize=20)
plt.ylabel('Energy (keV)',fontsize=20)
plt.xlim(0, 140)
plt.ylim(0, 140)
plt.grid(True)
########################################################################################################



####################### Extract the cluster in the ROI ##################################################
##### Reorganize the clustered scattering points
y_hat = np.array([-1]*len(y_hat1)) # initialize
# main band
y_hat[np.where( (y_hat2 != -1) ),] = 0
y_hat[np.where( (y_hat2 == 11) | (y_hat2 == 12)| (y_hat2 == 13)| \
               (y_hat2 == 16)| (y_hat2 == 17)| (y_hat2 == 18) | (y_hat2 == 20)),] = -1
# flo up
y_hat[np.where( (y_hat1 == 1) | (y_hat1 == 4) | (y_hat2 == 15) ),] = 1
# flo down
y_hat[np.where( (y_hat1 == 3) | (y_hat2 == 5) | (y_hat2 == 14) | (y_hat2 == 19) ),] = 2
# little modification
y_hat[reduce( np.intersect1d, (np.where(y_hat == 0),np.where(Energy_sum <= 116),np.where(Energy_sub[:,0] <= 10)) ), ] = -1
y_hat[reduce( np.intersect1d, (np.where(y_hat == 0),np.where(Energy_sum <= 116),np.where(Energy_sub[:,1] <= 10)) ), ] = -1
y_hat[reduce( np.intersect1d, (np.where(y_hat == 0),np.where(Energy_sub[:,0] <= 100),\
                               np.where(Energy_sub[:,0] >= 90),np.where(Energy_sum >= 119)) ), ] = 2
y_hat[reduce( np.intersect1d, (np.where(y_hat == 0),np.where(Energy_sub[:,0] <= 90),\
                               np.where(Energy_sub[:,0] >= 80),np.where(Energy_sum >= 118)) ), ] = 2
    
y_hat[reduce( np.intersect1d, (np.where(y_hat == 0),np.where(Energy_sub[:,1] <= 100),\
                               np.where(Energy_sub[:,1] >= 90),np.where(Energy_sum >= 119)) ), ] = 1
y_hat[reduce( np.intersect1d, (np.where(y_hat == 0),np.where(Energy_sub[:,1] <= 90),\
                               np.where(Energy_sub[:,1] >= 80),np.where(Energy_sum >= 118)) ), ] = 1
y_unique = np.unique(y_hat)

##### divide into different cluster by the labels
cluster_lab_0 = Energy_sub[np.where( ( y_hat == 0 ) )]
cluster_lab_1 = Energy_sub[np.where( ( y_hat == 1 ) )]
cluster_lab_2 = Energy_sub[np.where( ( y_hat == 2 ) )]
cluster_lab_noise = Energy_sub[np.where( ( y_hat == -1 ) )]

##### Check if reorganization is correct
plt.figure(figsize=(12, 12), facecolor='w')
plt.grid('on')
for k in y_unique:
    cur = (y_hat == k)
    if k == -1:
        plt.scatter(Energy_sub[cur, 0], Energy_sub[cur, 1], s=5, c='k') # non-clustered points
        continue
    if k == 0:
        plt.scatter(Energy_sub[cur, 0], Energy_sub[cur, 1], s=5, c='b') # non-clustered points
        continue
    if k == 1:
        plt.scatter(Energy_sub[cur, 0], Energy_sub[cur, 1], s=5, c='r') # non-clustered points
        continue
    if k == 2:
        plt.scatter(Energy_sub[cur, 0], Energy_sub[cur, 1], s=5, c='g') # non-clustered points
        continue
    #plt.scatter(Energy_sub[cur & core_indices2, 0], Energy_sub[cur & core_indices2, 1], s=10, c=clr, \
     #                       marker='o', edgecolors='k')
plt.xlabel('Energy (keV)',fontsize=20)
plt.ylabel('Energy (keV)',fontsize=20)
plt.xlim(0, 140)
plt.ylim(0, 140)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.grid(True)
########################################################################################################  
        

####################### "Rot -> MC Shifting -> Rot" CSC function #######################################
basis_old =np.mat( [ [1,0],
                     [0,1] ] ) #[x, y]
basis_new = np.mat( [ [  1/np.sqrt(2),  1/np.sqrt(2) ],
                      [ -1/np.sqrt(2),  1/np.sqrt(2) ] ] )
##### Initialize the CSC object, charge sharing band correction
seg_size = 2
CSC_122_2pix = CSC( CS_dim=CS_dim, basis_old=basis_old, basis_new=basis_new, peak_energy=122, max_energy_range=140, seg_size=seg_size )
cluster_lab_0_corrected, wet_x, wet_w_122, shift_w_122, seg_unit = CSC_122_2pix.Pix2_Correction( CS_data_labeled = cluster_lab_0 )
# check the scattering plot and MC plot
CSC.Scatter2D_plot(x=cluster_lab_0_corrected[:,0].tolist(), y=cluster_lab_0_corrected[:,1].tolist(), x_lim_left=0, x_lim_right=140, \
                   y_lim_left=0, y_lim_right=140 )

CSC.Line_plot(x=wet_x, y=wet_w_60, color='green', x_label='Spatial-axis', y_label='Energy-axis')
CSC.Line_plot(x=wet_x, y=wet_w_81, color='red', x_label='Spatial-axis', y_label='Energy-axis')
CSC.Line_plot(x=wet_x, y=wet_w_122, color='blue', x_label='Spatial-axis', y_label='Energy-axis')

CSC.Line_plot(x=wet_x, y=shift_w_60, color='green', x_label='Spatial-axis', y_label='Energy-axis')
CSC.Line_plot(x=wet_x, y=shift_w_81, color='red', x_label='Spatial-axis', y_label='Energy-axis')
CSC.Line_plot(x=wet_x, y=shift_w_122, color='blue', x_label='Spatial-axis', y_label='Energy-axis')

##### fluorescence events correction
cluster_lab_1_corrected, wet_x, wet_w, shift_w, seg_unit = CSC_122_2pix.Pix2_Correction( CS_data_labeled = cluster_lab_1 )
cluster_lab_2_corrected, wet_x, wet_w, shift_w, seg_unit = CSC_122_2pix.Pix2_Correction( CS_data_labeled = cluster_lab_2 )
##### plot the local energy spectrum before and after CS correction
Energy_sub_corrected = np.vstack( (cluster_lab_noise, cluster_lab_0_corrected, cluster_lab_1_corrected, cluster_lab_2_corrected) )
CSC.Scatter2D_plot(x=Energy_sub_corrected[:,0].tolist(), y=Energy_sub_corrected[:,1].tolist(), x_lim_left=0, x_lim_right=140, \
                   y_lim_left=0, y_lim_right=140 )

CSC.Histogram_barplot(Hist=Energy_sum, Bins=100, x_lim_low=20, x_lim_high=140)
CSC.Histogram_barplot(Hist=np.sum( Energy_sub_corrected, axis=1 ).reshape(-1,1), Bins=100, x_lim_low=20, x_lim_high=140)
################################################################################################################



########################## save the wet_w_energy and shift_w_energy ############################################
CSC.SaveFiles(var=wet_x, dim=0, var_name='wet_x', location="C:\Jiajin\Mfile\Training_Sample_Analysis\wet_x.csv")

CSC.SaveFiles(var=wet_w_60, dim=0, var_name='wet_w_60', location="C:\Jiajin\Mfile\Training_Sample_Analysis\wet_w_60.csv")
CSC.SaveFiles(var=wet_w_81, dim=0, var_name='wet_w_81', location="C:\Jiajin\Mfile\Training_Sample_Analysis\wet_w_81.csv")
CSC.SaveFiles(var=wet_w_122, dim=0, var_name='wet_w_122', location="C:\Jiajin\Mfile\Training_Sample_Analysis\wet_w_122.csv")

CSC.SaveFiles(var=shift_w_60, dim=0, var_name='shift_w_60', location="C:\Jiajin\Mfile\Training_Sample_Analysis\shift_w_60.csv")
CSC.SaveFiles(var=shift_w_81, dim=0, var_name='shift_w_81', location="C:\Jiajin\Mfile\Training_Sample_Analysis\shift_w_81.csv")
CSC.SaveFiles(var=shift_w_122, dim=0, var_name='shift_w_122', location="C:\Jiajin\Mfile\Training_Sample_Analysis\shift_w_122.csv")
################################################################################################################




####################### Smooth the shift curve and MC curve ####################################################
##### shift & wet & filter
# 60 keV
wet_w_60_filter = np.zeros(len(wet_w_60))
left = min( min( np.where(wet_w_60!=0) ) )
right = max( max( np.where(wet_w_60!=0) ) )
num = (right-left)//2 * 2 + 1
wet_w_60_filter[left:right+1,] = signal.savgol_filter(wet_w_60[left:right+1,], num, 5)
  
shift_w_60_filter = np.zeros(len(shift_w_60))
left = min( min( np.where(shift_w_60!=0) ) )
right = max( max( np.where(shift_w_60!=0) ) )
num = (right-left)//2 * 2 + 1
shift_w_60_filter[left:right+1,] = signal.savgol_filter(shift_w_60[left:right+1,], num, 5)

CSC.Line_plot(x=wet_x, y=shift_w_60, color='green', x_label='Spatial-axis', y_label='Energy-axis')
CSC.Line_plot(x=wet_x, y=shift_w_60_filter, color='green', x_label='Spatial-axis', y_label='Energy-axis')
CSC.Line_plot(x=wet_x, y=wet_w_60, color='green', x_label='Spatial-axis', y_label='Energy-axis')
CSC.Line_plot(x=wet_x, y=wet_w_60_filter, color='green', x_label='Spatial-axis', y_label='Energy-axis')

# 81 keV
wet_w_81_filter = np.zeros(len(wet_w_81))
left = min( min( np.where(wet_w_81!=0) ) )
right = max( max( np.where(wet_w_81!=0) ) )
num = (right-left)//2 * 2 + 1
wet_w_81_filter[left:right+1,] = signal.savgol_filter(wet_w_81[left:right+1,], num, 5)
  
shift_w_81_filter = np.zeros(len(shift_w_81))
left = min( min( np.where(shift_w_81!=0) ) )
right = max( max( np.where(shift_w_81!=0) ) )
num = (right-left)//2 * 2 + 1
shift_w_81_filter[left:right+1,] = signal.savgol_filter(shift_w_81[left:right+1,], num, 5)

CSC.Line_plot(x=wet_x, y=shift_w_81, color='red', x_label='Spatial-axis', y_label='Energy-axis')
CSC.Line_plot(x=wet_x, y=shift_w_81_filter, color='red', x_label='Spatial-axis', y_label='Energy-axis')
CSC.Line_plot(x=wet_x, y=wet_w_81, color='red', x_label='Spatial-axis', y_label='Energy-axis')
CSC.Line_plot(x=wet_x, y=wet_w_81_filter, color='red', x_label='Spatial-axis', y_label='Energy-axis')

# 122 keV
wet_w_122_filter = np.zeros(len(wet_w_122))
left = min( min( np.where(wet_w_122!=0) ) )
right = max( max( np.where(wet_w_122!=0) ) )
num = (right-left)//2 * 2 + 1
wet_w_122_filter[left:right+1,] = signal.savgol_filter(wet_w_122[left:right+1,], num, 5)
  
shift_w_122_filter = np.zeros(len(shift_w_122))
left = min( min( np.where(shift_w_122!=0) ) )
right = max( max( np.where(shift_w_122!=0) ) )
num = (right-left)//2 * 2 + 1
shift_w_122_filter[left:right+1,] = signal.savgol_filter(shift_w_122[left:right+1,], num, 5)

CSC.Line_plot(x=wet_x, y=shift_w_122, color='blue', x_label='Spatial-axis', y_label='Energy-axis')
CSC.Line_plot(x=wet_x, y=shift_w_122_filter, color='blue', x_label='Spatial-axis', y_label='Energy-axis')
CSC.Line_plot(x=wet_x, y=wet_w_122, color='blue', x_label='Spatial-axis', y_label='Energy-axis')
CSC.Line_plot(x=wet_x, y=wet_w_122_filter, color='blue', x_label='Spatial-axis', y_label='Energy-axis')



##### check the linearity
x = np.array((60/np.sqrt(2),81/np.sqrt(2),122/np.sqrt(2)))
for ii in np.arange(28,70):
    y = np.array( (shift_w_60_filter[ii,],shift_w_81_filter[ii,],shift_w_122_filter[ii,]) )
    CSC.Line_plot(x=x, y=y, color='blue', x_label='Energy-axis', y_label='Compensation energy')

x = np.array((60/np.sqrt(2),81/np.sqrt(2),122/np.sqrt(2)))
for ii in np.arange(28,70):
    y = np.array( (wet_w_60_filter[ii,],wet_w_81_filter[ii,],wet_w_122_filter[ii,]) )
    CSC.Line_plot(x=x, y=y, color='blue', x_label='Energy-axis', y_label='Compensation energy')
##################################################################################################################




####################### interpolation and visualization ##########################################################
x = wet_x                            # center position of each grid
y = np.arange(1,140/np.sqrt(2),2)    # center position of each grid
z = np.zeros( (len(y),len(x)) )      # charge compensation of each grid

y_w1 = wet_w_60_filter
y_w2 = wet_w_81_filter
y_w3 = wet_w_122_filter

y_s1 = shift_w_60_filter
y_s2 = shift_w_81_filter
y_s3 = shift_w_122_filter

index = min( min( np.where(y_s1 != 0) ) )
y_s1_left = y_s1[index,]
x_s1_left = wet_x[index,]
index = max( max( np.where(y_s1 != 0) ) )
y_s1_right = y_s1[index,]
x_s1_right = wet_x[index,]

index = min( min( np.where(y_s2 != 0) ) )
y_s2_left = y_s2[index,]
x_s2_left = wet_x[index,]
index = max( max( np.where(y_s2 != 0) ) )
y_s2_right = y_s2[index,]
x_s2_right = wet_x[index,]

index = min( min( np.where(y_s3 != 0) ) )
y_s3_left = y_s3[index,]
x_s3_left = wet_x[index,]
index = max( max( np.where(y_s3 != 0) ) )
y_s3_right = y_s3[index,]
x_s3_right = wet_x[index,]

CSC.Line_plot(x=wet_x, y=y_w1, color='green', x_label='Spatial-axis', y_label='Energy-axis')
CSC.Line_plot(x=wet_x, y=y_w2, color='red', x_label='Spatial-axis', y_label='Energy-axis')
CSC.Line_plot(x=wet_x, y=y_w3, color='blue', x_label='Spatial-axis', y_label='Energy-axis')

CSC.Line_plot(x=wet_x, y=y_s1, color='green', x_label='Spatial-axis', y_label='Energy-axis')
CSC.Line_plot(x=wet_x, y=y_s2, color='red', x_label='Spatial-axis', y_label='Energy-axis')
CSC.Line_plot(x=wet_x, y=y_s3, color='blue', x_label='Spatial-axis', y_label='Energy-axis')

##### Interpolation
nx,ny = (-1,-1)
for x0 in x:
    nx += 1
    y_w0 = np.fabs(x0)
    flag = 0 # flag is to tell if the point is on the edge or not
    for y0 in y:
        ny += 1
        if ny == len(y):
            ny = 0
        if y0>=y_w0:
            if flag == 0: # flag is to tell if the point is on the edge or not
                flag = 1
                if y0<=y_w1[nx,]:
                    if x0<=0:
                        z[ny,nx] = y_s1_left
                    else:
                        z[ny,nx] = y_s1_right
                if y0>y_w1[nx,] and y0<=y_w2[nx,]:
                    if x0<=0:
                        z[ny,nx] = (y_s2_left-y_s1_left)/(x_s2_left-x_s1_left)*(x0-x_s1_left) + y_s1_left
                    else:
                        z[ny,nx] = (y_s2_right-y_s1_right)/(x_s2_right-x_s1_right)*(x0-x_s1_right) + y_s1_right
                if y0>y_w2[nx,]:
                    if x0<=0:
                        z[ny,nx] = (y_s3_left-y_s2_left)/(x_s3_left-x_s2_left)*(x0-x_s2_left) + y_s2_left
                    else:
                        z[ny,nx] = (y_s3_right-y_s2_right)/(x_s3_right-x_s2_right)*(x0-x_s2_right) + y_s2_right
                z_edge = z[ny,nx]
            else:
                if y0<=y_w1[nx,]:
                    z[ny,nx] = (y_s1[nx,]-z_edge)*(y0-y_w0)/(y_w1[nx,]-y_w0) + z_edge
                if y0>y_w1[nx,] and y0<=y_w2[nx,]:
                    if y_w1[nx,] != 0:
                        z[ny,nx] = (y_s2[nx,]-y_s1[nx,])*(y0-y_w1[nx,])/(y_w2[nx,]-y_w1[nx,])+y_s1[nx,]
                    else:
                        z[ny,nx] = (y_s2[nx,]-z_edge)*(y0-y_w0)/(y_w2[nx,]-y_w0)+z_edge
                if y0>y_w2[nx,]:
                    if y_w2[nx,] != 0:
                        z[ny,nx] = (y_s3[nx,]-y_s2[nx,])*(y0-y_w2[nx,])/(y_w3[nx,]-y_w2[nx,])+y_s2[nx,]
                    else:
                        z[ny,nx] = (y_s3[nx,]-z_edge)*(y0-y_w0)/(y_w3[nx,]-y_w0)+z_edge

##### Visualize the interpolation map                         
X = np.tile( x.reshape(-1,1), (1,len(y)) ).T
Y = np.tile( y.reshape(-1,1).T, (len(x),1) ).T
Z = z            
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

##### Visualize the cross section
### x-direction
for ii in np.arange(5,49,5):
    x=x
    y=Z[ii,:]
    CSC.Line_plot(x=x, y=y, color='blue', x_label='Spatial-axis', y_label='Charge loss compensation(0.7 keV)')
'''    
    y_filter = np.zeros(len(y))
    left = min( min( np.where(y>0) ) )
    right = max( max( np.where(y>0) ) )
    num = (right-left)//2 * 2 + 1
    y_filter[left:right,] = signal.savgol_filter(y[left:right,], num, 5)
    CSC.Line_plot(x=x, y=y_filter, color='red', x_label='Spatial-axis', y_label='Charge loss compensation(0.7 keV)')
'''   
### y-direction
x = wet_x                            # center position of each grid
y = np.arange(1,140/np.sqrt(2),2)    # center position of each grid
for ii in np.arange(0,99,5):
    CSC.Line_plot(x=y, y=Z[:,ii], color='red', x_label='Energy-axis', y_label='Charge loss compensation(0.7 keV)')
##################################################################################################################



'''
####################### Full range correction I ##################################################################
from charge_sharing_correction import charge_sharing_correction as CSC # class file

##### reload full range 2-pixel charge sharing data
CS_data = pd.read_csv( 'C:\Jiajin\Mfile\Training_Sample_Analysis\Pix2Sharing.csv' )
Energy = CS_data.iloc[:, :].values   #取第一行
CS_dim = Energy.shape[1]

Energy_sum = np.sum( Energy, axis=1 ).reshape(-1,1)
CSC.Histogram_lineplot(Hist=Energy_sum,Bins=1000,x_lim_low=20,x_lim_high=140,color='red')

Energy_E1_E2 = Energy[ np.where(Energy_sum <= 140)[0] ]
Energy_sum = np.sum( Energy_E1_E2, axis=1 ).reshape(-1,1)
# plot
CSC.Scatter2D_plot(x=Energy_E1_E2[:,0],y=Energy_E1_E2[:,1],x_lim_left=0, x_lim_right=140, y_lim_left=0, y_lim_right=140)

##### calculate the correction curve for different energy peak
basis_old =np.mat( [ [1,0],
                     [0,1] ] ) #[x, y]
basis_new = np.mat( [ [  1/np.sqrt(2),  1/np.sqrt(2) ],
                      [ -1/np.sqrt(2),  1/np.sqrt(2) ] ] )

peak_energy = np.array( ( 60, 81, 122, 136 ) )
CSC_2pix = CSC( CS_dim=CS_dim, basis_old=basis_old, basis_new=basis_new, peak_energy=60, max_energy_range=140, seg_size=2 )
Energy_E1_E2_rot = CSC_2pix.Corr_rot(basis_old=CSC_2pix.basis_old, basis_new=CSC_2pix.basis_new, original_position=Energy_E1_E2.T)
# plot
CSC.Scatter2D_plot(Energy_E1_E2_rot[:,0].tolist(),Energy_E1_E2_rot[:,1].tolist(), x_lim_left=-100, x_lim_right=100,\
                   y_lim_left=30, y_lim_right=100)

##### reload the charge loss curve
data = pd.read_csv('C:\Jiajin\Mfile\Training_Sample_Analysis\wet_w_60.csv')
wet_w_60 = data.iloc[:, :].values.reshape(-1,)  #取第一行
wet_w_60_filter = np.zeros(len(wet_w_60))
wet_w_60_filter[31:66,] = signal.savgol_filter(wet_w_60[31:66,], 35, 5)
CSC.Line_plot(x=wet_x, y=wet_w_60_filter, color='green', x_label='Spatial-axis', y_label='Energy-axis')

data = pd.read_csv('C:\Jiajin\Mfile\Training_Sample_Analysis\wet_w_81.csv')
wet_w_81 = data.iloc[:, :].values.reshape(-1,)   #取第一行
wet_w_81_filter = np.zeros(len(wet_w_81))
wet_w_81_filter[25:74,] = signal.savgol_filter(wet_w_81[25:74,], 47, 5)
CSC.Line_plot(x=wet_x, y=wet_w_81_filter, color='red', x_label='Spatial-axis', y_label='Energy-axis')

data = pd.read_csv('C:\Jiajin\Mfile\Training_Sample_Analysis\wet_w_122.csv')
wet_w_122 = data.iloc[:, :].values.reshape(-1,)   #取第一行
wet_w_122_filter = np.zeros(len(wet_w_122))
wet_w_122_filter[10:88,] = signal.savgol_filter(wet_w_122[10:88,], 77, 5)
CSC.Line_plot(x=wet_x, y=wet_w_122_filter, color='blue', x_label='Spatial-axis', y_label='Energy-axis')

##### band correction 
CSC.Line_plot(x=wet_x, y=shift_w_60_filter, color='red', x_label='Energy-axis', y_label='Charge loss compensation(0.7 keV)')

ii = -1
delta = 2.5
for seg_left in seg_unit:
    ii += 1
    seg_right = seg_left + seg_size
    if wet_w_122_filter[ii,] != 0:
        index = reduce( np.intersect1d, (np.where( Energy_E1_E2_rot[:,0] >= seg_left )[0],\
                                         np.where( Energy_E1_E2_rot[:,0] < seg_right )[0],\
                                         np.where( Energy_E1_E2_rot[:,1] >= (wet_w_122_filter[ii]-delta) )[0],\
                                         np.where( Energy_E1_E2_rot[:,1] < (wet_w_122_filter[ii]+delta) )[0]) )
        tmp = Energy_E1_E2_rot[index, : ]
        if len(tmp) != 0:
            shift = np.array([1]*len(tmp)).reshape(-1,1) * shift_w_122_filter[ii,]
            tmp_corrected = np.hstack( ( tmp[:,0].reshape(-1,1), tmp[:,1].reshape(-1,1)+shift) )
        Energy_E1_E2_rot[index, : ] = tmp_corrected
        
    if wet_w_81_filter[ii,] != 0:
        index = reduce( np.intersect1d, (np.where( Energy_E1_E2_rot[:,0] >= seg_left )[0],\
                                         np.where( Energy_E1_E2_rot[:,0] < seg_right )[0],\
                                         np.where( Energy_E1_E2_rot[:,1] >= (wet_w_81_filter[ii]-delta) )[0],\
                                         np.where( Energy_E1_E2_rot[:,1] < (wet_w_81_filter[ii]+delta) )[0]) )
        tmp = Energy_E1_E2_rot[index, : ]
        if len(tmp) != 0:
            shift = np.array([1]*len(tmp)).reshape(-1,1) * shift_w_81_filter[ii,]
            tmp_corrected = np.hstack( ( tmp[:,0].reshape(-1,1), tmp[:,1].reshape(-1,1)+shift) )
        Energy_E1_E2_rot[index, : ] = tmp_corrected
        
    if wet_w_60_filter[ii,] != 0:
        index = reduce( np.intersect1d, (np.where( Energy_E1_E2_rot[:,0] >= seg_left )[0],\
                                         np.where( Energy_E1_E2_rot[:,0] < seg_right )[0],\
                                         np.where( Energy_E1_E2_rot[:,1] >= (wet_w_60_filter[ii]-delta) )[0],\
                                         np.where( Energy_E1_E2_rot[:,1] < (wet_w_60_filter[ii]+delta) )[0]) )
        tmp = Energy_E1_E2_rot[index, : ]
        if len(tmp) != 0:
            shift = np.array([1]*len(tmp)).reshape(-1,1) * shift_w_60_filter[ii,]
            tmp_corrected = np.hstack( ( tmp[:,0].reshape(-1,1), tmp[:,1].reshape(-1,1)+shift) )
        Energy_E1_E2_rot[index, : ] = tmp_corrected
                            
# scatter/histogram plot
CSC.Scatter2D_plot(Energy_E1_E2_rot[:,0].tolist(),Energy_E1_E2_rot[:,1].tolist(), x_lim_left=-100, x_lim_right=100,\
                   y_lim_left=60, y_lim_right=100)

Energy_E1_E2_corrected = CSC_2pix.Corr_rot(basis_old=CSC_2pix.basis_new, basis_new=CSC_2pix.basis_old, original_position=Energy_E1_E2_rot.T)        
Energy_E1_E2_corrected_sum = np.sum( Energy_E1_E2_corrected, axis=1 ).reshape(-1,1)
CSC.Histogram_lineplot(Hist=Energy_sum, Bins=1000, x_lim_high=140, x_lim_low=20, color='blue')
CSC.Histogram_lineplot(Hist=Energy_E1_E2_corrected_sum, Bins=1000, x_lim_high=140, x_lim_low=20, color='red')
##################################################################################################################
'''





################################## extract the fluorescence events ##############################################
Energy_E1_E2_rot = CSC_2pix.Corr_rot(basis_old=CSC_2pix.basis_old, basis_new=CSC_2pix.basis_new, original_position=Energy_E1_E2.T)
Energy_E1_E2_label = np.array([-1]*len(Energy_E1_E2_rot)) # initialize the fluorescence label

################### 122 keV #################
y = CSC_2pix.Super_plane(energy=122)
x = CSC_2pix.Super_plane(energy=122)-CSC_2pix.Super_plane(energy=36)+5
delta = 2.5
Fluor_index = reduce( np.intersect1d, (np.where( Energy_E1_E2_rot[:,0] >= -x )[0],\
                                       np.where( Energy_E1_E2_rot[:,0] < x )[0],\
                                       np.where( Energy_E1_E2_rot[:,1] >= (y-delta) )[0],\
                                       np.where( Energy_E1_E2_rot[:,1] < (y+delta) )[0]) )
Fluor = Energy_E1_E2_rot[Fluor_index, :]
CSC.Scatter2D_plot(Fluor[:,0].tolist(),Fluor[:,1].tolist(), x_lim_left=-100, x_lim_right=100,\
                   y_lim_left=60, y_lim_right=100)

##### Model Fitting 
# Set Fitting Parameters
eps, min_samples = (1, 25) # 122 keV luorescence events
model1 = DBSCAN( eps=eps, min_samples=min_samples )
model1.fit( Fluor )
y_hat1 = model1.labels_

core_indices1 = np.zeros_like(y_hat1, dtype=bool) # create zero/boolean array with the same length
core_indices1[model1.core_sample_indices_] = True # 核样本的目录 < (label != 0)
 
y_unique1 = np.unique(y_hat1) # extract different Labels
n_clusters1 = y_unique1.size - (1 if -1 in y_hat1 else 0)
print(y_unique1, 'clustering number is :', n_clusters1)

# Plot the DBSCAN clustering results
clrs = plt.cm.Spectral(np.linspace(0, 0.8, 2*y_unique1.size))
plt.figure(figsize=(12, 12), facecolor='w')
plt.grid('on')
for k, clr in zip(y_unique1, clrs):
    cur = (y_hat1 == k)
    if k == -1:
        plt.scatter(Fluor[cur, 0].tolist(), Fluor[cur, 1].tolist(), s=5, c='k') # non-clustered points
        continue
    #plt.scatter(Fluor[cur, 0].tolist(), Fluor[cur, 1].tolist(), s=5, c=clr, edgecolors='k')
    #plt.scatter(Fluor[cur & core_indices1, 0].tolist(), Fluor[cur & core_indices1, 1].tolist(), s=10, c=clr, \
       #                     marker='o', edgecolors='k')
plt.xlabel('Energy (keV)',fontsize=20)
plt.ylabel('Energy (keV)',fontsize=20)
plt.xlim(-100, 100)
plt.ylim(60, 100)
plt.grid(True)

##### Reorganize
y_hat = np.array([-1]*len(y_hat1)) # initialize
y_hat[np.where( (y_hat1 == 0) | (y_hat1 == 2) ),] = 0

##### Check clustering results
plt.figure(figsize=(12, 12), facecolor='w')
plt.grid('on')
for k in y_unique1:
    cur = (y_hat == k)
    if k == -1:
        plt.scatter(Fluor[cur, 0].tolist(), Fluor[cur, 1].tolist(), s=5, c='k') # non-clustered points
        continue
    if k == 0:
        plt.scatter(Fluor[cur, 0].tolist(), Fluor[cur, 1].tolist(), s=5, c='b') # non-clustered points
        continue
plt.xlabel('Energy (keV)',fontsize=20)
plt.ylabel('Energy (keV)',fontsize=20)
plt.xlim(-100, 100)
plt.ylim(60, 100)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.grid(True)

##### label the fluorescence events
Energy_E1_E2_label[Fluor_index[(y_hat == 0),],] = 0



################### 81 keV #################
y = CSC_2pix.Super_plane(energy=81)
x = CSC_2pix.Super_plane(energy=81)-CSC_2pix.Super_plane(energy=36)+5
delta = 2.8
Fluor_index = reduce( np.intersect1d, (np.where( Energy_E1_E2_rot[:,0] >= -x )[0],\
                                       np.where( Energy_E1_E2_rot[:,0] < x )[0],\
                                       np.where( Energy_E1_E2_rot[:,1] >= (y-delta) )[0],\
                                       np.where( Energy_E1_E2_rot[:,1] < (y+delta) )[0]) )
Fluor = Energy_E1_E2_rot[Fluor_index, :]
CSC.Scatter2D_plot(Fluor[:,0].tolist(),Fluor[:,1].tolist(), x_lim_left=-100, x_lim_right=100,\
                   y_lim_left=50, y_lim_right=65)

##### Model Fitting 
# Set Fitting Parameters
eps, min_samples = (1, 20) # 122 keV luorescence events
model1 = DBSCAN( eps=eps, min_samples=min_samples )
model1.fit( Fluor )
y_hat1 = model1.labels_

core_indices1 = np.zeros_like(y_hat1, dtype=bool) # create zero/boolean array with the same length
core_indices1[model1.core_sample_indices_] = True # 核样本的目录 < (label != 0)
 
y_unique1 = np.unique(y_hat1) # extract different Labels
n_clusters1 = y_unique1.size - (1 if -1 in y_hat1 else 0)
print(y_unique1, 'clustering number is :', n_clusters1)

# Plot the DBSCAN clustering results
clrs = plt.cm.Spectral(np.linspace(0, 0.8, 2*y_unique1.size))
plt.figure(figsize=(12, 12), facecolor='w')
plt.grid('on')
for k, clr in zip(y_unique1, clrs):
    cur = (y_hat1 == k)
    if k == 1:
        plt.scatter(Fluor[cur, 0].tolist(), Fluor[cur, 1].tolist(), s=5, c='k') # non-clustered points
        continue
#    plt.scatter(Fluor[cur, 0].tolist(), Fluor[cur, 1].tolist(), s=5, c=clr, edgecolors='k')
#    plt.scatter(Fluor[cur & core_indices1, 0].tolist(), Fluor[cur & core_indices1, 1].tolist(), s=10, c=clr, \
#                     marker='o', edgecolors='k')
plt.xlabel('Energy (keV)',fontsize=20)
plt.ylabel('Energy (keV)',fontsize=20)
plt.xlim(-100, 100)
plt.ylim(50, 65)
plt.grid(True)

##### Reorganize
y_hat = np.array([-1]*len(y_hat1)) # initialize
y_hat[np.where( (y_hat1 == 0) | (y_hat1 == 1) ),] = 0

##### Check clustering results
plt.figure(figsize=(12, 12), facecolor='w')
plt.grid('on')
for k in y_unique1:
    cur = (y_hat == k)
    if k == -1:
        plt.scatter(Fluor[cur, 0].tolist(), Fluor[cur, 1].tolist(), s=5, c='k') # non-clustered points
        continue
    if k == 0:
        plt.scatter(Fluor[cur, 0].tolist(), Fluor[cur, 1].tolist(), s=5, c='b') # non-clustered points
        continue
plt.xlabel('Energy (keV)',fontsize=20)
plt.ylabel('Energy (keV)',fontsize=20)
plt.xlim(-100, 100)
plt.ylim(50, 65)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.grid(True)

##### label the fluorescence events
Energy_E1_E2_label[Fluor_index[(y_hat == 0),],] = 1



################### 60 keV #################
y = CSC_2pix.Super_plane(energy=60)
x = CSC_2pix.Super_plane(energy=60)-CSC_2pix.Super_plane(energy=36)+5
delta = 2.8
Fluor_index = reduce( np.intersect1d, (np.where( Energy_E1_E2_rot[:,0] >= -x )[0],\
                                       np.where( Energy_E1_E2_rot[:,0] < x )[0],\
                                       np.where( Energy_E1_E2_rot[:,1] >= (y-delta) )[0],\
                                       np.where( Energy_E1_E2_rot[:,1] < (y+delta) )[0]) )
Fluor = Energy_E1_E2_rot[Fluor_index, :]
CSC.Scatter2D_plot(Fluor[:,0].tolist(),Fluor[:,1].tolist(), x_lim_left=-100, x_lim_right=100,\
                   y_lim_left=30, y_lim_right=50)

##### Model Fitting 
# Set Fitting Parameters
eps, min_samples = (0.4, 15) # 122 keV luorescence events
model1 = DBSCAN( eps=eps, min_samples=min_samples )
model1.fit( Fluor )
y_hat1 = model1.labels_

core_indices1 = np.zeros_like(y_hat1, dtype=bool) # create zero/boolean array with the same length
core_indices1[model1.core_sample_indices_] = True # 核样本的目录 < (label != 0)
 
y_unique1 = np.unique(y_hat1) # extract different Labels
n_clusters1 = y_unique1.size - (1 if -1 in y_hat1 else 0)
print(y_unique1, 'clustering number is :', n_clusters1)

# Plot the DBSCAN clustering results
clrs = plt.cm.Spectral(np.linspace(0, 0.8, 2*y_unique1.size))
plt.figure(figsize=(12, 12), facecolor='w')
plt.grid('on')
for k, clr in zip(y_unique1, clrs):
    cur = (y_hat1 == k)
    if k == 14:
        plt.scatter(Fluor[cur, 0].tolist(), Fluor[cur, 1].tolist(), s=5, c='k') # non-clustered points
        continue
#    plt.scatter(Fluor[cur, 0].tolist(), Fluor[cur, 1].tolist(), s=5, c=clr, edgecolors='k')
#    plt.scatter(Fluor[cur & core_indices1, 0].tolist(), Fluor[cur & core_indices1, 1].tolist(), s=10, c=clr, \
#                     marker='o', edgecolors='k')
plt.xlabel('Energy (keV)',fontsize=20)
plt.ylabel('Energy (keV)',fontsize=20)
plt.xlim(-100, 100)
plt.ylim(30, 50)
plt.grid(True)

##### Reorganize
y_hat = np.array([-1]*len(y_hat1)) # initialize
y_hat[np.where( (y_hat1 == 0) | (y_hat1 == 1) | (y_hat1 == 7) | (y_hat1 == 12) | (y_hat1 == 13)),] = 0

##### Check clustering results
plt.figure(figsize=(12, 12), facecolor='w')
plt.grid('on')
for k in y_unique1:
    cur = (y_hat == k)
    if k == -1:
        plt.scatter(Fluor[cur, 0].tolist(), Fluor[cur, 1].tolist(), s=5, c='k') # non-clustered points
        continue
    if k == 0:
        plt.scatter(Fluor[cur, 0].tolist(), Fluor[cur, 1].tolist(), s=5, c='b') # non-clustered points
        continue
plt.xlabel('Energy (keV)',fontsize=20)
plt.ylabel('Energy (keV)',fontsize=20)
plt.xlim(-100, 100)
plt.ylim(30, 50)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.grid(True)

##### label the fluorescence events
Energy_E1_E2_label[Fluor_index[(y_hat == 0),],] = 2


##### full range check and visualization
Energy_E1_E2_unique = np.unique(Energy_E1_E2_label)
plt.figure(figsize=(12, 12), facecolor='w')
plt.grid('on')

for k in Energy_E1_E2_unique:
    cur = (Energy_E1_E2_label == k)
    if k == -1:
        plt.scatter(Energy_E1_E2_rot[cur, 0].tolist(), Energy_E1_E2_rot[cur, 1].tolist(), s=5, c='k') # non-clustered points
        continue
    if k == 0:
        plt.scatter(Energy_E1_E2_rot[cur, 0].tolist(), Energy_E1_E2_rot[cur, 1].tolist(), s=5, c='blue') # non-clustered points
        continue
    if k == 1:
        plt.scatter(Energy_E1_E2_rot[cur, 0].tolist(), Energy_E1_E2_rot[cur, 1].tolist(), s=5, c='red') # non-clustered points
        continue
    if k == 2:
        plt.scatter(Energy_E1_E2_rot[cur, 0].tolist(), Energy_E1_E2_rot[cur, 1].tolist(), s=5, c='green') # non-clustered points
        continue
plt.xlabel('Spatial-axis',fontsize=20)
plt.ylabel('Energy-axis',fontsize=20)
plt.xlim(-100, 100)
plt.ylim(30, 100)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.grid(True)

CSC.Line_plot(x=wet_x, y=wet_w_60_filter, color='green', x_label='Spatial-axis', y_label='Energy-axis')
CSC.Line_plot(x=wet_x, y=wet_w_81_filter, color='red', x_label='Spatial-axis', y_label='Energy-axis')
CSC.Line_plot(x=wet_x, y=wet_w_122_filter, color='blue', x_label='Spatial-axis', y_label='Energy-axis')
##################################################################################################################





####################### Full range correction II #################################################################
from charge_sharing_correction import charge_sharing_correction as CSC # class file

##### reload full range 2-pixel charge sharing data
CS_data = pd.read_csv( 'C:\Jiajin\Mfile\Training_Sample_Analysis\Pix2Sharing.csv' )
Energy = CS_data.iloc[:, :].values   #取第一行
CS_dim = Energy.shape[1]

Energy_sum = np.sum( Energy, axis=1 ).reshape(-1,1)
CSC.Histogram_lineplot(Hist=Energy_sum,Bins=1000,x_lim_low=20,x_lim_high=140,color='red')

Energy_E1_E2 = Energy[ np.where(Energy_sum <= 140)[0] ]
Energy_sum = np.sum( Energy_E1_E2, axis=1 ).reshape(-1,1)
# plot
CSC.Scatter2D_plot(x=Energy_E1_E2[:,0],y=Energy_E1_E2[:,1],x_lim_left=0, x_lim_right=140, y_lim_left=0, y_lim_right=140)

##### calculate the correction curve for different energy peak
basis_old =np.mat( [ [1,0],
                     [0,1] ] ) #[x, y]
basis_new = np.mat( [ [  1/np.sqrt(2),  1/np.sqrt(2) ],
                      [ -1/np.sqrt(2),  1/np.sqrt(2) ] ] )

peak_energy = np.array( ( 60, 81, 122, 136 ) )
CSC_2pix = CSC( CS_dim=CS_dim, basis_old=basis_old, basis_new=basis_new, peak_energy=60, max_energy_range=140, seg_size=2 )
Energy_E1_E2_rot = CSC_2pix.Corr_rot(basis_old=CSC_2pix.basis_old, basis_new=CSC_2pix.basis_new, original_position=Energy_E1_E2.T)
# plot
CSC.Scatter2D_plot(Energy_E1_E2_rot[:,0].tolist(),Energy_E1_E2_rot[:,1].tolist(), x_lim_left=-100, x_lim_right=100,\
                   y_lim_left=30, y_lim_right=100)

##### Saperate the fluorescence events
Energy_E1_E2_rot0 = Energy_E1_E2_rot[np.where(Energy_E1_E2_label == -1)]
Energy_E1_E2_rot1 = Energy_E1_E2_rot[np.where(Energy_E1_E2_label != -1)]
CSC.Scatter2D_plot(Energy_E1_E2_rot0[:,0].tolist(),Energy_E1_E2_rot0[:,1].tolist(), x_lim_left=-100, x_lim_right=100,\
                   y_lim_left=30, y_lim_right=100)


seg_unit2 = np.arange(0,140/np.sqrt(2)-1,seg_size)    # down side position of each grid
ii,jj = (-1,-1)
ncount = 0
for seg_left in seg_unit:
    ii += 1
    seg_right = seg_left + seg_size
    for seg_down in seg_unit2:
        jj += 1
        if jj == len(seg_unit2): # jj = 0 1 2 ... len(seg_y)-1
            jj = 0
        seg_up = seg_down + seg_size
        
        tmp = Energy_E1_E2_rot0[ reduce( np.intersect1d, (np.where(Energy_E1_E2_rot0[:,0] >= seg_left)[0],\
                                                          np.where(Energy_E1_E2_rot0[:,0] < seg_right)[0],\
                                                          np.where(Energy_E1_E2_rot0[:,1] >= seg_down)[0],\
                                                          np.where(Energy_E1_E2_rot0[:,1] < seg_up)[0]) ), :]
    
        if len(tmp) != 0:
            ncount += 1
            # Shift shift_z from the MC position, get the corrected cluster: cluster_lab_1_rotated_corrected
            shift = np.array([1]*len(tmp)).reshape(-1,1) * z[jj,ii]
            tmp_corrected = np.hstack( (tmp[:,0].reshape(-1,1), tmp[:,1].reshape(-1,1)+shift) )
            if ncount == 1:
                Energy_E1_E2_rot0_corrected = tmp_corrected
            else:
                Energy_E1_E2_rot0_corrected = np.vstack( (Energy_E1_E2_rot0_corrected, tmp_corrected) )
            
# plot
CSC.Scatter2D_plot(Energy_E1_E2_rot0_corrected[:,0].tolist(),Energy_E1_E2_rot0_corrected[:,1].tolist(), \
                   x_lim_left=-100, x_lim_right=100, y_lim_left=30, y_lim_right=100)
           
Energy_E1_E2_rot_corrected = np.vstack( (Energy_E1_E2_rot0_corrected, Energy_E1_E2_rot1) )
# plot
CSC.Scatter2D_plot(Energy_E1_E2_rot_corrected[:,0].tolist(),Energy_E1_E2_rot_corrected[:,1].tolist(), \
                   x_lim_left=-100, x_lim_right=100, y_lim_left=30, y_lim_right=100)


# rotate back
Energy_E1_E2_corrected = CSC_2pix.Corr_rot(basis_old=CSC_2pix.basis_new, basis_new=CSC_2pix.basis_old, \
                                           original_position=Energy_E1_E2_rot_corrected.T)

Energy_sum = np.sum( Energy_E1_E2, axis=1 ).reshape(-1,1)
Energy_sum_corrected = np.sum( Energy_E1_E2_corrected, axis=1 ).reshape(-1,1)

# compare the full range energy spectrum
CSC.Histogram_lineplot(Hist=Energy_sum, Bins=200, x_lim_low=20, x_lim_high=140, color='blue')
CSC.Histogram_lineplot(Hist=Energy_sum_corrected, Bins=200, x_lim_low=20, x_lim_high=140, color='red')

# save the corrected files
CSC.SaveFiles(var=Energy_E1_E2_corrected, dim=2, var_name='d', \
              location="C:\Jiajin\Mfile\Training_Sample_Analysis\FullRangeCorrected_2pix.csv")
