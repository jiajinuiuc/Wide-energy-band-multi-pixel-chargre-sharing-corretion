# -*- coding: utf-8 -*-
"""
Objective： 
Full range energy spectrum correction， try with 2-pixel charge sharing events
0. charge sharing events clustering on 60 keV(Am), 80.99 keV(Ba), 122 keV（Co）
1. full "spatial" (Charge sharing ratio) range segmentation and calculate the projection distance in each channel
3. calculate the projection distance of each charge sharing band at each channel
4. linear interpolation of the porjection distance between each band at different channel
5. based on the linear interpolation results, do the full range charge sharing correction
6. save data for the following step of matlab visualiztion
Version 0
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

%matplotlib qt5

################################################################################################################
################################### Load the 60-keV charge sharing events ######################################
CS_data = pd.read_csv( 'C:\Jiajin\ChaSha_2017\CSC_Data\CSC_extracted_data\Am_2Pix_new.csv' )
Energy = CS_data.iloc[:, :].values   #取第一行
CS_dim = Energy.shape[1]
del(CS_data)

Energy_sum = np.sum( Energy, axis=1 ).reshape(-1,1)
Energy_E1_E2_sum = Energy[ np.intersect1d(np.where(Energy_sum >= 50)[0],np.where(Energy_sum <= 70)[0]) ]
Energy_E1_E2_else = np.delete( Energy, np.intersect1d(np.where(Energy_sum >= 50)[0],np.where(Energy_sum <= 70)[0]), axis =0 )

##### plot the histogram of sum_Energy within the selected ROI
CF.Histogram_lineplot(Hist=Energy_sum, Bins=300, x_lim_high=100, x_lim_low=0, color='red')

##### plot the raw scatter figures within the selected ROI
CF.Scatter2D_plot(x=Energy_E1_E2_sum[:,0],y=Energy_E1_E2_sum[:,1],x_lim_left=0, x_lim_right=65, y_lim_left=0, y_lim_right=65)
########################################################################################################

##### Initialize the system basis and the segmentation size
basis_old =np.mat( [ [1,0],
                     [0,1] ] ) #[x, y]
basis_new = np.mat( [ [  1/np.sqrt(2),  1/np.sqrt(2) ],
                      [ -1/np.sqrt(2),  1/np.sqrt(2) ] ] )
seg_size = 0.01



####################### DBSCAN Clustering and Plot the results #########################################
Energy_sub = np.vstack( ( Energy_E1_E2_sum[:,0], Energy_E1_E2_sum[:,1] ) ).T
Energy_E1_E2_sum = np.sum( Energy_sub, axis=1 ).reshape(-1,1)

##### Model Fitting I        
# Set Fitting Parameters
y_hat1, y_unique1, core_indices1 = SC.Clustering(radius=1, density=30, data=Energy_sub)

##### Model Fitting II    
# Set Fitting Parameters
y_hat2, y_unique2, core_indices2 = SC.Clustering(radius=1, density=46, data=Energy_sub)

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
Energy_sum = np.sum( Energy, axis=1 ).reshape(-1,1)

y_hat = np.array([-1]*len(Energy))
y_hat[np.where( y_hat1 != -1 ),] = 0
y_hat[reduce( np.intersect1d, (np.where(y_hat2 == 2), np.where(Energy_sum >= 54)) ), ] = 1
y_hat[reduce( np.intersect1d, (np.where(y_hat2 == 3), np.where(Energy_sum >= 54)) ), ] = 2
y_hat[reduce( np.intersect1d, (np.where(y_hat == 0)[0],\
                               np.where(Energy_sum >= 55)[0],\
                               np.where(Energy[:,0] >= 20)[0],\
                               np.where(Energy[:,0] <= 40)[0]) ), ] = -1
    
y_unique = np.unique(y_hat)

cluster_60_lab_0 = Energy[np.where( ( y_hat == 0 ) )]
cluster_60_lab_1 = Energy[np.where( ( y_hat == 1 ) )]
cluster_60_lab_2 = Energy[np.where( ( y_hat == 2 ) )]
cluster_60_lab_noise = Energy[np.where( ( y_hat == -1 ) )]

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
        

#######################      CSC --- parameters measurement      #######################################
##### Initialize the CSC object
CSC_60_2pix = CSC( CS_dim=CS_dim, basis_old=basis_old, basis_new=basis_new, peak_energy=60, max_energy_range=140, seg_size=seg_size )
##### Calculate the MC-curve
wet_x_60, wet_w_60, shift_w_60, seg_unit_60, Energy_rotation = CSC_60_2pix.Pix2_Measurement( CS_data_labeled = Energy )
Energy_rotation = np.hstack( ( Energy_rotation[:,0], Energy_rotation[:,1] ) )
##### Calculate the rotated (Ratio, E-sum) points
wet_x_60, wet_w_60, shift_w_60, seg_unit_60, Energy_rot = CSC_60_2pix.Pix2_Measurement( CS_data_labeled = cluster_60_lab_0 )
##### Extend MC-curve to -1 and 1
left = min( min( np.where(wet_w_60 != 0) ) )
right = max( max( np.where(wet_w_60 != 0) ) )
wet_w_60[0:left+1,] = np.linspace(60, wet_w_60[left,], num=(left+1))
wet_w_60[right:len(wet_w_60),] = np.linspace(wet_w_60[right,], 60, num=(len(wet_w_60)-right))

left = min( min( np.where(shift_w_60!=0) ) )
right = max( max( np.where(shift_w_60!=0) ) )
shift_w_60[0:left+1,] = np.linspace(0, shift_w_60[left,], num=(left+1))
shift_w_60[right:len(shift_w_60),] = np.linspace(shift_w_60[right,], 0, num=(len(shift_w_60)-right))

##### Calculate the S-G filtered MC-curve
wet_w_60_filter = np.zeros(len(wet_w_60))
num = len(wet_w_60)//2 * 2 - 1
wet_w_60_filter = signal.savgol_filter(wet_w_60, num, 7)

shift_w_60_filter = np.zeros(len(shift_w_60))
num = len(shift_w_60)//2 * 2 - 1
shift_w_60_filter = signal.savgol_filter(shift_w_60, num, 7)

##### Check the scattering plot and MC plot
plt.figure(figsize=(12, 12), facecolor='w')
plt.grid('on')
for k in y_unique:
    cur = (y_hat == k)
    if k == -1:
        plt.scatter(Energy_rotation[cur,0].tolist(), Energy_rotation[cur,1].tolist(), s=2, c='k') # non-clustered points
        continue
    if k == 0:
        plt.scatter(Energy_rotation[cur,0].tolist(), Energy_rotation[cur,1].tolist(), s=7, c='b') # non-clustered points
        continue
    if k == 1:
        plt.scatter(Energy_rotation[cur,0].tolist(), Energy_rotation[cur,1].tolist(), s=7, c='r') # non-clustered points
        continue
    if k == 2:
        plt.scatter(Energy_rotation[cur,0].tolist(), Energy_rotation[cur,1].tolist(), s=7, c='y') # non-clustered points
        continue
plt.xlabel('Energy (keV)',fontsize=20)
plt.ylabel('Energy (keV)',fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlim(-1, 1)
plt.ylim(0, 70)
plt.grid(True)


CF.Line_plot(x=wet_x_60, y=wet_w_60, color='red', x_label='Spatial-axis', y_label='Energy-axis')
CF.Line_plot(x=wet_x_60, y=shift_w_60, color='blue', x_label='Spatial-axis', y_label='Energy-axis')

CF.Line_plot(x=wet_x_60, y=wet_w_60_filter, color='red', x_label='Spatial-axis', y_label='Energy-axis')
CF.Line_plot(x=wet_x_60, y=shift_w_60_filter, color='blue', x_label='Spatial-axis', y_label='Energy-axis')

##### fluorescence events correction
wet_x_60, wet_w_60_fluo1, shift_w_60_fluo1, seg_unit_60, Energy_rotation_fluo1 = CSC_60_2pix.Pix2_Measurement( CS_data_labeled = cluster_60_lab_1 )
wet_x_60, wet_w_60_fluo2, shift_w_60_fluo2, seg_unit_60, Energy_rotation_fluo2 = CSC_60_2pix.Pix2_Measurement( CS_data_labeled = cluster_60_lab_2 )
##### Calculate the S-G filtered MC-curve

wet_w_60_fluo1_filter = np.zeros(len(wet_w_60_fluo1))
left = min( min( np.where(wet_w_60_fluo1!=0) ) )
right = max( max( np.where(wet_w_60_fluo1!=0) ) )
num = (right-left)//2 * 2 + 1
wet_w_60_fluo1_filter[left:right+1,] = signal.savgol_filter(wet_w_60_fluo1[left:right+1,], num, 5)

wet_w_60_fluo2_filter = np.zeros(len(wet_w_60_fluo2))
left = min( min( np.where(wet_w_60_fluo2!=0) ) )
right = max( max( np.where(wet_w_60_fluo2!=0) ) )
num = (right-left)//2 * 2 + 1
wet_w_60_fluo2_filter[left:right+1,] = signal.savgol_filter(wet_w_60_fluo2[left:right+1,], num, 5)


##### Check fluo
CF.Line_plot(x=wet_x_60, y=wet_w_60_fluo1, color='red', x_label='Spatial-axis', y_label='Energy-axis')
CF.Line_plot(x=wet_x_60, y=wet_w_60_fluo2, color='blue', x_label='Spatial-axis', y_label='Energy-axis')

CF.Line_plot(x=wet_x_60, y=wet_w_60_fluo1_filter, color='red', x_label='Spatial-axis', y_label='Energy-axis')
CF.Line_plot(x=wet_x_60, y=wet_w_60_fluo2_filter, color='blue', x_label='Spatial-axis', y_label='Energy-axis')

###### Save data Figure 6 122-keV, 2-pixel, scattering + DBSCAN scattering
data_save = np.hstack( (Energy_rotation, y_hat.reshape(-1,1)) )
CF.SaveFiles(var=data_save, var_name=['Ratio', 'E', 'label'], \
              location="C:\Jiajin\ChaSha_2017\CSC_Data\CSC_extracted_data\Figure_data\Figure_678_Am_events_transformed_labeled.csv")

data_save = np.hstack( (wet_x_60.reshape(-1,1), wet_w_60_fluo1.reshape(-1,1), wet_w_60_fluo1_filter.reshape(-1,1), \
                        wet_w_60_fluo2.reshape(-1,1), wet_w_60_fluo2_filter.reshape(-1,1)) )
CF.SaveFiles(var=data_save, var_name=['X', 'wet_w_60_fluo1', 'wet_w_60_fluo1_filter', 'wet_w_60_fluo2', 'wet_w_60_fluo2_filter'], \
              location="C:\Jiajin\ChaSha_2017\CSC_Data\CSC_extracted_data\Figure_data\Figure_678_Am_fluo12_loss.csv")

data_save = np.hstack( (wet_x_60.reshape(-1,1), wet_w_60.reshape(-1,1), wet_w_60_filter.reshape(-1,1)) )
CF.SaveFiles(var=data_save, var_name=['X', 'loss', 'loss_SG'], \
              location="C:\Jiajin\ChaSha_2017\CSC_Data\CSC_extracted_data\Figure_data\Figure_678_Am_loss.csv")

data_save = np.hstack( (wet_x_60.reshape(-1,1), shift_w_60.reshape(-1,1), shift_w_60_filter.reshape(-1,1)) )
CF.SaveFiles(var=data_save, var_name=['X', 'comp', 'comp_SG'], \
              location="C:\Jiajin\ChaSha_2017\CSC_Data\CSC_extracted_data\Figure_data\Figure_678_Am_compensate.csv")

del(data_save) 
################################################################################################################





################################################################################################################
################################### Load the 81-keV charge sharing events ######################################
CS_data = pd.read_csv( 'C:\Jiajin\ChaSha_2017\CSC_Data\CSC_extracted_data\Ba_2Pix_new.csv' )
Energy = CS_data.iloc[:, :].values   #取第一行

CS_dim = Energy.shape[1]
del(CS_data)

Energy_sum = np.sum( Energy, axis=1 ).reshape(-1,1)
Energy_E1_E2_sum = Energy[ np.intersect1d(np.where(Energy_sum >= 60)[0],np.where(Energy_sum <= 90)[0]) ]
Energy_E1_E2_else = np.delete( Energy, np.intersect1d(np.where(Energy_sum >= 60)[0],np.where(Energy_sum <= 90)[0]), axis =0 )

##### plot the histogram of sum_Energy within the selected ROI
CF.Histogram_lineplot(Hist=Energy_sum, Bins=300, x_lim_high=100, x_lim_low=0, color='red')

##### plot the raw scatter figures within the selected ROI
CF.Scatter2D_plot(x=Energy_E1_E2_sum[:,0],y=Energy_E1_E2_sum[:,1],x_lim_left=0, x_lim_right=65, y_lim_left=0, y_lim_right=65)
########################################################################################################


####################### DBSCAN Clustering and Plot the results #########################################
Energy_sub = np.vstack( ( Energy_E1_E2_sum[:,0], Energy_E1_E2_sum[:,1] ) ).T
Energy_E1_E2_sum = np.sum( Energy_sub, axis=1 ).reshape(-1,1)

##### Model Fitting I        
# Set Fitting Parameters
y_hat1, y_unique1, core_indices1 = SC.Clustering(radius=1, density=15, data=Energy_sub)

##### Model Fitting II    
# Set Fitting Parameters
y_hat2, y_unique2, core_indices2 = SC.Clustering(radius=1, density=26, data=Energy_sub)
########################################################################################################


####################### Extract the cluster in the ROI #################################################
##### Reorganize the clustered scattering points
Energy = np.vstack( (Energy_sub, Energy_E1_E2_else) )
Energy_sum = np.sum( Energy, axis=1 ).reshape(-1,1)


y_hat = np.array([-1]*len(Energy))
y_hat[reduce( np.intersect1d, (np.where(y_hat1 != -1), np.where(y_hat1 != 4)) ),] = 0

y_hat[reduce( np.intersect1d, (np.where(y_hat2 == 2), np.where(Energy_sum >= 75.5)) ), ] = 1
y_hat[reduce( np.intersect1d, (np.where(y_hat2 == 3), np.where(Energy_sum >= 75.5)) ), ] = 2

y_hat[reduce( np.intersect1d, (np.where(y_hat == 0)[0],\
                               np.where(Energy_sum >= 75.5)[0],\
                               np.where(Energy[:,0] >= 40)[0],\
                               np.where(Energy[:,0] <= 60)[0]) ), ] = -1

y_hat[reduce( np.intersect1d, (np.where(y_hat == 0)[0],\
                               np.where(Energy_sum >= 75.5)[0],\
                               np.where(Energy[:,0] >= 20)[0],\
                               np.where(Energy[:,0] <= 40)[0]) ), ] = -1
y_unique = np.unique(y_hat)


cluster_81_lab_0 = Energy[np.where( ( y_hat == 0 ) )]
cluster_81_lab_1 = Energy[np.where( ( y_hat == 1 ) )]
cluster_81_lab_2 = Energy[np.where( ( y_hat == 2 ) )]
cluster_81_lab_noise = Energy[np.where( ( y_hat == -1 ) )]

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
plt.xlim(0, 90)
plt.ylim(0, 90)
plt.grid(True)

###### Save data Figure 5 122-keV, 2-pixel, scattering + DBSCAN scattering
data_save = np.hstack( (Energy, y_hat.reshape(-1,1)) ) ### ( E1, E2, label)
CF.SaveFiles(var=data_save, var_name=['E1', 'E2', 'label'], \
              location="C:\Jiajin\ChaSha_2017\CSC_Data\CSC_extracted_data\Figure_data\Figure_9_Ba_DBSCAN.csv")
del(data_save)
######################################################################################################## 


####################### "Rot -> MC Shifting -> Rot" CSC function #######################################
##### Initialize the CSC object
CSC_81_2pix = CSC( CS_dim=CS_dim, basis_old=basis_old, basis_new=basis_new, peak_energy=81, max_energy_range=140, seg_size=seg_size )
##### Calculate the MC-curve
wet_x_81, wet_w_81, shift_w_81, seg_unit_81, Energy_rotation = CSC_81_2pix.Pix2_Measurement( CS_data_labeled = Energy )
Energy_rotation = np.hstack( ( Energy_rotation[:,0], Energy_rotation[:,1] ) )
##### Calculate the rotated (Ratio, E-sum) points
wet_x_81, wet_w_81, shift_w_81, seg_unit_81, Energy_rot = CSC_81_2pix.Pix2_Measurement( CS_data_labeled = cluster_81_lab_0 )
##### Extend MC-curve to -1 and 1
left = min( min( np.where(wet_w_81 != 0) ) )
right = max( max( np.where(wet_w_81 != 0) ) )
wet_w_81[0:left+1,] = np.linspace(81, wet_w_81[left,], num=(left+1))
wet_w_81[right:len(wet_w_81),] = np.linspace(wet_w_81[right,], 81, num=(len(wet_w_81)-right))

left = min( min( np.where(shift_w_81!=0) ) )
right = max( max( np.where(shift_w_81!=0) ) )
shift_w_81[0:left+1,] = np.linspace(0, shift_w_81[left,], num=(left+1))
shift_w_81[right:len(shift_w_81),] = np.linspace(shift_w_81[right,], 0, num=(len(shift_w_81)-right))

##### Calculate the S-G filtered MC-curve
wet_w_81_filter = np.zeros(len(wet_w_81))
num = len(wet_w_81)//2 * 2 - 1
wet_w_81_filter = signal.savgol_filter(wet_w_81, num, 7)

shift_w_81_filter = np.zeros(len(shift_w_81))
num = len(shift_w_81)//2 * 2 - 1
shift_w_81_filter = signal.savgol_filter(shift_w_81, num, 7)
wet_w_81_filter = 81 - shift_w_81_filter
##### Check the scattering plot and MC plot
plt.figure(figsize=(12, 12), facecolor='w')
plt.grid('on')
for k in y_unique:
    cur = (y_hat == k)
    if k == -1:
        plt.scatter(Energy_rotation[cur,0].tolist(), Energy_rotation[cur,1].tolist(), s=2, c='k') # non-clustered points
        continue
    if k == 0:
        plt.scatter(Energy_rotation[cur,0].tolist(), Energy_rotation[cur,1].tolist(), s=7, c='b') # non-clustered points
        continue
    if k == 1:
        plt.scatter(Energy_rotation[cur,0].tolist(), Energy_rotation[cur,1].tolist(), s=7, c='r') # non-clustered points
        continue
    if k == 2:
        plt.scatter(Energy_rotation[cur,0].tolist(), Energy_rotation[cur,1].tolist(), s=7, c='y') # non-clustered points
        continue
plt.xlabel('Energy (keV)',fontsize=20)
plt.ylabel('Energy (keV)',fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlim(-1, 1)
plt.ylim(0, 90)
plt.grid(True)

CF.Scatter2D_plot(x=Energy_rotation[:,0].tolist(), y=Energy_rotation[:,1].tolist(), x_lim_left=-1, \
                  x_lim_right=1, y_lim_left=0, y_lim_right=65)

CF.Line_plot(x=wet_x_81, y=wet_w_81, color='red', x_label='Spatial-axis', y_label='Energy-axis')
CF.Line_plot(x=wet_x_81, y=shift_w_81, color='blue', x_label='Spatial-axis', y_label='Energy-axis')

CF.Line_plot(x=wet_x_81, y=wet_w_81_filter, color='red', x_label='Spatial-axis', y_label='Energy-axis')
CF.Line_plot(x=wet_x_81, y=shift_w_81_filter, color='blue', x_label='Spatial-axis', y_label='Energy-axis')

##### fluorescence events correction
wet_x_81, wet_w_81_fluo1, shift_w_81_fluo1, seg_unit_81, Energy_rotation = CSC_81_2pix.Pix2_Measurement( CS_data_labeled = cluster_81_lab_1 )
wet_x_81, wet_w_81_fluo2, shift_w_81_fluo2, seg_unit_81, Energy_rotation = CSC_81_2pix.Pix2_Measurement( CS_data_labeled = cluster_81_lab_2 )
##### Calculate the S-G filtered MC-curve

wet_w_81_fluo1_filter = np.zeros(len(wet_w_81_fluo1))
left = min( min( np.where(wet_w_81_fluo1!=0) ) )
right = max( max( np.where(wet_w_81_fluo1!=0) ) )
num = (right-left)//2 * 2 + 1
wet_w_81_fluo1_filter[left:right+1,] = signal.savgol_filter(wet_w_81_fluo1[left:right+1,], num, 5)

wet_w_81_fluo2_filter = np.zeros(len(wet_w_81_fluo2))
left = min( min( np.where(wet_w_81_fluo2!=0) ) )
right = max( max( np.where(wet_w_81_fluo2!=0) ) )
num = (right-left)//2 * 2 + 1
wet_w_81_fluo2_filter[left:right+1,] = signal.savgol_filter(wet_w_81_fluo2[left:right+1,], num, 5)


##### Check fluo
CF.Line_plot(x=wet_x_81, y=wet_w_81_fluo1, color='red', x_label='Spatial-axis', y_label='Energy-axis')
CF.Line_plot(x=wet_x_81, y=wet_w_81_fluo2, color='blue', x_label='Spatial-axis', y_label='Energy-axis')

CF.Line_plot(x=wet_x_81, y=wet_w_81_fluo1_filter, color='red', x_label='Spatial-axis', y_label='Energy-axis')
CF.Line_plot(x=wet_x_81, y=wet_w_81_fluo2_filter, color='blue', x_label='Spatial-axis', y_label='Energy-axis')

###### Save data Figure 6 122-keV, 2-pixel, scattering + DBSCAN scattering
data_save = np.hstack( (Energy_rotation, y_hat.reshape(-1,1)) )
CF.SaveFiles(var=data_save, var_name=['Ratio', 'E', 'label'], \
              location="C:\Jiajin\ChaSha_2017\CSC_Data\CSC_extracted_data\Figure_data\Figure_9_Ba_events_transformed_labeled.csv")

data_save = np.hstack( (wet_x_81.reshape(-1,1), wet_w_81_fluo1.reshape(-1,1), wet_w_81_fluo1_filter.reshape(-1,1), \
                        wet_w_81_fluo2.reshape(-1,1), wet_w_81_fluo2_filter.reshape(-1,1)) )
CF.SaveFiles(var=data_save, var_name=['X', 'wet_w_81_fluo1', 'wet_w_81_fluo1_filter', 'wet_w_81_fluo2', 'wet_w_81_fluo2_filter'], \
              location="C:\Jiajin\ChaSha_2017\CSC_Data\CSC_extracted_data\Figure_data\Figure_9_Ba_fluo12_loss.csv")

data_save = np.hstack( (wet_x_81.reshape(-1,1), wet_w_81.reshape(-1,1), wet_w_81_filter.reshape(-1,1)) )
CF.SaveFiles(var=data_save, var_name=['X', 'loss', 'loss_SG'], \
              location="C:\Jiajin\ChaSha_2017\CSC_Data\CSC_extracted_data\Figure_data\Figure_9_Ba_loss.csv")

data_save = np.hstack( (wet_x_81.reshape(-1,1), shift_w_81.reshape(-1,1), shift_w_81_filter.reshape(-1,1)) )
CF.SaveFiles(var=data_save, var_name=['X', 'comp', 'comp_SG'], \
              location="C:\Jiajin\ChaSha_2017\CSC_Data\CSC_extracted_data\Figure_data\Figure_9_Ba_compensate.csv")

del(data_save) 
################################################################################################################





################################################################################################################
################################### Load the 122-keV charge sharing events ######################################
CS_data = pd.read_csv( 'C:\Jiajin\ChaSha_2017\CSC_Data\CSC_extracted_data\Co_2Pix_new.csv' )
Energy = CS_data.iloc[:, :].values   #取第一行
CS_dim = Energy.shape[1]
del(CS_data)

Energy_sum = np.sum( Energy, axis=1 ).reshape(-1,1)
Energy_E1_E2_sum = Energy[ np.intersect1d(np.where(Energy_sum >= 100)[0],np.where(Energy_sum <= 130)[0]) ]
Energy_E1_E2_else = np.delete( Energy, np.intersect1d(np.where(Energy_sum >= 100)[0],np.where(Energy_sum <= 130)[0]), axis =0 )

##### plot the histogram of sum_Energy within the selected ROI
CF.Histogram_lineplot(Hist=Energy_sum, Bins=300, x_lim_high=100, x_lim_low=0, color='red')

##### plot the raw scatter figures within the selected ROI
CF.Scatter2D_plot(x=Energy_E1_E2_sum[:,0],y=Energy_E1_E2_sum[:,1],x_lim_left=0, x_lim_right=65, y_lim_left=0, y_lim_right=65)
########################################################################################################


####################### DBSCAN Clustering and Plot the results #########################################
Energy_sub = np.vstack( ( Energy_E1_E2_sum[:,0], Energy_E1_E2_sum[:,1] ) ).T
Energy_E1_E2_sum = np.sum( Energy_sub, axis=1 ).reshape(-1,1)

##### Model Fitting I        
# Set Fitting Parameters
y_hat1, y_unique1, core_indices1 = SC.Clustering(radius=2, density=14, data=Energy_sub)

##### Model Fitting II    
# Set Fitting Parameters
y_hat2, y_unique2, core_indices2 = SC.Clustering(radius=1, density=16, data=Energy_sub)
########################################################################################################


####################### Extract the cluster in the ROI #################################################
##### Reorganize the clustered scattering points
Energy = np.vstack( (Energy_sub, Energy_E1_E2_else) )
Energy_sum = np.sum( Energy, axis=1 ).reshape(-1,1)


y_hat = np.array([-1]*len(Energy))
y_hat[reduce( np.intersect1d, (np.where(y_hat1 != -1), np.where(y_hat1 != 4)) ),] = 0

y_hat[reduce( np.intersect1d, (np.where(y_hat2 == 3), np.where(Energy_sum >= 75.5)) ), ] = 1
y_hat[reduce( np.intersect1d, (np.where(y_hat2 == 4), np.where(Energy_sum >= 75.5)) ), ] = 2

y_hat[reduce( np.intersect1d, (np.where(y_hat == 0)[0],\
                               np.where(Energy_sum >= 114)[0],\
                               np.where(Energy[:,0] >= 80)[0],\
                               np.where(Energy[:,0] <= 100)[0]) ), ] = 2

y_hat[reduce( np.intersect1d, (np.where(y_hat == 0)[0],\
                               np.where(Energy_sum >= 116)[0],\
                               np.where(Energy[:,0] >= 100)[0],\
                               np.where(Energy[:,0] <= 104)[0]) ), ] = 2
    
y_hat[reduce( np.intersect1d, (np.where(y_hat == 0)[0],\
                               np.where(Energy_sum >= 116)[0],\
                               np.where(Energy[:,0] >= 18)[0],\
                               np.where(Energy[:,0] <= 30)[0]) ), ] = 1
    
y_hat[reduce( np.intersect1d, (np.where(y_hat == 0)[0],\
                               np.where(Energy_sum >= 114)[0],\
                               np.where(Energy[:,0] >= 30)[0],\
                               np.where(Energy[:,0] <= 40)[0]) ), ] = 1

y_hat[reduce( np.intersect1d, (np.where(y_hat == 0)[0],\
                               np.where(Energy_sum >= 122)[0],\
                               np.where(Energy[:,0] >= 110)[0]) ), ] = -1

y_hat[reduce( np.intersect1d, (np.where(y_hat == 0)[0],\
                               np.where(Energy_sum <= 110)[0],\
                               np.where(Energy[:,0] >= 0)[0],\
                               np.where(Energy[:,0] <= 10)[0]) ), ] = -1
    
y_hat[reduce( np.intersect1d, (np.where(y_hat == 0)[0],\
                               np.where(Energy_sum <= 110)[0],\
                               np.where(Energy[:,0] >= 100)[0],\
                               np.where(Energy[:,0] <= 110)[0]) ), ] = -1

y_hat[reduce( np.intersect1d, (np.where(y_hat == 0)[0],\
                               np.where(Energy_sum <= 108)[0],\
                               np.where(Energy[:,0] >= 90)[0],\
                               np.where(Energy[:,0] <= 100)[0]) ), ] = -1   
y_unique = np.unique(y_hat)


cluster_122_lab_0 = Energy[np.where( ( y_hat == 0 ) )]
cluster_122_lab_1 = Energy[np.where( ( y_hat == 1 ) )]
cluster_122_lab_2 = Energy[np.where( ( y_hat == 2 ) )]
cluster_122_lab_noise = Energy[np.where( ( y_hat == -1 ) )]

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
plt.xlim(0, 140)
plt.ylim(0, 140)
plt.grid(True)

###### Save data Figure 5 122-keV, 2-pixel, scattering + DBSCAN scattering
data_save = np.hstack( (Energy, y_hat.reshape(-1,1)) ) ### ( E1, E2, label)
CF.SaveFiles(var=data_save, var_name=['E1', 'E2', 'label'], \
              location="C:\Jiajin\ChaSha_2017\CSC_Data\CSC_extracted_data\Figure_data\Figure_9_Co_DBSCAN.csv")
del(data_save)
######################################################################################################## 




####################### "Rot -> MC Shifting -> Rot" CSC function #######################################
##### Initialize the CSC object
CSC_122_2pix = CSC( CS_dim=CS_dim, basis_old=basis_old, basis_new=basis_new, peak_energy=122, max_energy_range=140, seg_size=seg_size )
##### Calculate the rotated (Ratio, E-sum) points 
wet_x_122, wet_w_122, shift_w_122, seg_unit_122, Energy_rotation = CSC_122_2pix.Pix2_Measurement( CS_data_labeled = Energy )
Energy_rotation = np.hstack( ( Energy_rotation[:,0], Energy_rotation[:,1] ) )
##### Calculate the MC-curve
wet_x_122, wet_w_122, shift_w_122, seg_unit_122, Energy_rot = CSC_122_2pix.Pix2_Measurement( CS_data_labeled = cluster_122_lab_0 )
##### Extend MC-curve to -1 and 1
left = min( min( np.where(wet_w_122 != 0) ) )
right = max( max( np.where(wet_w_122 != 0) ) )
wet_w_122[0:left+1,] = np.linspace(122, wet_w_122[left,], num=(left+1))
wet_w_122[right:len(wet_w_122),] = np.linspace(wet_w_122[right,], 122, num=(len(wet_w_122)-right))

left = min( min( np.where(shift_w_122!=0) ) )
right = max( max( np.where(shift_w_122!=0) ) )
shift_w_122[0:left+1,] = np.linspace(0, shift_w_122[left,], num=(left+1))
shift_w_122[right:len(shift_w_122),] = np.linspace(shift_w_122[right,], 0, num=(len(shift_w_122)-right))

##### Calculate the S-G filtered MC-curve
wet_w_122_filter = np.zeros(len(wet_w_122))
num = len(wet_w_122)//2 * 2 - 1
wet_w_122_filter = signal.savgol_filter(wet_w_122, num, 7)

shift_w_122_filter = np.zeros(len(shift_w_122))
num = len(shift_w_122)//2 * 2 - 1
shift_w_122_filter = signal.savgol_filter(shift_w_122, num, 7)

##### Check the scattering plot and MC plot
plt.figure(figsize=(12, 12), facecolor='w')
plt.grid('on')
for k in y_unique:
    cur = (y_hat == k)
    if k == -1:
        plt.scatter(Energy_rotation[cur,0].tolist(), Energy_rotation[cur,1].tolist(), s=2, c='k') # non-clustered points
        continue
    if k == 0:
        plt.scatter(Energy_rotation[cur,0].tolist(), Energy_rotation[cur,1].tolist(), s=7, c='b') # non-clustered points
        continue
    if k == 1:
        plt.scatter(Energy_rotation[cur,0].tolist(), Energy_rotation[cur,1].tolist(), s=7, c='r') # non-clustered points
        continue
    if k == 2:
        plt.scatter(Energy_rotation[cur,0].tolist(), Energy_rotation[cur,1].tolist(), s=7, c='y') # non-clustered points
        continue
plt.xlabel('Energy (keV)',fontsize=20)
plt.ylabel('Energy (keV)',fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlim(-1, 1)
plt.ylim(0, 140)
plt.grid(True)

CF.Line_plot(x=wet_x_122, y=wet_w_122, color='red', x_label='Spatial-axis', y_label='Energy-axis')
CF.Line_plot(x=wet_x_122, y=shift_w_122, color='blue', x_label='Spatial-axis', y_label='Energy-axis')

CF.Line_plot(x=wet_x_122, y=wet_w_122_filter, color='red', x_label='Spatial-axis', y_label='Energy-axis')
CF.Line_plot(x=wet_x_122, y=shift_w_122_filter, color='blue', x_label='Spatial-axis', y_label='Energy-axis')

##### fluorescence events correction
wet_x_122, wet_w_122_fluo1, shift_w_122_fluo1, seg_unit_122, Energy_rotation_fluo1 = CSC_122_2pix.Pix2_Measurement( CS_data_labeled = cluster_81_lab_1 )
wet_x_122, wet_w_122_fluo2, shift_w_122_fluo2, seg_unit_122, Energy_rotation_fluo2 = CSC_122_2pix.Pix2_Measurement( CS_data_labeled = cluster_81_lab_2 )
##### Calculate the S-G filtered MC-curve

wet_w_122_fluo1_filter = np.zeros(len(wet_w_122_fluo1))
left = min( min( np.where(wet_w_122_fluo1!=0) ) )
right = max( max( np.where(wet_w_122_fluo1!=0) ) )
num = (right-left)//2 * 2 + 1
wet_w_122_fluo1_filter[left:right+1,] = signal.savgol_filter(wet_w_122_fluo1[left:right+1,], num, 7)

wet_w_122_fluo2_filter = np.zeros(len(wet_w_122_fluo2))
left = min( min( np.where(wet_w_122_fluo2!=0) ) )
right = max( max( np.where(wet_w_122_fluo2!=0) ) )
num = (right-left)//2 * 2 + 1
wet_w_122_fluo2_filter[left:right+1,] = signal.savgol_filter(wet_w_122_fluo2[left:right+1,], num, 7)


##### Check fluo
CF.Line_plot(x=wet_x_122, y=wet_w_122_fluo1, color='red', x_label='Spatial-axis', y_label='Energy-axis')
CF.Line_plot(x=wet_x_122, y=wet_w_122_fluo2, color='blue', x_label='Spatial-axis', y_label='Energy-axis')

CF.Line_plot(x=wet_x_122, y=wet_w_122_fluo1_filter, color='red', x_label='Spatial-axis', y_label='Energy-axis')
CF.Line_plot(x=wet_x_122, y=wet_w_122_fluo2_filter, color='blue', x_label='Spatial-axis', y_label='Energy-axis')

###### Save data Figure 6 122-keV, 2-pixel, scattering + DBSCAN scattering
data_save = np.hstack( (Energy_rotation, y_hat.reshape(-1,1)) )
CF.SaveFiles(var=data_save, var_name=['Ratio', 'E', 'label'], \
              location="C:\Jiajin\ChaSha_2017\CSC_Data\CSC_extracted_data\Figure_data\Figure_9_Co_events_transformed_labeled.csv")

data_save = np.hstack( (wet_x_122.reshape(-1,1), wet_w_122_fluo1.reshape(-1,1), wet_w_122_fluo1_filter.reshape(-1,1), \
                        wet_w_122_fluo2.reshape(-1,1), wet_w_122_fluo2_filter.reshape(-1,1)) )
CF.SaveFiles(var=data_save, var_name=['X', 'wet_w_122_fluo1', 'wet_w_122_fluo1_filter', 'wet_w_122_fluo2', 'wet_w_122_fluo2_filter'], \
              location="C:\Jiajin\ChaSha_2017\CSC_Data\CSC_extracted_data\Figure_data\Figure_9_Co_fluo12_loss.csv")

data_save = np.hstack( (wet_x_122.reshape(-1,1), wet_w_122.reshape(-1,1), wet_w_122_filter.reshape(-1,1)) )
CF.SaveFiles(var=data_save, var_name=['X', 'loss', 'loss_SG'], \
              location="C:\Jiajin\ChaSha_2017\CSC_Data\CSC_extracted_data\Figure_data\Figure_9_Co_loss.csv")

data_save = np.hstack( (wet_x_122.reshape(-1,1), shift_w_122.reshape(-1,1), shift_w_122_filter.reshape(-1,1)) )
CF.SaveFiles(var=data_save, var_name=['X', 'comp', 'comp_SG'], \
              location="C:\Jiajin\ChaSha_2017\CSC_Data\CSC_extracted_data\Figure_data\Figure_9_Co_compensate.csv")

del(data_save) 
################################################################################################################




################################################################################################################
###################################    Compensation mapping calculation    #####################################
##### Three Calibration line visulization
CF.Line_plot(x=wet_x_60, y=shift_w_60_filter, color='red', x_label='Ratio = (E1 - E2)/(E1 + E2)', y_label='Energy compensation (keV)')
CF.Line_plot(x=wet_x_60, y=shift_w_81_filter, color='blue', x_label='Ratio = (E1 - E2)/(E1 + E2)', y_label='Energy compensation (keV)')
CF.Line_plot(x=wet_x_60, y=shift_w_122_filter, color='green', x_label='Ratio = (E1 - E2)/(E1 + E2)', y_label='Energy compensation (keV)')

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
CSC2_60_2Pix = CSC2( CS_dim=CS_dim, basis_old=basis_old, basis_new=basis_new, peak_energy=60, max_energy_range=140, seg_size=2 )
wet_x, wet_w, shift_w, seg_unit = CSC2_60_2Pix.Pix2_Measurement( CS_data_labeled=cluster_60_lab_0 )
cluster_60_lab_0_corrected = CSC2_60_2Pix.Pix2_Correction( seg_unit=seg_unit, shift_w=shift_w, CS_data_labeled=cluster_60_lab_0 )
cluster_60_lab_noise_corrected = CSC2_60_2Pix.Pix2_Correction( seg_unit=seg_unit, shift_w=shift_w, CS_data_labeled=cluster_60_lab_noise )

wet_x, wet_w, shift_w, seg_unit = CSC2_60_2Pix.Pix2_Measurement( CS_data_labeled=cluster_60_lab_1 )
cluster_60_lab_1_corrected = CSC2_60_2Pix.Pix2_Correction( seg_unit=seg_unit, shift_w=shift_w, CS_data_labeled=cluster_60_lab_1 )

wet_x, wet_w, shift_w, seg_unit = CSC2_60_2Pix.Pix2_Measurement( CS_data_labeled=cluster_60_lab_2 )
cluster_60_lab_2_corrected = CSC2_60_2Pix.Pix2_Correction( seg_unit=seg_unit, shift_w=shift_w, CS_data_labeled=cluster_60_lab_2 )

### 81
CSC2_81_2Pix = CSC2(CS_dim=CS_dim, basis_old=basis_old, basis_new=basis_new, peak_energy=81, max_energy_range=140, seg_size=2)
wet_x, wet_w, shift_w, seg_unit = CSC2_81_2Pix.Pix2_Measurement( CS_data_labeled=cluster_81_lab_0 )
cluster_81_lab_0_corrected = CSC2_81_2Pix.Pix2_Correction( seg_unit=seg_unit, shift_w=shift_w, CS_data_labeled=cluster_81_lab_0 )
cluster_81_lab_noise_corrected = CSC2_81_2Pix.Pix2_Correction( seg_unit=seg_unit, shift_w=shift_w, CS_data_labeled=cluster_81_lab_noise )

wet_x, wet_w, shift_w, seg_unit = CSC2_81_2Pix.Pix2_Measurement( CS_data_labeled=cluster_81_lab_1 )
cluster_81_lab_1_corrected = CSC2_81_2Pix.Pix2_Correction( seg_unit=seg_unit, shift_w=shift_w, CS_data_labeled=cluster_81_lab_1 )

wet_x, wet_w, shift_w, seg_unit = CSC2_81_2Pix.Pix2_Measurement( CS_data_labeled=cluster_81_lab_2 )
cluster_81_lab_2_corrected = CSC2_81_2Pix.Pix2_Correction( seg_unit=seg_unit, shift_w=shift_w, CS_data_labeled=cluster_81_lab_2 )

### 122
CSC2_122_2Pix = CSC2(CS_dim=CS_dim, basis_old=basis_old, basis_new=basis_new, peak_energy=122, max_energy_range=140, seg_size=2)
wet_x, wet_w, shift_w, seg_unit = CSC2_122_2Pix.Pix2_Measurement( CS_data_labeled=cluster_122_lab_0 )
cluster_122_lab_0_corrected = CSC2_122_2Pix.Pix2_Correction( seg_unit=seg_unit, shift_w=shift_w, CS_data_labeled=cluster_122_lab_0 )
cluster_122_lab_noise_corrected = CSC2_122_2Pix.Pix2_Correction( seg_unit=seg_unit, shift_w=shift_w, CS_data_labeled=cluster_122_lab_noise )

wet_x, wet_w, shift_w, seg_unit = CSC2_122_2Pix.Pix2_Measurement( CS_data_labeled=cluster_122_lab_1 )
cluster_122_lab_1_corrected = CSC2_122_2Pix.Pix2_Correction( seg_unit=seg_unit, shift_w=shift_w, CS_data_labeled=cluster_122_lab_1 )

wet_x, wet_w, shift_w, seg_unit = CSC2_122_2Pix.Pix2_Measurement( CS_data_labeled=cluster_122_lab_2 )
cluster_122_lab_2_corrected = CSC2_122_2Pix.Pix2_Correction( seg_unit=seg_unit, shift_w=shift_w, CS_data_labeled=cluster_122_lab_2 )

### noise
#noise = np.vstack( (cluster_60_lab_noise, cluster_81_lab_noise, cluster_122_lab_noise) )
#CSC_122_2pix = CSC( CS_dim=CS_dim, basis_old=basis_old, basis_new=basis_new, peak_energy=122, max_energy_range=140, seg_size=seg_size )
#noise_corrected = CSC_122_2pix.Pix2_Correction( CS_data_labeled=noise, correction_map=Map, R=R, E=E )


data = np.vstack( (cluster_60_lab_0, cluster_60_lab_1, cluster_60_lab_2, cluster_60_lab_noise,\
                   cluster_81_lab_0, cluster_81_lab_1, cluster_81_lab_2, cluster_81_lab_noise,\
                   cluster_122_lab_0, cluster_122_lab_1, cluster_122_lab_2, cluster_122_lab_noise) )

data_corrected = np.vstack( (cluster_60_lab_0_corrected, cluster_60_lab_1_corrected, cluster_60_lab_2_corrected, cluster_60_lab_noise_corrected, \
                             cluster_81_lab_0_corrected, cluster_81_lab_1_corrected, cluster_81_lab_2_corrected, cluster_81_lab_noise_corrected, \
                             cluster_122_lab_0_corrected, cluster_122_lab_1_corrected, cluster_122_lab_2_corrected, cluster_122_lab_noise_corrected) )


Energy_sum = np.sum(data, axis=1).reshape(-1,1)
CF.Histogram_lineplot(Hist=Energy_sum, Bins=300, x_lim_low=0, x_lim_high=200, color='blue')

Energy_sum_corrected = np.sum(data_corrected, axis=1).reshape(-1,1)
CF.Histogram_lineplot(Hist=Energy_sum_corrected, Bins=300, x_lim_low=0, x_lim_high=200, color='red')
plt.ylim(0, 1000)

### save the data and corrected data
CF.SaveFiles(var=data, var_name=['E1', 'E2'], location='C:\Jiajin\ChaSha_2017\CSC_Data\CSC_extracted_data\Figure_data\Figure_12_raw.csv')
CF.SaveFiles(var=data_corrected, var_name=['E1', 'E2'], location='C:\Jiajin\ChaSha_2017\CSC_Data\CSC_extracted_data\Figure_data\Figure_12_corrected.csv')

### save separately
# raw
data = np.vstack( ( cluster_60_lab_0, cluster_60_lab_1, cluster_60_lab_2, cluster_60_lab_noise ) )
CF.SaveFiles(var=data, var_name=['E1', 'E2'], location='C:\Jiajin\ChaSha_2017\CSC_Data\CSC_extracted_data\spectra_data\Am_2pix_raw.csv')

data = np.vstack( ( cluster_81_lab_0, cluster_81_lab_1, cluster_81_lab_2, cluster_81_lab_noise ) )
CF.SaveFiles(var=data, var_name=['E1', 'E2'], location='C:\Jiajin\ChaSha_2017\CSC_Data\CSC_extracted_data\spectra_data\Ba_2pix_raw.csv')

data = np.vstack( ( cluster_122_lab_0, cluster_122_lab_1, cluster_122_lab_2, cluster_122_lab_noise ) )
CF.SaveFiles(var=data, var_name=['E1', 'E2'], location='C:\Jiajin\ChaSha_2017\CSC_Data\CSC_extracted_data\spectra_data\Co_2pix_raw.csv')


# corrected
data = np.vstack( ( cluster_60_lab_0_corrected, cluster_60_lab_1_corrected, cluster_60_lab_2_corrected, cluster_60_lab_noise_corrected ) )
CF.SaveFiles(var=data, var_name=['E1', 'E2'], location='C:\Jiajin\ChaSha_2017\CSC_Data\CSC_extracted_data\spectra_data\Am_2pix_corrected.csv')

data = np.vstack( ( cluster_81_lab_0_corrected, cluster_81_lab_1_corrected, cluster_81_lab_2_corrected, cluster_81_lab_noise_corrected ) )
CF.SaveFiles(var=data, var_name=['E1', 'E2'], location='C:\Jiajin\ChaSha_2017\CSC_Data\CSC_extracted_data\spectra_data\Ba_2pix_corrected.csv')

data = np.vstack( ( cluster_122_lab_0_corrected, cluster_122_lab_1_corrected, cluster_122_lab_2_corrected, cluster_122_lab_noise_corrected ) )
CF.SaveFiles(var=data, var_name=['E1', 'E2'], location='C:\Jiajin\ChaSha_2017\CSC_Data\CSC_extracted_data\spectra_data\Co_2pix_corrected.csv')





