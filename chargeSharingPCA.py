# -*- coding: utf-8 -*-
"""
Objective: principle component analysis on the charge sharing vectors
before correction & after correction
@author: J. J. Zhang
Last update: May, 2019
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.datasets as ds
import matplotlib.colors
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from functools import reduce

%matplotlib qt5

######################################### 2-pixel PCA ###################################################
##### Load the charge sharing events ####################################################################
CS_data = pd.read_csv( 'C:\Jiajin\Mfile\Training_Sample_Analysis\Pix2Sharing.csv' )
# CS_data = pd.read_csv( 'C:\Jiajin\Mfile\Training_Sample_Analysis\Pix2Sharing_corr.csv' )
Energy = CS_data.iloc[:, :].values   #取第一行
Energy_sum = ( Energy[:,0] + Energy[:,1] ).reshape(-1,1)

Energy_E1_E2_sum = Energy[ np.intersect1d(np.where(Energy_sum >= 105)[0],np.where(Energy_sum <= 140)[0]) ]
Energy_sum = Energy_sum[ np.intersect1d(np.where(Energy_sum >= 105)[0],np.where(Energy_sum <= 140)[0]) ]
Energy_E1_E2_sum = np.hstack( (Energy_E1_E2_sum, Energy_sum) )

##### plot the histogram of sum_Energy within the selected ROI
plt.figure(figsize=(12, 12), facecolor='w')
plt.grid('on')
plt.suptitle('Histogram of the sum of 2-pixel chargesharing events', fontsize = 24, fontweight = 'bold')
plt.hist(Energy_E1_E2_sum[:,2], bins=100, normed=0, facecolor="blue", edgecolor="black", alpha=0.7)
plt.xlim(50, 180)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Energy (keV)',fontsize=20)
plt.ylabel('Counts per bin size',fontsize=20)

##### plot the raw scatter figures within the selected ROI
plt.figure(figsize=(12, 12), facecolor='w')
plt.suptitle(u'Charge Sharing Raw Scattering Data', fontsize=20)
plt.scatter(Energy_E1_E2_sum[:,0],Energy_E1_E2_sum[:,1], s=10, c='k',marker='.')
plt.xlabel('Energy (keV)',fontsize=20)
plt.ylabel('Energy (keV)',fontsize=20)
plt.xlim(0, 140)
plt.ylim(0, 140)
########################################################################################################

##### PCA #####
pca = PCA(n_components=2)
Energy_new = pca.fit_transform(Energy_E1_E2_sum[:,0:2])
print( pca.explained_variance_ratio_ )
print( pca.explained_variance_ )

fig = plt.figure(110)
plt.scatter(Energy_new[:, 0], Energy_new[:, 1], s=10, c='k', marker='.')
plt.xlabel(' "Energy" along principle axis ',fontsize=20)
plt.ylabel(' "Energy" along non-principle axis',fontsize=20)

fig2 = plt.figure(111)
plt.suptitle('Histogram of Scattering Points along Principle Component Axis', fontsize = 24, fontweight = 'bold')
plt.hist(Energy_new[:, 0], bins=400, normed=0, facecolor="blue", edgecolor="black", alpha=0.7)
plt.xlabel(' "Position" ',fontsize=20)
plt.ylabel(' Counts ',fontsize=20)


fig3 = plt.figure(112)
plt.suptitle('Histogram of Scattering Points along Non-Principle Component Axis', fontsize = 24, fontweight = 'bold')
plt.hist(Energy_new[:, 1], bins=100, normed=0, facecolor="blue", edgecolor="black", alpha=0.7)
plt.xlim(-40, 40)
plt.xlabel(' "Energy" ',fontsize=20)
plt.ylabel(' Counts ',fontsize=20)
########################################################################################################




######################################### 3-pixel PCA ###################################################
##### Load the charge sharing events ####################################################################
CS_data = pd.read_csv( 'C:\Jiajin\Mfile\Training_Sample_Analysis\Pix3Sharing.csv' )
# CS_data = pd.read_csv( 'C:\Jiajin\Mfile\Training_Sample_Analysis\Pix3Sharing_corr.csv' )
Energy = CS_data.iloc[:, :].values  
Energy_sum = ( Energy[:,0] + Energy[:,1] + Energy[:,2] ).reshape(-1,1)

Energy_E1_E2_sum = Energy[ np.intersect1d(np.where(Energy_sum >= 100)[0],np.where(Energy_sum <= 130)[0]) ]
Energy_sum = Energy_sum[ np.intersect1d(np.where(Energy_sum >= 100)[0],np.where(Energy_sum <= 130)[0]) ]
Energy_E1_E2_sum = np.hstack( (Energy_E1_E2_sum, Energy_sum) )

##### plot the histogram of sum_Energy within the selected ROI
plt.figure(figsize=(12, 12), facecolor='w')
plt.grid('on')
plt.suptitle('Histogram of the sum of 3-pixel charge sharing events', fontsize = 24, fontweight = 'bold')
plt.hist(Energy_E1_E2_sum[:,3], bins=100, normed=0, facecolor="blue", edgecolor="black", alpha=0.7)
plt.xlim(50, 180)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Energy (keV)',fontsize=20)
plt.ylabel('Counts per bin size',fontsize=20)

##### plot the raw scatter figures within the selected ROI
fig = plt.figure()
ax = Axes3D(fig, rect=[0, 0, 1, 1], elev=45, azim=45)
plt.suptitle(u'3-pixel charge sharing raw scattering data', fontsize=20)
ax = plt.subplot(111, projection='3d')
ax.scatter(Energy_E1_E2_sum[:,0], Energy_E1_E2_sum[:,1], Energy_E1_E2_sum[:,2], s=10, c='k',marker='.')
plt.xlabel('Energy (keV)',fontsize=20)
plt.ylabel('Energy (keV)',fontsize=20)
plt.xlim(0, 140)
plt.ylim(0, 140)
########################################################################################################

##### PCA ######
pca = PCA( n_components=3 )
Energy_new = pca.fit_transform(cluster_lab_corrected_noise[:,0:4])
print( pca.explained_variance_ratio_ )
print( pca.explained_variance_ )

fig = plt.figure()
ax = Axes3D(fig, rect=[0, 0, 1, 1], elev=45, azim=45)
plt.suptitle(u'3-pixel charge sharing raw scattering data', fontsize=20)
ax.scatter(Energy_new[:, 0], Energy_new[:, 1], Energy_new[:, 2], c='k', marker='.')
plt.xlabel(' "Spatial dimension 1" ',fontsize=20)
plt.ylabel(' "Spatial dimension 1" ',fontsize=20)

fig2 = plt.figure(111)
plt.suptitle('Histogram of Scattering Points along Principle Component Axis', fontsize = 24, fontweight = 'bold')
plt.hexbin(Energy_new[:, 0], Energy_new[:, 1], gridsize=(100,100),cmap=plt.cm.BuGn_r )  #cmap="viridis" )
plt.colorbar()
plt.xlabel(' "Spatial dimension 1" ',fontsize=20)
plt.ylabel(' "Spatial dimension 2" ',fontsize=20)
plt.xlim(-80, 85)
plt.ylim(-70, 98)

fig3 = plt.figure(112)
plt.suptitle('Histogram of Scattering Points along Non-Principle Component Axis', fontsize = 24, fontweight = 'bold')
plt.hist(Energy_new[:, 2], bins=100, normed=0, facecolor="blue", edgecolor="black", alpha=0.7)
plt.xlim(-40, 40)
plt.xlabel(' "Energy" ',fontsize=20)
plt.ylabel(' Counts ',fontsize=20)
########################################################################################################




######################################### 4-pixel PCA ###################################################
##### Load the charge sharing events ####################################################################
CS_data = pd.read_csv( 'C:\Jiajin\Mfile\Training_Sample_Analysis\Pix4Sharing.csv' )
# CS_data = pd.read_csv( 'C:\Jiajin\Mfile\Training_Sample_Analysis\Pix4Sharing_corr.csv' )
Energy = CS_data.iloc[:, :].values  
Energy_sum = np.sum(Energy, axis = 1).reshape(-1,1)

Energy_E1_E2_sum = Energy[ reduce( np.intersect1d, ( np.where(Energy_sum >= 100)[0],np.where(Energy_sum <= 130)[0] ) ), : ]
Energy_sum = Energy_sum[ np.intersect1d(np.where(Energy_sum >= 100)[0],np.where(Energy_sum <= 130)[0]) ]
Energy_E1_E2_sum = np.hstack( (Energy_E1_E2_sum, Energy_sum) )

##### plot the histogram of sum_Energy within the selected ROI
plt.figure(figsize=(12, 12), facecolor='w')
plt.grid('on')
plt.suptitle('Histogram of the sum of 3-pixel charge sharing events', fontsize = 24, fontweight = 'bold')
plt.hist(Energy_E1_E2_sum[:,4], bins=100, normed=0, facecolor="blue", edgecolor="black", alpha=0.7)
plt.xlim(50, 180)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Energy (keV)',fontsize=20)
plt.ylabel('Counts per bin size',fontsize=20)

##### plot the raw scatter figures within the selected ROI
fig = plt.figure()
ax = Axes3D(fig, rect=[0, 0, 1, 1], elev=45, azim=45)
plt.suptitle(u'3-pixel charge sharing raw scattering data', fontsize=20)
ax = plt.subplot(111, projection='3d')
ax.scatter(Energy_E1_E2_sum[:,0], Energy_E1_E2_sum[:,1], Energy_E1_E2_sum[:,2], s=10, c='k',marker='.')
plt.xlabel('Energy (keV)',fontsize=20)
plt.ylabel('Energy (keV)',fontsize=20)
plt.xlim(0, 140)
plt.ylim(0, 140)
########################################################################################################

##### PCA ######
pca = PCA( n_components=4 )
Energy_new = pca.fit_transform(cluster_lab__noise[:,0:5])
print( pca.explained_variance_ratio_ )
print( pca.explained_variance_ )

fig = plt.figure(figsize=(12, 12), facecolor='w')
ax = Axes3D(fig, rect=[0, 0, 1, 1], elev=75, azim=45)
plt.suptitle(u'4-pixel Scattering Points along Principle Component Axis before correction', fontsize=20)
ax.scatter(Energy_new[:, 0], Energy_new[:, 1], Energy_new[:, 2], c='r', marker='.')
plt.xlabel(' "Spatial dimension 1" ',fontsize=20)
plt.ylabel(' "Spatial dimension 1" ',fontsize=20)

fig3 = plt.figure(112)
plt.suptitle('Histogram of Scattering Points along Non-Principle Component Axis', fontsize = 24, fontweight = 'bold')
plt.hist(Energy_new[:, 3], bins=100, normed=0, facecolor="blue", edgecolor="black", alpha=0.7)
plt.xlim(-40, 40)
plt.xlabel(' "Energy" ',fontsize=20)
plt.ylabel(' Counts ',fontsize=20)