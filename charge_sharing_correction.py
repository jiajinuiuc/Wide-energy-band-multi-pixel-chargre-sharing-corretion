# -*- coding: utf-8 -*-
"""
charge_sharing_correction class
Objective: Assembled module used in multi-pixel charge sharing correction method
Class charge_sharing_correction:
    Basic functions:
    1. Corr_rot: coordinate transform based on the basis transformation
    2. Super_plane: calculate the CS super plane in the new coordinate
    3. Segment: data segmentation perpendicular to the super plane
    
    Correction functions:
    1. PixN_Measurement: measure the charge sharinge band shape with the number of CS pixels = N
    2. PixN_Correction: correct all events with the number of CS pixels = N
    3.* Correction: function can deal with arbitrary number of CS pixel events
      
Class SG_Filter:
    1. 1d S-G filter
    2. 2d S-G filter
    3.* 3d S-G filter
    
Class Common-used functions:
    1. Line_plot: class self-implemented line plot
    2. Scatter2D_plot: class self-implemented 2D scattering plot
    3. Scatter3D_plot: class self-implemented 3D scattering plot
    4. Histogram_plot: class self-implemented histogram plot
    5. Surface3D_plot: class self-implemented 3D surface plot
    6. SaveFiles: Save variables into .csv files
    
@author: J. J. Zhang
Last update: May, 2019
"""
import numpy as np
from functools import reduce
import matplotlib.pyplot as plt
import pandas as pd
from math import factorial
import scipy.signal

class charge_sharing_correction():
    def __init__( self, CS_dim, basis_old, basis_new, peak_energy, max_energy_range, seg_size ):
        self.CS_dim = CS_dim
        self.peak_energy = peak_energy
        self.max_energy_range = max_energy_range
        self.basis_old = basis_old
        self.basis_new = basis_new
        self.seg_size = seg_size
        
##### basic mathematics functions   
    def Corr_rot( self, basis_old, basis_new, original_position ): # coordinate transformation, input basis is required
        rotation_matrix = basis_old.dot( basis_new.I )
        rotated_position =  rotation_matrix.dot( original_position )
        rotated_position = rotated_position.T
        return rotated_position
    
    def Super_plane( self, energy ):
        Super_plane_const = np.sqrt( pow( (energy/self.CS_dim),2 )*self.CS_dim )
        return Super_plane_const
    
    def Segment( self ): # initialize the segmentation grid based on the max energy range
        ### each segmentation dimension share the same seg_unit
        ### cluster is segmented to [seg_min : self.seg_size : seg_max - self.seg_size]
        vertic_distance = np.sqrt(self.max_energy_range**2 - self.Super_plane(energy=self.max_energy_range)**2)
        seg_max = np.ceil(vertic_distance)
        seg_min = - np.floor(vertic_distance)
        seg_unit = np.arange( seg_min, seg_max, self.seg_size )
        return seg_unit  


    
##### CS measurement functions
    def Pix2_Measurement( self, CS_data_labeled ):
        ### Rotate to the decoupled space
        CS_data_labeled_rotated = self.Corr_rot( self.basis_old, self.basis_new, CS_data_labeled.T )
        ### Data segmentation and correction
        seg_unit = self.Segment() # cluster if segmented to [seg_left:self.seg_size:seg_right - self.seg_size]
        ### initialize parameters
        c = self.Super_plane( energy = self.peak_energy ) # c is the z-value of the correct plane in the new coordinate  
        ii = -1
        ncount = 0
        wet_x = np.zeros( [len(seg_unit)] ) # xyz-w weighted MC
        wet_w = np.zeros( [len(seg_unit)] )
        shift_w = np.zeros( [len(seg_unit)] )
        
        for low_seg_x in seg_unit :
            ii += 1
            if ii == len(seg_unit): # ii = 0 1 2 ... len(seg_x)-1
                ii = 0   
            high_seg_x = low_seg_x + self.seg_size
            
            wet_x[ii] = np.mean( [low_seg_x, high_seg_x] )
            tmp = CS_data_labeled_rotated[ reduce( np.intersect1d, (np.where(CS_data_labeled_rotated[:,0] >= low_seg_x)[0],\
                                                                    np.where(CS_data_labeled_rotated[:,0] < high_seg_x)[0]) ), : ]
            if len(tmp) != 0:
                ncount += 1
                wet_w[ii] = np.mean( tmp[:,1] )
                shift_w[ii] = c - wet_w[ii]
        return( wet_x, wet_w, shift_w, seg_unit )
        
        
    def Pix3_Measurement( self, CS_data_labeled ):   # Input the clustered charge sharing events
        ### Rotate to the decoupled space 
        CS_data_labeled_rotated = self.Corr_rot( self.basis_old, self.basis_new, CS_data_labeled.T )
        ### Data segmentation and correction
        seg_unit = self.Segment() # cluster if segmented to [seg_left:self.seg_size:seg_right - self.seg_size]
        ### initialize parameters
        c = self.Super_plane( energy = self.peak_energy ) # c is the z-value of the correct plane in the new coordinate  
        ii, jj = ( -1,-1 )
        ncount = 0
        wet_x = np.zeros( [len(seg_unit), len(seg_unit)] )
        wet_y = np.zeros( [len(seg_unit), len(seg_unit)] )
        wet_w = np.zeros( [len(seg_unit), len(seg_unit)] )
        shift_w = np.zeros( [len(seg_unit), len(seg_unit)] )

        for low_seg_x in seg_unit :
            ii += 1
            if ii == len(seg_unit): # ii = 0 1 2 ... len(seg_x)-1
                ii = 0
            high_seg_x = low_seg_x + self.seg_size
        
            for low_seg_y in seg_unit:
                jj += 1
                if jj == len(seg_unit): # jj = 0 1 2 ... len(seg_y)-1
                    jj = 0       
                high_seg_y = low_seg_y + self.seg_size
                
                wet_x[ii,jj] = np.mean( [low_seg_x, high_seg_x] )
                wet_y[ii,jj] = np.mean( [low_seg_y, high_seg_y] )
                tmp = CS_data_labeled_rotated[ reduce( np.intersect1d, (np.where(CS_data_labeled_rotated[:,0] >= low_seg_x)[0],\
                                                                        np.where(CS_data_labeled_rotated[:,0] < high_seg_x)[0],\
                                                                        np.where(CS_data_labeled_rotated[:,1] >= low_seg_y)[0],\
                                                                        np.where(CS_data_labeled_rotated[:,1] < high_seg_y)[0]) ), : ]
                if len(tmp) != 0:
                    ncount += 1
                    wet_w[ii,jj] = np.mean( tmp[:,2] )
                    shift_w[ii,jj] = c - wet_w[ii,jj]
        return( wet_x, wet_y, wet_w, shift_w, seg_unit  ) 
        
        
    def Pix4_Measurement( self, CS_data_labeled ):   
        ### Rotate to the decoupled space 
        CS_data_labeled_rotated = self.Corr_rot( self.basis_old, self.basis_new, CS_data_labeled.T )
        ### Data segmentation and correction
        seg_unit = self.Segment() # cluster if segmented to [seg_left:self.seg_size:seg_right - self.seg_size]
        ### initialize parameters
        c = self.Super_plane( energy = self.peak_energy ) # c is the z-value of the correct plane in the new coordinate   
        
        ii, jj, kk = ( -1,-1,-1 )
        ncount = 0
        wet_x = np.zeros( [len(seg_unit), len(seg_unit), len(seg_unit)] )
        wet_y = np.zeros( [len(seg_unit), len(seg_unit), len(seg_unit)] )
        wet_z = np.zeros( [len(seg_unit), len(seg_unit), len(seg_unit)] )
        wet_w = np.zeros( [len(seg_unit), len(seg_unit), len(seg_unit)] )
        shift_w = np.zeros( [len(seg_unit), len(seg_unit), len(seg_unit)] )

        for low_seg_x in seg_unit :
            ii += 1
            if ii == len(seg_unit): # ii = 0 1 2 ... len(seg_x)-1
                ii = 0
            high_seg_x = low_seg_x + self.seg_size
            
            for low_seg_y in seg_unit:
                jj += 1
                if jj == len(seg_unit): # jj = 0 1 2 ... len(seg_y)-1
                    jj = 0
                high_seg_y = low_seg_y + self.seg_size
        
                for low_seg_z in seg_unit:
                    kk += 1
                    if kk == len(seg_unit): # jj = 0 1 2 ... len(seg_y)-1
                        kk = 0
                    high_seg_z = low_seg_z + self.seg_size
                    
                    wet_x[ii,jj,kk] = np.mean( [low_seg_x, high_seg_x] )
                    wet_y[ii,jj,kk] = np.mean( [low_seg_y, high_seg_y] )
                    wet_z[ii,jj,kk] = np.mean( [low_seg_z, high_seg_z] )
                    tmp = CS_data_labeled_rotated[ reduce( np.intersect1d, (np.where(CS_data_labeled_rotated[:,0] >= low_seg_x)[0],\
                                                                            np.where(CS_data_labeled_rotated[:,0] < high_seg_x)[0],\
                                                                            np.where(CS_data_labeled_rotated[:,1] >= low_seg_y)[0],\
                                                                            np.where(CS_data_labeled_rotated[:,1] < high_seg_y)[0],\
                                                                            np.where(CS_data_labeled_rotated[:,2] >= low_seg_z)[0],\
                                                                            np.where(CS_data_labeled_rotated[:,2] < high_seg_z)[0]) ), : ]
                    if len(tmp) != 0:
                        ncount += 1
                        wet_w[ii,jj,kk] = np.mean( tmp[:,3] )
                        shift_w[ii,jj,kk] = c - wet_w[ii,jj,kk]
        return( wet_x, wet_y, wet_z, wet_w, shift_w, seg_unit  )     




##### CS correction functions        
    def Pix2_Correction( self, seg_unit, shift_w, CS_data_labeled ):
        ### Rotate to the decoupled space
        CS_data_labeled_rotated = self.Corr_rot( self.basis_old, self.basis_new, CS_data_labeled.T )
        ### initialize parameters 
        ii = -1
        ncount = 0
        
        for low_seg_x in seg_unit :
            ii += 1
            if ii == len(seg_unit): # ii = 0 1 2 ... len(seg_x)-1
                ii = 0   
            high_seg_x = low_seg_x + self.seg_size
            
            tmp = CS_data_labeled_rotated[ reduce( np.intersect1d, (np.where(CS_data_labeled_rotated[:,0] >= low_seg_x)[0],\
                                                                    np.where(CS_data_labeled_rotated[:,0] < high_seg_x)[0]) ), : ]
            if len(tmp) != 0:
                ncount += 1  
                shift = np.array([1]*len(tmp)).reshape(-1,1) * shift_w[ii]
                # Shift shift_z from the MC position, get the corrected cluster: cluster_lab_1_rotated_corrected
                tmp_corrected = np.hstack( ( tmp[:,0].reshape(-1,1), tmp[:,1].reshape(-1,1)+shift ) )
                    
                if ncount == 1:
                    print('Start 2-pixel correction !')
                    CS_data_labeled_rotated_corrected = tmp_corrected
                else:
                    CS_data_labeled_rotated_corrected = np.vstack( (CS_data_labeled_rotated_corrected, tmp_corrected) )

        ### Inverse rotate to the original space
        CS_data_labeled_rotated_corrected_rotated = self.Corr_rot( self.basis_new, self.basis_old, CS_data_labeled_rotated_corrected.T )
        return( CS_data_labeled_rotated_corrected_rotated )
   
    
    def Pix3_Correction( self, seg_unit, shift_w, CS_data_labeled ):   
        ### Rotate to the decoupled space 
        CS_data_labeled_rotated = self.Corr_rot( self.basis_old, self.basis_new, CS_data_labeled.T )
        ### initialize parameters
        ii, jj = ( -1,-1 )
        ncount = 0

        for low_seg_x in seg_unit :
            ii += 1
            if ii == len(seg_unit): # ii = 0 1 2 ... len(seg_x)-1
                ii = 0
            high_seg_x = low_seg_x + self.seg_size
        
            for low_seg_y in seg_unit:
                jj += 1
                if jj == len(seg_unit): # jj = 0 1 2 ... len(seg_y)-1
                    jj = 0       
                high_seg_y = low_seg_y + self.seg_size
                
                tmp = CS_data_labeled_rotated[ reduce( np.intersect1d, (np.where(CS_data_labeled_rotated[:,0] >= low_seg_x)[0],\
                                                                        np.where(CS_data_labeled_rotated[:,0] < high_seg_x)[0],\
                                                                        np.where(CS_data_labeled_rotated[:,1] >= low_seg_y)[0],\
                                                                        np.where(CS_data_labeled_rotated[:,1] < high_seg_y)[0]) ), : ]
                if len(tmp) != 0:
                    ncount += 1
                    # Shift shift_z from the MC position, get the corrected cluster: cluster_lab_1_rotated_corrected
                    shift = np.array([1]*len(tmp)).reshape(-1,1) * shift_w[ii,jj]
                    tmp_corrected = np.hstack( ( tmp[:,0].reshape(-1,1), tmp[:,1].reshape(-1,1), \
                                                 tmp[:,2].reshape(-1,1)+shift) )
                    if ncount == 1:
                        print('Start 3-pixel correction !')
                        CS_data_labeled_rotated_corrected = tmp_corrected
                    else:
                        CS_data_labeled_rotated_corrected = np.vstack( (CS_data_labeled_rotated_corrected, tmp_corrected) )
                        
        ### Inverse rotate to the original space
        CS_data_labeled_rotated_corrected_rotated = self.Corr_rot( self.basis_new, self.basis_old, CS_data_labeled_rotated_corrected.T )
        return( CS_data_labeled_rotated_corrected_rotated )
        
        
    def Pix4_Correction( self, seg_unit, shift_w, CS_data_labeled ):   
        ### Rotate to the decoupled space 
        CS_data_labeled_rotated = self.Corr_rot( self.basis_old, self.basis_new, CS_data_labeled.T )
        ### initialize parameters
        ii, jj, kk = ( -1,-1,-1 )
        ncount = 0

        for low_seg_x in seg_unit :
            ii += 1
            if ii == len(seg_unit): # ii = 0 1 2 ... len(seg_x)-1
                ii = 0
            high_seg_x = low_seg_x + self.seg_size
            
            for low_seg_y in seg_unit:
                jj += 1
                if jj == len(seg_unit): # jj = 0 1 2 ... len(seg_y)-1
                    jj = 0
                high_seg_y = low_seg_y + self.seg_size
        
                for low_seg_z in seg_unit:
                    kk += 1
                    if kk == len(seg_unit): # jj = 0 1 2 ... len(seg_y)-1
                        kk = 0
                    high_seg_z = low_seg_z + self.seg_size
                    
                    tmp = CS_data_labeled_rotated[ reduce( np.intersect1d, (np.where(CS_data_labeled_rotated[:,0] >= low_seg_x)[0],\
                                                                            np.where(CS_data_labeled_rotated[:,0] < high_seg_x)[0],\
                                                                            np.where(CS_data_labeled_rotated[:,1] >= low_seg_y)[0],\
                                                                            np.where(CS_data_labeled_rotated[:,1] < high_seg_y)[0],\
                                                                            np.where(CS_data_labeled_rotated[:,2] >= low_seg_z)[0],\
                                                                            np.where(CS_data_labeled_rotated[:,2] < high_seg_z)[0]) ), : ]
                    if len(tmp) != 0:
                        ncount += 1
                        # Shift shift_z from the MC position, get the corrected cluster: cluster_lab_1_rotated_corrected
                        shift = np.array([1]*len(tmp)).reshape(-1,1) * shift_w[ii,jj,kk]
                        tmp_corrected = np.hstack( ( tmp[:,0].reshape(-1,1), tmp[:,1].reshape(-1,1), tmp[:,2].reshape(-1,1),\
                                                     tmp[:,3].reshape(-1,1)+shift) )
                        if ncount == 1:
                            print('Start 4-pixel correction !')
                            CS_data_labeled_rotated_corrected = tmp_corrected
                        else:
                            CS_data_labeled_rotated_corrected = np.vstack( (CS_data_labeled_rotated_corrected, tmp_corrected) )
                            
        ### Inverse rotate to the original space
        CS_data_labeled_rotated_corrected_rotated = self.Corr_rot( self.basis_new, self.basis_old, CS_data_labeled_rotated_corrected.T )
        return( CS_data_labeled_rotated_corrected_rotated )     


##### common-used plotting functions
class Common_used_function():
    def Line_plot(x, y, color, x_label, y_label):
        plt.plot(x, y, color=color, marker='o', linestyle='dashed', linewidth=1.5, markersize=6)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.xlabel(x_label,fontsize=20)
        plt.ylabel(y_label,fontsize=20)
        plt.grid('on')

    def Scatter2D_plot(x, y, x_lim_left, x_lim_right, y_lim_left, y_lim_right, color='k'):
        plt.scatter(x, y, s=10, c=color,marker='.')
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.xlabel('Energy (keV)',fontsize=20)
        plt.ylabel('Energy (keV)',fontsize=20)
        plt.xlim(x_lim_left, x_lim_right)
        plt.ylim(y_lim_left, y_lim_right)
        plt.grid('on')

    def Histogram_barplot(Hist, Bins, x_lim_low, x_lim_high):
        plt.figure(figsize=(12, 12), facecolor='w')
        plt.hist(Hist, bins=Bins, normed=0, facecolor="blue", edgecolor="black", alpha=0.7)
        plt.xlim(x_lim_low, x_lim_high)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.xlabel('Energy (keV)',fontsize=20)
        plt.ylabel('Counts per bin size',fontsize=20)
        plt.grid('on')
        
    def Histogram_lineplot(Hist, Bins, x_lim_low, x_lim_high, color):
        bins = np.linspace( start=x_lim_low, stop=x_lim_high, num=Bins )
        hist, bin_edges = np.histogram( Hist, bins )
        plt.plot( bin_edges[:-1], hist, color=color, linewidth=3, alpha=0.7 )
        plt.xlim(x_lim_low, x_lim_high)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.xlabel('Energy (keV)',fontsize=20)
        plt.ylabel('Counts per bin size',fontsize=20)
        plt.grid('on')
        
    def Scatter3D_plot(ax, x, y, z, elev, azim, color='k', marker='.'):
        ax.view_init(elev=elev,azim=azim)#改变绘制图像的视角,即相机的位置,azim沿着z轴旋转，elev沿着y轴 (45,90)
        ax.scatter(x, y, z, s=10, c=color, marker=marker)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.xlabel('Energy (keV)',fontsize=20)
        plt.ylabel('Energy (keV)',fontsize=20)
        #plt.xlim(0, x_lim)
        #plt.ylim(0, y_lim)
        
    def Surface3D_plot(ax, x, y, z, elev, azim):
        ax.view_init(elev=elev,azim=azim)#改变绘制图像的视角,即相机的位置,azim沿着z轴旋转，elev沿着y轴 (45,90)
        ax.plot_surface(x, y, z)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        
    def SaveFiles(var, var_name, location):
        dataframe = pd.DataFrame( var )
        dataframe.columns = var_name
        dataframe.to_csv(location, index=False, sep=',')


###### S-G Filtering class
class SG_Filter():
    def sg_1d(y, window_size, order, deriv=0, rate=1): # smooth 2-pixel CS curve
        try:
            window_size = np.abs(np.int(window_size))
            order = np.abs(np.int(order))
        except ValueError:
            raise ValueError("window_size and order have to be of type int")
        if window_size % 2 != 1 or window_size < 1:
            raise TypeError("window_size size must be a positive odd number")
        if window_size < order + 2:
            raise TypeError("window_size is too small for the polynomials order")
        order_range = range(order+1)
        half_window = (window_size -1) // 2
        # precompute coefficients
        b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
        m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
        # pad the signal at the extremes with
        # values taken from the signal itself
        firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
        lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
        y = np.concatenate((firstvals, y, lastvals))
        return np.convolve( m[::-1], y, mode='valid')


    def sg_2d (z, window_size, order, derivative=None): # smooth 3-pixel CS curve
        # number of terms in the polynomial expression
        n_terms = ( order + 1 ) * ( order + 2)  / 2.0
        if  window_size % 2 == 0:
            raise ValueError('window_size must be odd')
        if window_size**2 < n_terms:
            raise ValueError('order is too high for the window size')
        half_size = window_size // 2
        exps = [ (k-n, n) for k in range(order+1) for n in range(k+1) ]
    
        # coordinates of points
        ind = np.arange(-half_size, half_size+1, dtype=np.float64)
        dx = np.repeat( ind, window_size )
        dy = np.tile( ind, [window_size, 1]).reshape(window_size**2, )
    
        # build matrix of system of equation
        A = np.empty( (window_size**2, len(exps)) )
        for i, exp in enumerate( exps ):
            A[:,i] = (dx**exp[0]) * (dy**exp[1])
    
        # pad input array with appropriate values at the four borders
        new_shape = z.shape[0] + 2*half_size, z.shape[1] + 2*half_size
        Z = np.zeros( (new_shape) )
        # top band
        band = z[0, :]
        Z[:half_size, half_size:-half_size] =  band -  np.abs( np.flipud( z[1:half_size+1, :] ) - band )
        # bottom band
        band = z[-1, :]
        Z[-half_size:, half_size:-half_size] = band  + np.abs( np.flipud( z[-half_size-1:-1, :] )  -band )
        # left band
        band = np.tile( z[:,0].reshape(-1,1), [1,half_size])
        Z[half_size:-half_size, :half_size] = band - np.abs( np.fliplr( z[:, 1:half_size+1] ) - band )
        # right band
        band = np.tile( z[:,-1].reshape(-1,1), [1,half_size] )
        Z[half_size:-half_size, -half_size:] =  band + np.abs( np.fliplr( z[:, -half_size-1:-1] ) - band )
        # central band
        Z[half_size:-half_size, half_size:-half_size] = z
    
        # top left corner
        band = z[0,0]
        Z[:half_size,:half_size] = band - np.abs( np.flipud(np.fliplr(z[1:half_size+1,1:half_size+1]) ) - band )
        # bottom right corner
        band = z[-1,-1]
        Z[-half_size:,-half_size:] = band + np.abs( np.flipud(np.fliplr(z[-half_size-1:-1,-half_size-1:-1]) ) - band )
    
        # top right corner
        band = Z[half_size,-half_size:]
        Z[:half_size,-half_size:] = band - np.abs( np.flipud(Z[half_size+1:2*half_size+1,-half_size:]) - band )
        # bottom left corner
        band = Z[-half_size:,half_size].reshape(-1,1)
        Z[-half_size:,:half_size] = band - np.abs( np.fliplr(Z[-half_size:, half_size+1:2*half_size+1]) - band )
    
        # solve system and convolve
        if derivative == None:
            m = np.linalg.pinv(A)[0].reshape((window_size, -1))
            return scipy.signal.fftconvolve(Z, m, mode='valid')
        elif derivative == 'col':
            c = np.linalg.pinv(A)[1].reshape((window_size, -1))
            return scipy.signal.fftconvolve(Z, -c, mode='valid')
        elif derivative == 'row':
            r = np.linalg.pinv(A)[2].reshape((window_size, -1))
            return scipy.signal.fftconvolve(Z, -r, mode='valid')
        elif derivative == 'both':
            c = np.linalg.pinv(A)[1].reshape((window_size, -1))
            r = np.linalg.pinv(A)[2].reshape((window_size, -1))
            return scipy.signal.fftconvolve(Z, -r, mode='valid'), scipy.signal.fftconvolve(Z, -c, mode='valid')
        


################# Under development #####################
'''          
    def Correction( self, CS_data_labeled_rotated, self.seg_size ):   # for all situation
        ### initialize data segmentation unit       
        seg_unit = self.Segment( self.seg_size = self.seg_size)
        ### initialize mass center matrix of each dimension
        wet_xyz = np.zeros( [np.power(len(seg_unit), self.CS_dim-1), self.CS_dim-1] ) # store the grid center position
        wet_w = np.zeros( [np.power(len(seg_unit), self.CS_dim-1), 1] ) # store the MC on the energy dimension
        shift_w = np.zeros( [len(seg_unit), 1] ) # store the difference between the MC and the super plane
        ### initialize other parameters
        c = self.Super_plane( energy = self.peak_energy )
        index = [0]*len(self.CS_dim-1) # store the segment unit position
        index[0] = -1
        ncount = 0
        np_where = [ [0 for t in range(1)] for i in range(len(seg_unit)) ]# store the index of events fall in a vertain interval
        ### scan each segment
        for index_count in np.arange( np.power(len(seg_unit), self.CS_dim-1) ): # total times = len(seg_unit).^(CS_dim-1)
            index[0] += 1
            for ii in np.arange( len(index)-1 ):
                if index[ii] == len(seg_unit):
                    index[ii] = 0
                    index[ii+1] += 1
                seg_left = seg_unit[index[ii]]
                seg_right = seg_left + self.seg_size
                np_where[ii] = reduce( np.intersect1d, (np.where(CS_data_labeled_rotated[:,ii] >= seg_left)[0],\
                                                        np.where(CS_data_labeled_rotated[:,ii] < seg_right)[0] ) )
    
'''
    