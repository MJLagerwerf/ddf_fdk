#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 11:40:55 2018

@author: lagerwer
"""
import odl
import numpy as np
import ddf_fdk as ddf
ddf.import_astra_GPU()
#import real_data_class as RD
import time
import pylab
import scipy.interpolate as sp
pylab.close('all')
t = time.time()
# %% Set variables
# The size of the measured objects in voxels
pix = 256
voxels = [pix, pix, pix]

# Pick your phantom
# Options: 'Shepp-Logan', 'Defrise', 'Derenzo', 'Hollow cube', 'Cube', 'Var obj'
phantom = 'FORBILD'
#lp = '/export/scratch2/lagerwer/NNFDK_results/nTrain_optim_1024_lim_ang/'
#f_load_path = lp + 'CS_f.npy'
#g_load_path = lp + 'CS_A64_g.npy'
noise = ['Poisson', 2 ** 8]
det_rad = 0
src_rad = 10
angles = 360
# The amount of projection angles in the measurements

# Source to center of rotation radius

# Variables above are expressed in the phyisical size of the measured object

# Noise model
# Options: None,  ['Gaussian', %intensity], ['Poisson', I_0], ['loaded data',
#                    filename]
#sc = 1
#zoom = False
#lp = '/export/scratch2/lagerwer/data/FleXray/walnuts_10MAY/walnut_11/' 
#dset = 'noisy'
#dataset = ddf.load_and_preprocess_real_data(lp, dset, sc=sc)
##
##proc_dat = 'processed_data/'
##dataset = {'g' : lp + proc_dat +  'g_good_sc4_shift.npy'}#,
##               'ground_truth' : lp + 'ground_truth_sc4.npy',
##                'mask' : lp + 'mask_sc4.npy'}
#
#ang_freq = 1
### %%
#meta = ddf.load_meta(lp + dset + '/', sc=sc)
#src_rad = meta['s2o'] 
#det_rad = meta['o2d']
#pix_size = meta['pix_size']
# Create a data object
data_obj = ddf.phantom(voxels, phantom, angles, noise, src_rad, det_rad)
#data_obj = ddf.real_data(dataset, pix_size, src_rad, det_rad, ang_freq,)
#                         zoom=zoom)
# Expansion operator and binnin parameter
#expansion_op = 'linear'
#bin_param = 2
#
## %% Create the circular cone beam CT class
case = ddf.CCB_CT(data_obj)#
## Initialize the algorithms (FDK, SIRT)
case.init_algo()
case.init_DDF_FDK()
# %%
case.TFDK.optim_param()
case.PIFDK.optim_param()
case.FDK.do('Ram-Lak')
case.TFDK.do('optim')
case.PIFDK.do('optim')

# %% Show results
case.table()
case.TFDK.show()
case.PIFDK.show()


