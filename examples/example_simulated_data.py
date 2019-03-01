#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 11:40:55 2018

@author: lagerwer
"""

import numpy as np
import ddf_fdk as ddf
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
noise = ['Poisson', 2 ** 10]

# The amount of projection angles in the measurements
angles = 360
# Source to center of rotation radius
src_rad = 10
det_rad = 0
# Variables above are expressed in the phyisical size of the measured object

# Noise model
# Options: None,  ['Gaussian', %intensity], ['Poisson', I_0], ['loaded data',
#                    filename]

# Create a data object
data_obj = ddf.phantom(voxels, phantom, angles, noise, src_rad, det_rad)#, load_data=f_load_path)

# Expansion operator and binnin parameter
expansion_op = 'linear'
bin_param = 2

# %% Create the circular cone beam CT class
case = ddf.CCB_CT(data_obj)#,
##                  load_data=g_load_path)
## Initialize the algorithms (FDK, SIRT)
case.init_algo()
case.init_DDF_FDK()
# %%
case.FDK.do('Ram-Lak', backend='ODL')#, compute_results='no')
case.TFDK.do(lam=0.002)#, compute_results='no')
#case.SFDK.do(lam=0.004)#
# %% 
case.table()
#case.FDK.show()
#case.TFDK.show()
#case.TFDK.show_filt()
# %% Check convolution
pylab.close('all')
import odl
#x = case.spf_space.element(np.arange(case.spf_space.size))
x = case.spf_space.element(np.arange(case.spf_space.size)[::-1])
#x = case.spf_space.zero()
#x[0] = 1
#x[:4] = 1
#x = case.spf_space.one()
conv_x = case.conv_op(case.Exp_op(x))
det_line = odl.uniform_discr(-data_obj.detecsize[0], data_obj.detecsize[0],
                             int(2 * pix))
res_det = odl.ResizingOperator(case.filter_space, range=det_line)

an = 44
np_conv_x = res_det(np.convolve(case.Exp_op(x), case.g_scl[an, :, pix // 2],
                                'same'))

#print(np.linalg.norm(np_conv_x - np.asarray(conv_x[an, :, pix // 2])) / \
#      np.size(np_conv_x)) 
#((np_conv_x - np.asarray(conv_x[an, :, pix // 2]))).show()
#np_conv_x.show()
pylab.figure()
pylab.plot(np.roll(np_conv_x, -1))
pylab.plot(np.asarray(conv_x[an, :, pix // 2]))



