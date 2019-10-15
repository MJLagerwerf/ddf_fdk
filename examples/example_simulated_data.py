#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 11:40:55 2018

@author: lagerwer
"""
import odl
import numpy as np
import ddf_fdk as ddf
import astra
ddf.import_astra_GPU()
#import real_data_class as RD
import time
import pylab
import scipy.interpolate as sp
import gc

pylab.close('all')
t = time.time()
# %% Set variables
# The size of the measured objects in voxels
pix = 256
voxels = [pix, pix, pix]

# Pick your phantom
# Options: 'Shepp-Logan', 'Defrise', 'Derenzo', 'Hollow cube', 'Cube', 'Var obj'
phantom = 'Fourshape_test'
#lp = '/export/scratch2/lagerwer/NNFDK_results/nTrain_optim_1024_lim_ang/'
#f_load_path = lp + 'CS_f.npy'
#g_load_path = lp + 'CS_A64_g.npy'
noise = None #['Poisson', 2 ** 8]
det_rad = 0
src_rad = 100
angles = 1500

data_obj = ddf.phantom(voxels, phantom, angles, noise, src_rad, det_rad,
                       compute_xHQ=True)

## %% Create the circular cone beam CT class
case = ddf.CCB_CT(data_obj)#
## Initialize the algorithms (FDK, SIRT)
case.init_algo()
#case.init_DDF_FDK()
# %%
#    case.TFDK.optim_param()
case.FDK.do('Ram-Lak')
rec = case.FDK.do('Hann', compute_results=False)
#case.TFDK.do(lam=1e-5)
#case.SIRT.do(100)
# %% Show results
pylab.close('all')
case.table()
case.FDK.show(0)
case.FDK.show()
case.show_phantom()
case.show_xHQ()


#case.SIRT.show()
# %%    
for i in range(10, 40, 5):
    pylab.figure()
    pylab.imshow()