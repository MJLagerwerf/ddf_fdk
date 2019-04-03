#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 16:28:24 2018

@author: lagerwer
"""

import numpy as np
import ddf_fdk as ddf
from sacred import Experiment
from sacred.observers import FileStorageObserver
import gc
import pylab
import os
import time


ex = Experiment()
# %%
@ex.config
def cfg():
    it_i = 0
    it_j = 1

    pix = 1024 
    # Specific phantom
    phantoms = ['FORBILD', 'Cluttered sphere']
    phantom = phantoms[it_j]
    # Number of angles
    ang = [32, 360]
    noises = [None, ['Poisson', 2 ** 10]]    
    src_rads = [20, 17, 13, 10, 7, 5, 3, 2, 1]

    angles = ang[it_j]
    noise = noises[it_j]
    lam_T = lam_Ts[it_j][it_i] 
    lam_S = lam_Ss[it_j][it_i] 
    # Source radius
    src_rad = src_rads[it_i]

    lp = '/export/scratch2/lagerwer/NNFDK/phantoms/'
    f_load_path = None
    g_load_path = None


    # Specifics for the expansion operator
    PH = ['FB', 'CS']
    Exp_bin = 'linear'
    bin_param = 2
    specifics = PH[it_j] +'_SR' + str(src_rads[it_i])
    

# %%
@ex.capture
def CT(pix, phantom, angles, src_rad, noise, Exp_bin, bin_param, f_load_path,
       g_load_path):
    voxels = [pix, pix, pix]
    det_rad = 0
    if g_load_path is not None and f_load_path is not None:
        data_obj = ddf.phantom(voxels, phantom, angles, noise, src_rad, 
                               det_rad, load_data_g=g_load_path, 
                               load_data_f=f_load_path)
    elif g_load_path is not None:
        data_obj = ddf.phantom(voxels, phantom, angles, noise, src_rad, 
                               det_rad, load_data_g=g_load_path)
    elif f_load_path is not None:
        data_obj = ddf.phantom(voxels, phantom, angles, noise, src_rad, 
                               det_rad, load_data_f=f_load_path)
    else:
        data_obj = ddf.phantom(voxels, phantom, angles, noise, src_rad, 
                               det_rad)

  
    # %% Create the circular cone beam CT class
    CT_obj = ddf.CCB_CT(data_obj)

    CT_obj.init_algo()
    CT_obj.init_DDF_FDK(bin_param, Exp_bin)
    return CT_obj

# %%
@ex.automain
def main(specifics):
    if not os.path.exists('AFFDK_results'):
        os.makedirs('AFFDK_results')
    t2 = time.time()
    # Create a data object
    case = CT()
    t3 = time.time()
    print(t3 - t2, 'seconds to initialize CT object')

    Q = np.zeros((11, 3))

    
    np.save(case.WV_path + specifics + '_g.npy', case.g)
    ex.add_artifact(case.WV_path + specifics + '_g.npy')

    f = 'Shepp-Logan'
    LP_filts = [['Gauss', 8], ['Gauss', 5], ['Bin', 2], ['Bin', 5]]
    
    case.FDK.do(f)
    np.save(case.WV_path + specifics + '_FDKSL_rec.npy',
            case.FDK.results.rec_axis[-1])
    ex.add_artifact(case.WV_path + specifics + '_FDKSL_rec.npy')
    
    for lp in LP_filts:
        case.FDK.filt_LP(f, lp)
        if lp[0] == 'Gauss':
            np.save(case.WV_path + specifics + '_FDKSL_GS' + str(lp[1]) + 
                    '_rec.npy', case.FDK.results.rec_axis[-1])
            ex.add_artifact(case.WV_path + specifics + '_FDKSL_GS' + str(lp[1])
                            + '_rec.npy')
        elif lp[0] == 'Bin':
            np.save(case.WV_path + specifics + '_FDKSL_BN' + str(lp[1]) + 
                    '_rec.npy', case.FDK.results.rec_axis[-1])
            ex.add_artifact(case.WV_path + specifics + '_FDKSL_BN' + str(lp[1])
                            + '_rec.npy')
    
    Q[:5, :] = case.FDK.results.Q

    print('Finished FDKs')
    
    
    case.TFDK.do(lam='optim')
    np.save(case.WV_path + specifics + '_TFDK_rec.npy',
            case.TFDK.results.rec_axis[-1])
    ex.add_artifact(case.WV_path + specifics + '_TFDK_rec.npy')
    Q[6, :] = case.TFDK.results.Q
    

    print('Finished AF-FDK')




    case.table()
    latex_table = open(case.WV_path + specifics + '_latex_table.txt', 'w')
    latex_table.write(case.table_latex)
    latex_table.close()
    ex.add_artifact(case.WV_path + specifics + '_latex_table.txt')


    np.save(case.WV_path + specifics + '_Q.npy', Q)
    ex.add_artifact(case.WV_path + specifics + '_Q.npy')
    case = None
    gc.collect()
    return Q