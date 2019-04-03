#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 13:53:18 2018

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
    pix = 1024
    # Specific phantom
    phantom = 'Low contrast'
    # Number of angles
    angles = 360
    I0 = [2 ** 10,  2 ** 14]
    noise = ['Poisson', I0[it_i]]
    lam_Ts = [6e-4, 2e-5]
    lam_Ss = [2e-3, 8e-4]

    lam_T = lam_Ts[it_i] 
    lam_S = lam_Ss[it_i] 
    # Source radius
    src_rad = 10

    lp = '/export/scratch2/lagerwer/NNFDK/phantoms/'
    f_load_path = None
    g_load_path = None



    # Specifics for the expansion operator
    Exp_bin = 'linear'
    bin_param = 2
    specifics = 'I0' + str(I0[it_i])
    

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
def main(specifics, lam_T, lam_S):
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
    

    
    case.TFDK.do(lam_T)
    np.save(case.WV_path + specifics + '_TFDK_rec.npy',
            case.TFDK.results.rec_axis[-1])
    ex.add_artifact(case.WV_path + specifics + '_TFDK_rec.npy')
    Q[6, :] = case.TFDK.results.Q
    


    np.save(case.WV_path + specifics + '_AtA.npy', case.AtA)
    ex.add_artifact(case.WV_path + specifics + '_AtA.npy')
    np.save(case.WV_path + specifics + '_Atg.npy', case.Atg)
    ex.add_artifact(case.WV_path + specifics + '_Atg.npy')
    np.save(case.WV_path + specifics + '_DDC_norm.npy', case.DDC_norm)
    ex.add_artifact(case.WV_path + specifics + '_DDC_norm.npy')


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