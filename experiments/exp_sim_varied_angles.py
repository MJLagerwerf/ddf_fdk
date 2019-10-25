#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 10:40:16 2018

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
ddf.import_astra_GPU()

ex = Experiment()
# %%
@ex.config
def cfg():
    it_i = 0
    pix = 1024
    # Specific phantom
    phantom = 'FORBILD'
    # Number of angles
    ang = [16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240,
           256, 288, 320, 352, 394]

    angles = ang[it_i]

    # Source radius
    src_rad = 10

    lp = '/export/scratch2/lagerwer/NNFDK/phantoms/'
    f_load_path = None
    g_load_path = None

    noise = None #['Poisson', I0[it_i]]

    # Specifics for the expansion operator
    Exp_bin = 'linear'
    bin_param = 2
    specifics = 'ang' + str(ang[it_i])
    

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
@ex.capture
def save_and_add_artifact(path, arr):
    np.save(path, arr)
    ex.add_artifact(path)

@ex.capture
def save_network(case, full_path, NW_path):
    NW_full = h5py.File(full_path + NW_path, 'r')
    NW = h5py.File(case.WV_path + NW_path, 'w')

    NW_full.copy(str(case.NNFDK.network[-1]['nNW']), NW, name='NW')
    NW_full.close()
    NW.close()
    ex.add_artifact(case.WV_path + NW_path)
    
@ex.capture
def save_table(case, WV_path):
    case.table()
    latex_table = open(WV_path + '_latex_table.txt', 'w')
    latex_table.write(case.table_latex)
    latex_table.close()
    ex.add_artifact(WV_path + '_latex_table.txt')

@ex.capture
def log_variables(results, Q, RT):
    Q = np.append(Q, results.Q, axis=0)
    RT = np.append(RT, results.rec_time)
    return Q, RT
    


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
    Q = np.zeros((0, 3))
    RT = np.zeros((0))

    save_and_add_artifact(f'{case.WV_path}_g.npy', case.g)

    f = 'Shepp-Logan'
    LP_filts = [['Gauss', 8], ['Gauss', 5], ['Bin', 2], ['Bin', 5]]
    
    case.FDK.do(f)
    save_and_add_artifact(f'{case.WV_path}{specifics}_FDKSL_rec.npy',
                          case.FDK.results.rec_axis[-1])


    
    for lp in LP_filts:
        case.FDK.filt_LP(f, lp)
        if lp[0] == 'Gauss':
            save_and_add_artifact(f'{case.WV_path}{specifics}'+ \
                                  f'_FDKSL_GS{lp[1]}_rec.npy',
                                  case.FDK.results.rec_axis[-1])
        elif lp[0] == 'Bin':
            save_and_add_artifact(f'{case.WV_path}{specifics}'+ \
                                  f'_FDKSL_BN{lp[1]}_rec.npy',
                                  case.FDK.results.rec_axis[-1])
    Q, RT = log_variables(case.FDK.results, Q, RT)
    print('Finished FDKs')
    
    
    case.TFDK.do(lam='optim')
    save_and_add_artifact(f'{case.WV_path}_TFDK_rec.npy',
                          case.TFDK.results.rec_axis[-1])
    Q, RT = log_variables(case.TFDK.results, Q, RT)
    
    
    save_and_add_artifact(f'{case.WV_path}_AtA.npy', case.AtA)
    save_and_add_artifact(f'{case.WV_path}_Atg.npy', case.Atg)
    save_and_add_artifact(f'{case.WV_path}_DDC_norm.npy', case.DDC_norm)


    print('Finished MR-FDK')

    save_table(case, WV_path)

    case = None
    gc.collect()
    return Q