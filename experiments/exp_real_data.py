#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 11:09:07 2019

@author: lagerwer
"""




import numpy as np
import ddf_fdk as ddf
from sacred import Experiment
from sacred.observers import FileStorageObserver
import gc
import os
import time
ddf.import_astra_GPU()
ex = Experiment()

# %%
@ex.config
def cfg():
    it_i = 0
    bpath ='/bigstore/lagerwer/data/FleXray/'
    load_path = f'{bpath}pomegranate1_02MAR/'
    
    dsets = ['noisy', 'good']
    dset = dsets[it_i]
    pd = 'processed_data/'
    sc = 1
    ang_freqs = [1, 32]
    ang_freq = ang_freqs[it_i]

    # Load data?
    f_load_path = None
    g_load_path = None
    
    # Specifics for the expansion operator
    Exp_bin = 'linear'
    bin_param = 2

    PH = 'pom1'
    specifics = PH + '_' + dset
    offset = 1.26/360 * 2 * np.pi

# %%  
@ex.capture
def CT(load_path, dset, sc, ang_freq, Exp_bin, bin_param, offset):
    dataset = ddf.load_and_preprocess_real_data(load_path, dset, sc)
    meta = ddf.load_meta(load_path + dset + '/', sc)
    pix_size = meta['pix_size']
    src_rad = meta['s2o']
    det_rad = meta['o2d']
    
    data_obj = ddf.real_data(dataset, pix_size, src_rad, det_rad, ang_freq,
                 zoom=True, offset=offset)

    CT_obj = ddf.CCB_CT(data_obj)
    CT_obj.init_algo()
    CT_obj.init_DDF_FDK()
    return CT_obj


# %%
@ex.capture
def save_and_add_artifact(path, arr):
    np.save(path, arr)
    ex.add_artifact(path)
    
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
    save_and_add_artifact(f'{case.WV_path}{specifics}_TFDK_rec.npy',
                          case.TFDK.results.rec_axis[-1])
    Q, RT = log_variables(case.TFDK.results, Q, RT)
    
    
    save_and_add_artifact(f'{case.WV_path}{specifics}_AtA.npy', case.AtA)
    save_and_add_artifact(f'{case.WV_path}{specifics}_Atg.npy', case.Atg)
    save_and_add_artifact(f'{case.WV_path}{specifics}_DDC_norm.npy',
                          case.DDC_norm)

    save_and_add_artifact(f'{case.WV_path}{specifics}_Q.npy', Q)
    save_and_add_artifact(f'{case.WV_path}{specifics}_RT.npy', RT)

    print('Finished MR-FDK')

    save_table(case, f'{case.WV_path}{specifics}')

    case = None
    gc.collect()
    return Q

