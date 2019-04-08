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

ex = Experiment()

# %%
@ex.config
def cfg():
    it_i = 0
    lp = '/export/scratch2/lagerwer/data/FleXray/pomegranate1_02MAR/' + \
                 'processed_data/'

    PH = 'pom1'
    ntype = ['good', 'noisy']

    dataset = {'g' : lp + 'g_'+ ntype[it_i] + '_ODL.npy',
               'ground_truth' : lp + 'ground_truth.npy',
                'mask' : lp + 'mask.npy'}
    # Create a data object
    # The amount of projection angles in the measurements
    if ntype[it_i] == 'good':
        angles = 2000
        ang_freq = 64
    elif ntype[it_i] == 'noisy':
        angles = 500
        ang_freq = 1

    # Source to center of rotation radius
    src_rad = 59
    det_rad = 10.9
    pix_size = 0.00748
    # Variables above are expressed in the phyisical size

    # Expansion operator and binnin parameter
    specifics = PH + '_' +  ntype[it_i]


# %%
@ex.capture
def CT(dataset, pix_size, src_rad, det_rad, angles, ang_freq):
    data_obj = ddf.real_data(dataset, pix_size, src_rad, det_rad, angles,
                             ang_freq)
  
    # %% Create the circular cone beam CT class
    CT_obj = ddf.CCB_CT(data_obj)
    CT_obj.init_algo()
    CT_obj.init_DDF_FDK()
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