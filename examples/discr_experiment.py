#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 16:34:31 2019

@author: lagerwer
"""

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
import gc

pylab.close('all')
t = time.time()
# %% Set variables
# The size of the measured objects in voxels
pix = 1024
voxels = [pix, pix, pix]

# Pick your phantom
# Options: 'Shepp-Logan', 'Defrise', 'Derenzo', 'Hollow cube', 'Cube', 'Var obj'
phantom = 'FORBILD'
#lp = '/export/scratch2/lagerwer/NNFDK_results/nTrain_optim_1024_lim_ang/'
#f_load_path = lp + 'CS_f.npy'
#g_load_path = lp + 'CS_A64_g.npy'
noise = None #['Poisson', 2 ** 8]
det_rad = 0
src_rad = 10
angles = 360
up_samp = [1.5, 2, np.exp(1), 3, np.pi]

Q_TFDK = np.zeros((np.size(up_samp), 3))
Q_FDK = np.zeros((np.size(up_samp), 3))
i = 0
for us in up_samp:
    print('starting case sc=', us)
    data_obj = ddf.phantom(voxels, phantom, angles, noise, src_rad, det_rad,
                           samp_fac=us)

    ## %% Create the circular cone beam CT class
    case = ddf.CCB_CT(data_obj)#
    ## Initialize the algorithms (FDK, SIRT)
    case.init_algo()
    case.init_DDF_FDK()
    # %%
    if i == 0:
        lam = case.TFDK.optim_param()
    case.FDK.do('Ram-Lak')
    case.TFDK.do(lam=lam)
    Q_TFDK[i, :] = case.TFDK.results.Q
    Q_FDK[i, :] = case.FDK.results.Q
    # %% Show results
    case.table()
    i += 1
    case = None
    data_obj = None
    gc.collect()
    



headers = ['Method', '$sc=1.5$', '$sc=2$', '$sc=e$', '$sc=3$', '$sc=\pi$']

Ql = [['Ram-Lak: MAE', *Q_FDK[:, 1]], ['Ram-Lak: SSIM', *Q_FDK[:, 2]],
      ['MR-filter: MAE', *Q_TFDK[:, 1]], ['MR-filter: SSIM', *Q_TFDK[:, 2]]]
      

import tabulate as tab
latex_table = tab.tabulate(Ql, headers, tablefmt='latex', floatfmt=('.s',".4e",
                                                         ".4f", ".4f"))
table = open(case.WV_path + '_table.txt', 'w')
table.write(latex_table)
table.close()
print(tab.tabulate(Ql, headers, tablefmt='latex', floatfmt=('.s',".4f",
                                                         ".4f", ".4f",
                                                         ".4f", ".4f")))