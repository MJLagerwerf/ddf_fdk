#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 10:01:12 2018

@author: lagerwer
"""

# SIRT_odl_astra_backend
import astra
import numpy as np
import time
from . import support_functions as sup
sup.import_astra_GPU()
# %%
def SIRT_astra(g, niter, geom, reco_space, WV_path, non_neg=False,
               ang_freq=None):
    # %%
    ang, u, v = g.shape
    minvox = reco_space.min_pt[0]
    maxvox = reco_space.max_pt[0]
    vol_geom = astra.create_vol_geom(v, v, v, minvox, maxvox, minvox, maxvox,
                                     minvox, maxvox)
    w_du, w_dv = (geom.detector.partition.max_pt \
                    -geom.detector.partition.min_pt)/ np.array([u,v])
    if ang_freq is not None:
        angles = np.linspace(np.pi/500, (2+1/500)*np.pi, 500, False)[::ang_freq]
    else:
        angles = np.linspace(np.pi/ang, (2+1/ang)*np.pi, ang, False)
    proj_geom = astra.create_proj_geom('cone', w_dv, w_du, v, u,
                                       angles, geom.src_radius, geom.det_radius)
    g = np.transpose(np.asarray(g.copy()), (2,0,1))
    # %%
    proj_id = astra.data3d.create('-proj3d', proj_geom, g)
    projector_id = astra.create_projector('cuda3d', proj_geom, vol_geom)
    # %%
    # Create a data object for the reconstruction
    rec_id = astra.data3d.create('-vol', vol_geom)

    # Set up the parameters for a reconstruction algorithm using the GPU
    astra.plugin.register(astra.plugins.SIRTPlugin)
    cfg = astra.astra_dict('SIRT-PLUGIN')
#    cfg = astra.astra_dict('SIRT3D_CUDA')
    cfg['ReconstructionDataId'] = rec_id
    cfg['ProjectionDataId'] = proj_id
    cfg['ProjectorId'] = projector_id

    if non_neg:
        cfg['option']={}
        cfg['option']['MinConstraint'] = 0
    
    # Create the algorithm object from the configuration structure
    alg_id = astra.algorithm.create(cfg)
    if type(niter) == list:
        t_rec = np.zeros(len(niter))
        xlist = []
        teller = 0
        for i in range(len(niter)):
            t = time.time()
            astra.algorithm.run(alg_id, niter[i]-teller)
            teller += niter[i]
            rec = astra.data3d.get(rec_id)
            rec = np.transpose(rec, (2,1,0))
            t_rec[i] = time.time() - t
            np.save(WV_path +'/SIRT_' + str(niter[i])+'.npy', rec)
            xlist += ['SIRT_' + str(niter[i]) + '.npy']
            t_tot = 0
        for i in range(len(niter)):
            t_tot += t_rec[i]
            t_rec[i] = t_tot
    if type(niter) is not list:
        t = time.time()
        astra.algorithm.run(alg_id, niter)
        rec = astra.data3d.get(rec_id)
        rec = np.transpose(rec, (2,1,0))
        t_rec = time.time() - t
        np.save(WV_path + '/SIRT_' + str(niter) + '.npy', rec)

    # Clean up. Note that GPU memory is tied up in the algorithm object,
    # and main RAM in the data objects.
    astra.algorithm.delete(alg_id)
    astra.data3d.delete(rec_id)
    astra.data3d.delete(proj_id)
    astra.projector3d.delete(projector_id)
    if type(niter) == list:
        return xlist, t_rec
    if type(niter) is not list:
        return 'SIRT_' + str(niter)+'.npy', t_rec
