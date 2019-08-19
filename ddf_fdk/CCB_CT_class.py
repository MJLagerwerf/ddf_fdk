#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 11:31:29 2018

@author: lagerwer
"""


import numpy as np
import odl
import tabulate
import os
import gc
import shutil
import pylab

from . import algorithm_class as algo
from . import support_functions as sup
from . import image_measures as im
from . import SIRT_ODL_astra_backend as SIRT
from . import phantom_class as PH
from . import real_data_class as RD

# %%
class CCB_CT:
    # %% Initialize the MR-FDK
    def __init__(self, data_obj, data_struct=True, **kwargs):
        # %% Store the input in the object
        # BE FUCKING CONSISTENT WITH RENAMING ETC. 
        self.pix = data_obj.voxels[0]
        self.angles = data_obj.angles
        self.src_rad = data_obj.src_rad
        self.det_rad = data_obj.det_rad
        self.magn = self.src_rad / (self.src_rad + self.det_rad)
        self.data_struct = data_struct
        self.rec_methods = []
        # %% If we need a data structure, make one:
        self.WV_obj = sup.working_var_map()
        self.WV_path = self.WV_obj.WV_path

        # %% Set up the geometry
        self.phantom = data_obj
        # Make the reconstruction space
        self.reco_space = self.phantom.reco_space
        voxels = self.phantom.voxels
        factor = 2
        dpix = [int(factor * voxels[0]), voxels[1]]
        self.w_detu = (2 * self.phantom.detecsize[0]) / dpix[0]
        self.w_detv = (2 * self.phantom.detecsize[1]) / dpix[1]
        if self.phantom.data_type == 'simulated':
            self.PH = data_obj.PH
            self.noise = data_obj.noise
            # Do we need a mask?
            if data_struct:
                self.phantom.make_mask(self.WV_path)
            else:
                pass
            src_radius = self.src_rad * self.phantom.volumesize[0] * 2
            det_radius = self.det_rad * self.phantom.volumesize[0] * 2
            # Make a circular scanning geometry
            angle_partition = odl.uniform_partition(0, 2 * np.pi, self.angles)
            self.angle_space = odl.uniform_discr_frompartition(angle_partition)
            # Make a flat detector space
            det_partition = odl.uniform_partition(-self.phantom.detecsize,
                                                   self.phantom.detecsize, dpix)
            self.det_space = odl.uniform_discr_frompartition(det_partition,
                                                         dtype='float32')
            # Create geometry
            self.geometry = odl.tomo.ConeFlatGeometry(
                angle_partition, det_partition, src_radius=src_radius,
                                det_radius=det_radius, axis=[0, 0, 1])
        else:
            src_radius = self.phantom.src_rad
            det_radius = self.phantom.det_rad
            self.pix_size = self.phantom.pix_size
            self.angle_space = self.phantom.angle_space
            self.angles = np.size(self.angle_space)
            self.det_space = self.phantom.det_space
            self.geometry = self.phantom.geometry
        # Create filter space, same size as the detector, because stability
        filter_part = odl.uniform_partition(- self.phantom.detecsize[0],
                                             self.phantom.detecsize[0],
                                                  dpix[0])
#        filter_part = odl.uniform_partition(-2 * self.phantom.detecsize[0],
#                                            2 * self.phantom.detecsize[0],
#                                                 2 * dpix[0])

        self.filter_space = odl.uniform_discr_frompartition(filter_part,
                                                            dtype='float64')
        # %% Create the FP and BP and the data
        self.g = self.phantom.g
        # Create fourier filter space
        fourier_filter_part = odl.uniform_partition(0, np.pi,
                                        (self.conv_op.frs_detu))
        self.fourier_filter_space = odl.uniform_discr_frompartition(
                fourier_filter_part, dtype='complex64')
        
        if self.phantom.vecs == False:
            # Forward Projection
            self.FwP = odl.tomo.RayTransform(self.reco_space, self.geometry,
                                             use_cache=False)
    
            # Backward Projection
            self.BwP = odl.tomo.RayBackProjection(self.reco_space, self.geometry,
                                                  use_cache=False)
    
            # %% Create the operators for the FDK framework
    
            # Create the FDK weighting for the data
            w_FDK = sup.FDK_weighting(dpix, self.det_space, self.w_detu,
                                      self.w_detv, src_radius)
    
            # Scale the data according to the FDK weighting
            self.g_scl = np.asarray(self.g.copy())
            for a in range(self.angles):
                self.g_scl[a, :, :] *= w_FDK
    
            self.conv_op = sup.ConvolutionOp(self.filter_space, self.FwP.range,
                                             self.det_space, self.w_detv,
                                             self.g_scl,
                                             self.angle_space.weighting.const)
    

    
            # Create FDK operator which takes filter
            self.FDK_op = self.BwP * self.conv_op

# %% Initialize the algorithms
    def init_algo(self):
        self.FDK = algo.FDK_class(self)
        self.rec_methods += [self.FDK]

        self.SIRT = algo.SIRT_class(self)
        self.rec_methods += [self.SIRT]
        
        self.SIRT_NN = algo.SIRT_class(self, non_neg=True)
        self.rec_methods += [self.SIRT_NN]

    def init_DDF_FDK(self, bin_param=2, expansion_op='linear'):
        self.spf_space, self.Exp_op = sup.ExpOp_builder(bin_param,
                                                        self.filter_space,
                                                        interp=expansion_op)
        # Create the forward-operator 'A' for our framework
        self.DDC = self.FwP * self.FDK_op * self.Exp_op

        self.DC = self.FDK_op * self.Exp_op
        # Create the FDK binned operator
        self.FDK_bin = self.FDK_op * self.Exp_op

        self.LSFDK = algo.LSFDK_class(self)
        self.rec_methods += [self.LSFDK]

        self.TFDK = algo.TFDK_class(self)
        self.rec_methods += [self.TFDK]

        self.SFDK = algo.SFDK_class(self)
        self.rec_methods += [self.SFDK]
        
        self.PIFDK = algo.PIFDK_class(self)
        self.rec_methods += [self.PIFDK]

    def init_LRF_FDK(self, factor, bin_param=2, expansion_op='linear'):
        # Get a low resolution phantom
        if self.phantom.data_type == 'simulated':
            vox_LR = np.asarray(self.phantom.voxels) // factor
            pha_LR = PH.phantom(vox_LR, self.phantom.PH)
            self.CT_LR = CCB_CT(pha_LR, angles=self.angles,
                                src_rad=self.src_rad, det_rad=self.det_rad,
                               noise=self.noise, data_struct='no')
        else:

            LR_dataset = {'g' : self.phantom.dataset['LR_data']}
            pha_LR = RD.real_data(LR_dataset, self.phantom.pix_size * factor,
                                  self.phantom.src_rad, self.phantom.det_rad,
                                  self.phantom.angles_in, self.phantom.ang_freq)

            self.CT_LR = CCB_CT(pha_LR, angles=None, src_rad=None, det_rad=None,
                               noise=None, data_struct='no')

        self.CT_LR.spf_space, self.CT_LR.Exp_op = sup.ExpOp_builder(bin_param,
                                                        self.CT_LR.filter_space,
                                                        interp=expansion_op)
        # Create the forward-operator 'A' for our framework
        self.CT_LR.DDC = self.CT_LR.FwP * self.CT_LR.FDK_op * self.CT_LR.Exp_op

        # Create the FDK binned operator
        self.CT_LR.FDK_bin = self.CT_LR.FDK_op * self.CT_LR.Exp_op

        self.LR_LSFDK = algo.LR_LSFDK_class(self)
        self.rec_methods += [self.LR_LSFDK]

        self.LR_TFDK = algo.LR_TFDK_class(self)
        self.rec_methods += [self.LR_TFDK]

        self.LR_SFDK = algo.LR_SFDK_class(self)
        self.rec_methods += [self.LR_SFDK]

# %% Compute the matrices for the filters, are only called in the algorithms
    def Matrix_Comp_A(self):
        if hasattr(self, 'AtA'):
            pass
        else:
            self.AtA = odl.operator.oputils.matrix_representation(
                    self.DDC.adjoint * self.DDC)
            self.Atg = np.asarray(self.DDC.adjoint(self.g))
            self.DDC_norm = np.linalg.norm(self.AtA) ** (1/2)

    def Matrix_Comp_B(self):
        if hasattr(self, 'BtB'):
            pass
        else:
            self.BtB = odl.operator.oputils.matrix_representation(
                    self.DC.adjoint * self.DC)
            self.Btf = np.asarray(self.DC.adjoint(self.phantom.f))
            self.DC_norm = np.linalg.norm(self.BtB) ** (1/2)

    def Matrix_Comp_C(self):
        if hasattr(self, 'CtC'):
            pass
        else:
            grad = odl.Gradient(self.reco_space)
            self.grad_FDK = grad * self.FDK_bin
            self.CtC = odl.operator.oputils.matrix_representation(
                    self.grad_FDK.adjoint * self.grad_FDK)
            self.gradFDK_norm = np.linalg.norm(self.CtC) ** (1/2)

# %% Algorithm support functions
    def pd_FFT(self, h):
            return self.fourier_filter_space.element(np.fft.rfft(
                    np.fft.ifftshift(self.conv_op.rs_filt(h))))

    def pd_IFFT(self, hf):
        return self.conv_op.rs_filt.inverse(np.fft.fftshift(np.fft.irfft(hf)))


# %% Compute a Golden Standard reconstruction on low resolution
    def compute_GS(self, factor=4, it=200):
        g_LR = sup.integrate_data(self.g, factor)
        if self.phantom.data_type == 'simulated':
            voxels_LR = np.asarray(self.phantom.voxels) // factor
            DO = PH.phantom(voxels_LR, self.PH, self.angles, self.noise,
                        self.src_rad, self.det_rad, load_data_g=g_LR)
        elif self.phantom.data_type == 'real':
            DO = RD.real_data(g_LR, self.pix_size, self.src_rad, self.det_rad,
                              self.angles, ang_freq=1)
        CTo = CCB_CT(DO, data_struct=False)
        CTo.init_algo()
        self.GS = CTo.SIRT_NN.do(niter=it, compute_results=False)
        CTo, DO, g_LR = None, None, None
        gc.collect()
        
# %% Quantify and show result
    def table(self, visualize='yes'):
        Ql = []
        # All possible methods
        headers = ['Method', 'MSE_msk', 'MAE_msk', 'SSIM_msk']
        for m in self.rec_methods:
            if hasattr(m, 'results'):
                Ql += [i for i in m.results.Ql]
        self.table_latex = tabulate.tabulate(Ql, headers, tablefmt="latex", 
                                               floatfmt=('.s',".4e",
                                                         ".4f", ".4f"))
        if visualize == "yes":
            print(tabulate.tabulate(Ql, headers, tablefmt="fancy_grid", 
                                    floatfmt=('.s',".4e", ".4f", ".4f")))
        elif visualize == "no":
            pass


# %%
    def show_phantom(self, clim=None, save_name=None, extension='.eps',
                     fontsize=20):
        space = self.reco_space
        if clim == None:
            clim = [np.min(self.phantom.f),
                    np.max(self.phantom.f)]
        mid = np.shape(self.phantom.f)[0] // 2
        xy = self.phantom.f[:, :, mid]
        xz = self.phantom.f[:, mid, :]
        yz = self.phantom.f[mid, :, :]
        fig, (ax1, ax2, ax3) = pylab.subplots(1, 3, figsize=[20, 6])
        fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
        ima = ax1.imshow(np.rot90(xy), clim=clim, extent=[space.min_pt[0],
                         space.max_pt[0],space.min_pt[1], space.max_pt[1]],
                            cmap='gray')
        ax1.set_xlabel('x', fontsize=fontsize)
        ax1.set_ylabel('y', fontsize=fontsize)
        ima = ax2.imshow(np.rot90(xz), clim=clim, extent=[space.min_pt[0],
                         space.max_pt[0],space.min_pt[2], space.max_pt[2]],
                            cmap='gray')
        ax2.set_xlabel('x', fontsize=fontsize)
        ax2.set_ylabel('z', fontsize=fontsize)
        ima = ax3.imshow(np.rot90(yz), clim=clim, extent=[space.min_pt[1],
                         space.max_pt[1],space.min_pt[2], space.max_pt[2]],
                            cmap='gray')
        ax3.set_xlabel('y', fontsize=fontsize)
        ax3.set_ylabel('z', fontsize=fontsize)
        fig.colorbar(ima, ax=(ax1, ax2, ax3))
        if save_name is None:
            if self.phantom.data_type == 'simulated':
                fig.suptitle('Ground truth', fontsize=fontsize+2)
            else:
                fig.suptitle('Gold standard', fontsize=fontsize+2)
        fig.show()
        if save_name is not None:
            pylab.savefig(save_name+extension, bbox_inches='tight')
