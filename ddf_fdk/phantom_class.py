#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 09:38:51 2018

@author: lagerwer
"""

import numpy as np
import odl
import scipy.ndimage.morphology as sp
import astra
from . import phantom_objects as po
from . import support_functions as sup
import gc

# %%
def make_spheres(center, radius, value=-1):
    rad = [radius, radius, radius]
    rot = [0, 0, 0]
    return [value, *rad, *center, *rot]

def make_ellipsoids(number):
    ellipsoids = []
    for i in range(number):
        value = np.random.rand(1)
        axis = np.random.rand(3)
        centers = np.random.rand(3) * 2 - 1
        rotations = np.random.rand(3) * 2 * np.pi
        ellipsoids += [[*value, *axis, *centers, *rotations]]
    return ellipsoids

def clip_circle(size, img):
    xx, yy = np.mgrid[:size, :size]
    mid = (size - 1. ) / 2.
    circle = (xx - mid) ** 2 + (yy - mid) ** 2
    bnd = size ** 2 / 4.
    outCircle = circle > bnd
    img[outCircle] = 0

def clip_cylinder(size, img):
    xx, yy = np.mgrid[:size, :size]
    mid = (size - 1) / 2
    circle = (xx - mid) ** 2 + (yy - mid) ** 2
    xx, yy = None, None
    bnd = size ** 2 / 4
    outCircle = circle > bnd
    img[:, :, :size // 8] = 0
    img[:, :, 7 * size // 8:] = 0
    img[outCircle] = 0
# %%
def FP_astra(f, reco_space, geom, factor):
    f = np.asarray(f)
    v = reco_space.shape[0]
    u = 2 * v
    a = geom.angles.size
    g = np.zeros((v, a, u), dtype='float32')
    minvox = reco_space.min_pt[0]
    maxvox = reco_space.max_pt[0]
    vol_geom = astra.create_vol_geom(v, v, v, minvox, maxvox, minvox, maxvox,
                                     minvox, maxvox)
    w_du, w_dv = (geom.detector.partition.max_pt \
                    -geom.detector.partition.min_pt) / np.array([u,v])
    
    ang = np.linspace(np.pi/a, (2 + 1 / a) * np.pi, a, False)
    
    proj_geom = astra.create_proj_geom('cone', w_du, w_dv, v, u,
                                       ang, geom.src_radius, geom.det_radius)

    project_id = astra.create_projector('cuda3d', proj_geom, vol_geom)
    W = astra.OpTomo(project_id)
    W.FP(f, out=g)
    return np.transpose(g, (1, 2, 0))

class phantom:
    def __init__(self, voxels, PH, angles, noise, src_rad, det_rad, **kwargs):
        self.data_type = 'simulated'
        voxels_up = [int(v * 1.5) for v in voxels]
        self.voxels = voxels
        self.PH = PH
        self.angles = angles
        self.noise = noise
        self.src_rad = src_rad
        self.det_rad = det_rad
        if self.PH == 'Zero' and self.noise == None:
            raise ValueError('Specify a noise level for this phantom')
            
        if 'load_data_g' in kwargs:
            if PH in ['Fourshape', 'Threeshape']:
                if 'load_data_f' not in kwargs:
                    raise ValueError('If you load data for this phantom,' + 
                                     ' you also need to provide a phantom')
            self.reco_space, self.f = self.phantom_creation(voxels, **kwargs)
            self.generate_data(voxels, self.reco_space, self.f, **kwargs)
        else:
            reco_space_up, f_up = self.phantom_creation(voxels_up, **kwargs)
            self.generate_data(voxels_up, reco_space_up, f_up, **kwargs)
            reco_space_up, f_up = None, None
            gc.collect()
            self.reco_space, self.f = self.phantom_creation(voxels, 
                                                            second=True,
                                                            **kwargs)
        
    def make_mask(self, WV_path):
        self.WV_path = WV_path
        if self.PH in ['Threeshape', 'Fourshape', '22 Ellipses',
                       'Fourshape_test']:
            mask = np.ones(self.voxels, dtype='bool')
            clip_cylinder(self.voxels[0], mask)
            np.save(WV_path + '/mask.npy', mask)
            self.mask_size = np.sum(mask)
        else:
            diter = int(1.5 * 2 ** (np.log2(self.voxels[0]) - 3))
            mask = sp.binary_fill_holes(np.asarray(self.f))
            mask = sp.binary_dilation(mask, iterations=diter)
            np.save(WV_path + '/mask.npy', mask)
            self.mask_size = np.sum(mask)

    def mask(self, x):
        mask = np.load(self.WV_path + '/mask.npy')
        return self.reco_space.element(mask * x)

    
    # %% Add noise function
    def add_poisson_noise(self, I_0, seed=None):
        seed_old = np.random.get_state()
        np.random.seed(seed)
        data = np.asarray(self.g.copy())
        Iclean = (I_0 * np.exp(-data))
        data = None
        Inoise = np.random.poisson(Iclean)
        Iclean = None
        np.random.set_state(seed_old)
        return  (-np.log(Inoise / I_0))
    

# %% Generate data and 
    def generate_data(self, voxels_up, reco_space_up, f_up, **kwargs):
        factor = 2
        dpix_up = [factor * voxels_up[0], voxels_up[1]]
        dpix = [int(factor * self.voxels[0]), self.voxels[0]]
        src_radius = self.src_rad * self.volumesize[0] * 2
        det_radius = self.det_rad * self.volumesize[0] * 2
        # Make a circular scanning geometry
        angle_partition = odl.uniform_partition(0, 2 * np.pi, self.angles)
        # Make a flat detector space
        det_partition = odl.uniform_partition(-self.detecsize,
                                               self.detecsize, dpix_up)
        # Create data_space_up and data_space
        data_space = odl.uniform_discr((0, *-self.detecsize),
                                       (2 * np.pi, *self.detecsize),
                                       [self.angles, *dpix], dtype='float32')
        data_space_up = odl.uniform_discr((0, *-self.detecsize),
                                       (2 * np.pi, *self.detecsize), 
                                       [self.angles, *dpix_up], dtype='float32')
        # Create geometry
        geometry = odl.tomo.ConeFlatGeometry(
            angle_partition, det_partition, src_radius=src_radius,
                            det_radius=det_radius, axis=[0, 0, 1])
        FP = odl.tomo.RayTransform(reco_space_up, geometry,
                                              use_cache=False)
        resamp = odl.Resampling(data_space_up, data_space)
        if 'load_data_g' in kwargs:
            if type(kwargs['load_data_g']) == str: 
                self.g = data_space.element(np.load(kwargs['load_data_g']))
            else:
                self.g = data_space.element(kwargs['load_data_g'])
        else:
            self.g = resamp(FP(f_up))#resamp(FP_astra(f_up, reco_space_up, geometry, factor))

            if self.noise == None:
                pass
            elif self.noise[0] == 'Gaussian':
                self.g += data_space.element(
                        odl.phantom.white_noise(resamp.range) * \
                        np.mean(self.g) * self.noise[1])
            elif self.noise[0] == 'Poisson':
                # 2**8 seems to be the minimal accepted I_0
                self.g = data_space.element(
                        self.add_poisson_noise(self.noise[1]))
            else:
                raise ValueError('unknown `noise type` ({})'
                             ''.format(self.noise[0]))

    def phantom_creation(self, voxels, **kwargs):
        if self.PH == "Shepp-Logan":
            # Size of the head is approximate 10 cm by 7.5 cm
            self.volumesize = np.array([5, 5, 5], dtype='float32')
            # Scale the detector correctly
            self.detecsize = np.array([2 * self.volumesize[0], 
                                       self.volumesize[1]])
            # Make the reconstruction space
            reco_space = odl.uniform_discr(min_pt=-self.volumesize,
                                           max_pt=self.volumesize,
                                            shape=voxels, dtype='float32',
                                            interp='linear')
            # Create the phantom:
            # Note: We rescale the gray matter in the brain to realistic
            # values but we use the scaled version of SL
            f = odl.phantom.shepp_logan(reco_space,
                                            True)/ 0.2 * 0.182
            return reco_space, f
        elif self.PH == "Defrise":
            # The defrise phantom is not a realistic phantom.
            # Therefore we must make assumptions such that it is a realistic phantom.
            # Let us set a diameter of 6 cm
            self.volumesize = np.array([5, 5, 5], dtype='float32')
            # Scale the detector correctly
            self.detecsize = np.array([2 * self.volumesize[0], 
                                       self.volumesize[1]])
            # Make the reconstruction space
            reco_space = odl.uniform_discr(min_pt=-self.volumesize,
                                           max_pt=self.volumesize,
                                            shape=voxels, dtype='float32')
            # Create the phantom:
            # We will set the ellipsoids to be plastic.
            # The plastic has density of +- 0.9 g/cm^3,
            # mass att. coeff = 0.27, --> att. coeff = 0.24
            f = odl.phantom.defrise(reco_space) * 0.24
            return reco_space, f
        elif self.PH == "Derenzo":
            # The derenzo phantom is 22 cm in diameter
            self.volumesize = np.array([4, 4, 4], dtype='float32')
            # Scale the detector correctly
            self.detecsize = np.array([2 * self.volumesize[0],
                                       self.volumesize[1]])
            # Make the reconstruction space
            reco_space = odl.uniform_discr(min_pt=-self.volumesize,
                                           max_pt=self.volumesize,
                                            shape=voxels, dtype='float32')
            # Create a support for the phantom


            # Create the phantom:
            # Derenzo phantom is actually a PET/SPECT phantom, so the real objects
            # are not really made for CT experiments. Therefore we do as if the rods
            # plastic are made from plastic
            # The plastic has density of +- 0.9 g/cm^3,
            # mass att. coeff = 0.27, --> att. coeff = 0.24
            f = odl.phantom.derenzo_sources(reco_space) * 0.24
            return reco_space, f
        elif self.PH == "Hollow cube":
            # Size of the cube is approximate 8 cm
            self.volumesize = np.array([4, 4, 4], dtype='float32')
            # Scale the detector correctly
            self.detecsize = np.array([2 * self.volumesize[0],
                                       self.volumesize[1]])
            # Make the reconstruction space
            reco_space = odl.uniform_discr(min_pt=-self.volumesize,
                                           max_pt=self.volumesize,
                                            shape=voxels, dtype='float32')

            # Create the phantom:
            # The plastic has density of +- 0.9 g/cm^3,
            # mass att. coeff = 0.27, --> att. coeff = 0.24
            p = np.zeros((voxels[0], voxels[1], voxels[2]))
            p[int(voxels[0]*2/16):int(voxels[0]*14/16),
              int(voxels[1]*2/16):int(voxels[1]*14/16),
              int(voxels[2]*2/16):int(voxels[2]*14/16)] = 1
            p[int(voxels[0]*4/16):int(voxels[0]*12/16),
              int(voxels[1]*4/16):int(voxels[1]*12/16),
              int(voxels[2]*4/16):int(voxels[2]*12/16)] = 0
            f = reco_space.element(p) * 0.22
            return reco_space, f
            
        elif self.PH == 'Cube':
            # Size of the cube is approximate 1.5 cm
            self.volumesize = np.array([.8, .8, .8], dtype='float32')
            # Scale the detector correctly
            self.detecsize = np.array([2 * self.volumesize[0],
                                       self.volumesize[1]])
            # Make the reconstruction space
            reco_space = odl.uniform_discr(min_pt=-self.volumesize,
                                           max_pt=self.volumesize,
                                            shape=voxels, dtype='float32')
            # Create the phantom:
            # The wooden block has density of +- 1 g/cm^3 --> att. coeff = 0.22
            p = np.zeros((voxels[0], voxels[1], voxels[2]))
            p[int(voxels[0]*4/16):int(voxels[0]*12/16),
              int(voxels[1]*4/16):int(voxels[1]*12/16),
              int(voxels[2]*4/16):int(voxels[2]*12/16)] = 1
            f = reco_space.element(p) * 0.22
            return reco_space, f
        
        elif self.PH == 'Cluttered sphere':
            # Size of the cube is approximate 8 cm
            self.volumesize = np.array([4, 4, 4], dtype='float32')
            # Scale the detector correctly
            self.detecsize = np.array([2 * self.volumesize[0],
                                       self.volumesize[1]])
            # Make the reconstruction space
            reco_space = odl.uniform_discr(min_pt=-self.volumesize,
                                           max_pt=self.volumesize,
                                            shape=voxels, dtype='float32')
            outer = [1, 0.75,0.75,0.75, 0,0,0, 0,0,0]
            inner1 = [-0.3, .6,.4,.2, 0.2,.2,.2,  0.5,0.25,0]
            inner2 = [-0.3, .22,.33,.44,- 0.2,-.2,-.2,  -0.75,-.1,.3]
            phantom = (odl.phantom.geometric.ellipsoid_phantom(reco_space,
                                                              [outer, inner1, inner2]))
            plus = np.zeros(np.shape(reco_space))

            plus[int(voxels[0]/4):int(voxels[0]/2)+2,
                 int(voxels[0]/4):int(voxels[0]/2)+2,
                 int(3*voxels[0]/4):int(3*voxels[0]/4)+4] = -.44
            plus[int(voxels[0]/4):int(voxels[0]/2)+2,
                 int(3*voxels[0]/8):int(3*voxels[0]/8)+4,
                 int(5*voxels[0]/8):int(6.5*voxels[0]/8)] = -.44
            plus[int(3*voxels[0]/8):int(3*voxels[0]/8)+4,
                 int(voxels[0]/4):int(voxels[0]/2)+2,
                 int(5*voxels[0]/8):int(6.5*voxels[0]/8)] = -.44
            plus[int(5*voxels[0]/8):int(5*voxels[0]/8)+4,
                 int(3*voxels[0]/8):int(3*voxels[0]/8)+4,
                 int(4*voxels[0]/8)-2:int(4*voxels[0]/8)+2] = -.5
            plus[int(5*voxels[0]/8)+6:int(5*voxels[0]/8)+10,
                 int(3*voxels[0]/8)+6:int(3*voxels[0]/8)+10,
                 int(4*voxels[0]/8)-2:int(4*voxels[0]/8)+2] = -.5
            plus[int(5*voxels[0]/8)+12:int(5*voxels[0]/8)+16,
                 int(3*voxels[0]/8)+12:int(3*voxels[0]/8)+16,
                 int(4*voxels[0]/8)-2:int(4*voxels[0]/8)+2] = -.5
            plus[int(5*voxels[0]/8)+18:int(5*voxels[0]/8)+22,
                 int(3*voxels[0]/8)+18:int(3*voxels[0]/8)+22,
                 int(4*voxels[0]/8)-2:int(4*voxels[0]/8)+2] = -.5
            plus[int(5*voxels[0]/8)+24:int(5*voxels[0]/8)+28,
                 int(3*voxels[0]/8)+24:int(3*voxels[0]/8)+28,
                 int(4*voxels[0]/8)-2:int(4*voxels[0]/8)+2] = -.5
            plus[int(5*voxels[0]/8)-8:int(5*voxels[0]/8)-4,
                 int(3*voxels[0]/8)-8:int(3*voxels[0]/8)-4,
                 int(4*voxels[0]/8)-2:int(4*voxels[0]/8)+2] = -.5
            plus[int(5*voxels[0]/8)-14:int(5*voxels[0]/8)-10,
                 int(3*voxels[0]/8)-14:int(3*voxels[0]/8)-10,
                 int(4*voxels[0]/8)-2:int(4*voxels[0]/8)+2] = -.5
            plus[int(5*voxels[0]/8)-20:int(5*voxels[0]/8)-16,
                 int(3*voxels[0]/8)-20:int(3*voxels[0]/8)-16,
                 int(4*voxels[0]/8)-2:int(4*voxels[0]/8)+2] = -.5
            plus[int(5*voxels[0]/8)-26:int(5*voxels[0]/8)-22,
                 int(3*voxels[0]/8)-26:int(3*voxels[0]/8)-22,
                 int(4*voxels[0]/8)-2:int(4*voxels[0]/8)+2] = -.5

            for i in range(int(voxels[0]/8)):
                plus[int(3*voxels[0]/8)+i, int(11*voxels[0]/16)-i:int(11*voxels[0]/16)+i,
                      int(8*voxels[0]/16)-i:int(8*voxels[0]/16)+i] = -(1-i*8/voxels[0])**3


            phantom = np.abs(phantom+plus)

            f = np.abs(phantom)*0.22
            return reco_space, f
        
        elif self.PH == 'Foam':
            self.volumesize = np.array([5, 5, 5], dtype='float32')
            # Scale the detector correctly
            self.detecsize = np.array([2 * self.volumesize[0],
                                       self.volumesize[1]])
            # Make the reconstruction space
            reco_space = odl.uniform_discr(min_pt=-self.volumesize,
                                           max_pt=self.volumesize,
                                            shape=voxels, dtype='float32')
            
            # ! ! ! TODO: fix a way to get this path to work automatically ! ! !
            path = '/export/scratch2/lagerwer/AFFDK_results/cone_foam/' + \
                    'cone_foam_spec.txt'

            ball_spec = np.loadtxt(path, dtype='float64')
            ball_pos = ball_spec[:, :3]
            ball_radius = ball_spec[:, 3]
            
            spheres = []
            # Add outer sphere
            spheres += [make_spheres([0, 0, 0], .8, 1)]
            # scale to phantom with diameter 80% of hte total size
            sc = 0.8 / 0.5
            # Add smaller spheres
            for i in range(len(ball_radius)):
                spheres += [make_spheres(ball_pos[i, :] * sc,
                                         ball_radius[i] * sc * 0.6)]

            # Create the foam phantom
            f = odl.phantom.ellipsoid_phantom(reco_space, spheres) * 0.24
            return reco_space, f

        elif self.PH == '22 Ellipses':
            if 'load_data_f' in kwargs:
                f = reco_space.element(np.load(kwargs['load_data_f']))
            else:
                if 'second' in kwargs:
                    seed_old = np.random.get_state()
                    np.random.set_state(self.seed)
                else:
                    self.seed = np.random.get_state()
            self.volumesize = np.array([4, 4, 4], dtype='float32')
            # Scale the detector correctly
            self.detecsize = np.array([2 * self.volumesize[0],
                                       self.volumesize[1]])
            # Make the reconstruction space
            reco_space = odl.uniform_discr(min_pt=-self.volumesize,
                                           max_pt=self.volumesize,
                                            shape=voxels, dtype='float32')

            if 'load_data_f' in kwargs:
                f = reco_space.element(np.load(kwargs['load_data_f']))
            else:
                ellipsoids = make_ellipsoids(22)
                phantom = odl.phantom.ellipsoid_phantom(reco_space,
                                                        ellipsoids,
                                                        min_pt=[-3.5, -3.5, -3.5],
                                                        max_pt=[3.5, 3.5, 3.5])
                f = phantom / np.max(phantom) * .22
            clip_cylinder(voxels[0], f)
            if 'second' in kwargs:
                np.random.set_state(seed_old)
            return reco_space, f

        elif self.PH == 'Threeshape':               
            self.volumesize = np.array([4, 4, 4], dtype='float32')
            # Scale the detector correctly
            self.detecsize = np.array([2 * self.volumesize[0],
                                       self.volumesize[1]])
            # Make the reconstruction space
            reco_space = odl.uniform_discr(min_pt=-self.volumesize,
                                           max_pt=self.volumesize,
                                            shape=voxels, dtype='float32')
            if 'load_data_f' in kwargs:
                f = reco_space.element(np.load(kwargs['load_data_f']))
            else:
                if 'second' in kwargs:
                    seed_old = np.random.get_state()
                    np.random.set_state(self.seed)
                else:
                    self.seed = np.random.get_state()

                f2d = po.addGaussianBlob2D(voxels[0])
                f2d += po.addGaussianBlob2D(voxels[0])
                f2d += po.addGaussianBlob2D(voxels[0])
                f2d += po.addRectangle2D(voxels[0])
                f2d += po.addRectangle2D(voxels[0])
                f2d += po.addRectangle2D(voxels[0])
                f2d += po.addSiemensStar2D(voxels[0])
                f2d += po.addSiemensStar2D(voxels[0])
                f2d += po.addSiemensStar2D(voxels[0])
                clip_circle(voxels[0], f2d)
                f2d = f2d / np.max(f2d) * .22
                f = reco_space.zero()

                for i in range(voxels[0] // 8, 7 * voxels[0] // 8):
                    f[:, :, i] = f2d

                clip_cylinder(voxels[0], f)
                if 'second' in kwargs:
                    np.random.set_state(seed_old)
            return reco_space, f
        
        elif self.PH == 'Fourshape':
            self.volumesize = np.array([4, 4, 4], dtype='float32')
            # Scale the detector correctly
            self.detecsize = np.array([2 * self.volumesize[0],
                                       self.volumesize[1]])
            # Make the reconstruction space
            reco_space = odl.uniform_discr(min_pt=-self.volumesize,
                                           max_pt=self.volumesize,
                                            shape=voxels, dtype='float32')
            if 'load_data_f' in kwargs:
                f = reco_space.element(np.load(kwargs['load_data_f']))
            else:
                if 'second' in kwargs:
                    seed_old = np.random.get_state()
                    np.random.set_state(self.seed)
                else:
                    self.seed = np.random.get_state()

                # Make three ellipsoids
                ellipsoids = make_ellipsoids(3)
                phantom = odl.phantom.ellipsoid_phantom(reco_space,
                                                        ellipsoids,
                                                        min_pt=[-3.5, -3.5, -3.5],
                                                        max_pt=[3.5, 3.5, 3.5])
                # Add three Gaussian blobs
                phantom += po.addGaussianBlob(voxels[0])
                phantom += po.addGaussianBlob(voxels[0])
                phantom += po.addGaussianBlob(voxels[0])
                # Add three rectangles
                phantom += po.addRectangle(voxels[0])
                phantom += po.addRectangle(voxels[0])
                phantom += po.addRectangle(voxels[0])
                # Add three Siemens stars
                phantom += po.addSiemensStar(voxels[0])
                phantom += po.addSiemensStar(voxels[0])
                phantom += po.addSiemensStar(voxels[0])
                clip_cylinder(voxels[0], (phantom))
                # Normalize phantom

                f = (phantom / np.max(phantom) * .22)
                if 'second' in kwargs:
                    np.random.set_state(seed_old)
            return reco_space, f
        
        
        elif self.PH == 'Fourshape_test':
            self.volumesize = np.array([4, 4, 4], dtype='float32')
            # Scale the detector correctly
            self.detecsize = np.array([2 * self.volumesize[0],
                                       self.volumesize[1]])
            # Make the reconstruction space
            reco_space = odl.uniform_discr(min_pt=-self.volumesize,
                                           max_pt=self.volumesize,
                                            shape=voxels, dtype='float32')
            pix = voxels[0]
            img = np.zeros((pix, pix, pix))
            img += po.addGaussianBlob_det(pix, x_c=0.4, y_c=0.6, z_c=0.44,
                                          a=.8, theta=0.44*np.pi,
                                          phi=0.05 * np.pi,
                                          x_s=.05, y_s=.15, z_s=.2)
            img += po.addGaussianBlob_det(pix, x_c=0.43, y_c=0.44, z_c=0.67,
                                          a=.55, theta=0.1*np.pi,
                                          phi=0.4 * np.pi, x_s=.1,
                                          y_s=.09, z_s=.0912)
            img += po.addGaussianBlob_det(pix, x_c=0.71, y_c=0.72, z_c=0.44,
                                          a=.13, theta=0.16*np.pi,
                                          phi=0.034 * np.pi,
                                          x_s=.084, y_s=.14, z_s=.078)
            
            img += po.addSiemensStar_det(pix, x_c=.5, y_c=.44, z_c=.5, a=.36,
                                         r2=.15, h2=.07, theta=0, phi=0)
            img += po.addSiemensStar_det(pix, x_c=.62, y_c=.49, z_c=.26, a=.24,
                                         r2=.12, h2=.125, theta=0,
                                         phi=0.55 * np.pi)
            img += po.addSiemensStar_det(pix, x_c=0.3, y_c=.33, z_c=.505,
                                         a=.34, r2=.075, h2=.1, 
                                         theta=0.44* np.pi, phi=0 * np.pi)
            
            
            img += po.addRectangle_det(pix, x_c=0.32, y_c=.70, z_c=.5, a=.2,
                                       w2=.3, h2=.05, d2=0.13,
                                       theta=0.25 * np.pi, phi=0)
            
            img += po.addRectangle_det(pix, x_c=0.5, y_c=.66, z_c=.32, a=.15,
                                       w2=.2, h2=.25, d2=0.06,
                                       theta=0.25 * np.pi, phi=0.6 * np.pi)
            
            img += po.addRectangle_det(pix, x_c=0.6, y_c=.5, z_c=.58, a=.19,
                                       w2=.8, h2=.02, d2=0.16,
                                       theta=0.55 * np.pi, phi=1.3 * np.pi)
            test_reco_space = odl.uniform_discr([0, 0, 0], [pix, pix, pix],
                                           [pix, pix, pix])
            a=.28
            x_s, y_s, z_s = .3, .4, .1
            x_c, y_c, z_c = 0.2, -0.5, 0
            rot1, rot2, rot3 = 0, 1.3, 0.25 * np.pi
            ellip = [[a, x_s, y_s, z_s, x_c, y_c, z_c, rot1, rot2, rot3]]
            phantom = odl.phantom.ellipsoid_phantom(test_reco_space, ellip,
                                                    min_pt=[.2 * pix, .2 * pix,
                                                            .2 * pix],
                                                    max_pt=[.8 * pix, .8 * pix,
                                                            .8 * pix])
            a=.18
            x_s, y_s, z_s=.2, .1, .35
            x_c, y_c, z_c=-.25, 0, .75
            rot1, rot2, rot3 = 0, 1.3, 0.25 * np.pi
            ellip = [[a, x_s, y_s, z_s, x_c, y_c, z_c, rot1, rot2, rot3]]
            phantom1 = odl.phantom.ellipsoid_phantom(test_reco_space, ellip,
                                                    min_pt=[.2 * pix, .2 * pix,
                                                            .2 * pix],
                                                    max_pt=[.8 * pix, .8 * pix,
                                                            .8 * pix])
            
            a=.23
            x_s, y_s, z_s=.6, .05, .2
            x_c, y_c, z_c=0, -.75, .6
            rot1, rot2, rot3 = 0, 1.3, 0.25 * np.pi
            ellip = [[a, x_s, y_s, z_s, x_c, y_c, z_c, rot1, rot2, rot3]]
            phantom2 = odl.phantom.ellipsoid_phantom(test_reco_space, ellip,
                                                    min_pt=[.2 * pix, .2 * pix,
                                                            .2 * pix],
                                                    max_pt=[.8 * pix, .8 * pix,
                                                            .8 * pix])
            
            
            img += phantom
            img += phantom1
            img += phantom2
            
            clip_cylinder(voxels[0], (img))
            # Normalize phantom

            f = (img / np.max(img) * .22)
            
            return reco_space, f
    
    
        elif self.PH == 'FORBILD':
            self.volumesize = np.array([15, 15, 15], dtype='float32')
            self.detecsize = np.array([2 * self.volumesize[0],
                                       self.volumesize[1]])
            # Make the reconstruction space
            reco_space = odl.uniform_discr(min_pt=-self.volumesize,
                                           max_pt=self.volumesize,
                                            shape=voxels, dtype='float32')
            volsize = self.volumesize[0]
            
            E5 = [1.8, 9.6 / volsize, 12 / volsize, 12.5 / volsize,
                  0, 0, 0, 0, 0, 0]
            E6 = [-.75, 9 / volsize, 11.4 / volsize, 11.9 / volsize,
                  0, 0, 0, 0, 0, 0]
            E12 = [-.005, 1.8 / volsize, 3.6 / volsize, 3.6 / volsize,
                   0, -3.6 / volsize, 0, 0, 0, 0]
            E7 = [-1.05, 1.8 / volsize, 3 / volsize, 3 / volsize,
                  0, 8.4 / volsize, 0, 0, 0, 0]
            
            S1 = [.01, 2 / volsize, 2 / volsize, 2 / volsize,
                  -4.7 / volsize, 4.3 / volsize, 0.872 / volsize,
                  0, 0, 0]
            S2 = [.01, 2 / volsize, 2 / volsize, 2 / volsize,
                  4.7 / volsize, 4.3 / volsize, 0.872 / volsize,
                  0, 0, 0]
            S3 = [.0025, .4 / volsize, .4 / volsize, .4 / volsize,
                  -1.08 / volsize, -9 / volsize, 0,
                  0, 0, 0]
            S4 = [-.0025, .4 / volsize, .4 / volsize, .4 / volsize,
                  1.08 / volsize, -9 / volsize, 0,
                  0, 0, 0]
            # Elipse 2D = 'value', 'x_r', 'y_r', 'center_x', 'center_y', 'rotation'
            
            E8 = [.75, -1.2 * np.cos(np.pi / 12) / volsize, 0.42 * np.cos(np.pi/ 12) / volsize,
                  3 / volsize, -1.9 / volsize, 5.4 / volsize, 0,
                   np.pi / 2, np.pi / 12, 7/6 * np.pi]
            E9 = [.75, -1.2 * np.cos(np.pi / 12) / volsize, 0.42 * np.cos(np.pi/ 12) / volsize,
                  3 / volsize, 1.9 / volsize, 5.4 / volsize, 0,
                   np.pi / 2, np.pi / 12, 5/6 * np.pi]
            
            E13 = [.005, 1.2 / volsize, 0.42 / volsize, 1.2 / volsize,
                   6.393945 / volsize, -6.393945 / volsize, 0, 
                   58.1/ 180 * np.pi, 0, 0]
            ER = [.75, 4.2 / volsize, 1.8 / volsize, 1.8 / volsize, .95, 0, 0, 0, 0, 0]
            # %%
            ph = odl.phantom.ellipsoid_phantom(reco_space, [E5, E6, E12, E7, S1, S2, S3, S4,
                                                           E8, E9, E13])
            ph += odl.phantom.ellipsoid_phantom(reco_space, [ER], 
                                        max_pt=(9.1, *self.volumesize[1:]))
            rad = .15 / volsize
            sha = 0.2 * np.sqrt(3) / volsize
            x_ear1 = np.array([8.8, 8.4, 8.0, 7.6, 7.2, 6.8, 6.4, 6.0, 5.6]) \
                    / volsize
            x_ear2 = np.array([8.6, 8.2, 7.8, 7.4, 7.0, 6.6, 6.2, 5.8]) \
                    / volsize
            SE1, SE2, SE3, SE4, SE5, SE6, SE7 = [], [], [], [], [], [], []
            for k in range(9):
                SE1 += [[-1.8, rad, rad, rad, x_ear1[k], 0, 0, 0, 0, 0]]
                if k < 8:
                    SE1 += [[-1.8, rad, rad, rad, x_ear2[k], 0, sha,
                             0, 0, 0]]
                    SE1 += [[-1.8, rad, rad, rad, x_ear1[k], 0, 2 * sha,
                             0, 0, 0]]
                    SE1 += [[-1.8, rad, rad, rad, x_ear2[k], 0, -sha,
                             0, 0, 0]]
                    SE1 += [[-1.8, rad, rad, rad, x_ear1[k], 0, 2 * -sha,
                             0, 0, 0]]
                    
                    SE2 += [[-1.8, rad, rad, rad, x_ear2[k], sha, 0,
                             0, 0, 0]]
                    SE2 += [[-1.8, rad, rad, rad, x_ear1[k], sha, sha,
                             0, 0, 0]]
                    SE2 += [[-1.8, rad, rad, rad, x_ear1[k], sha, -sha,
                             0, 0, 0]]
                    
                    SE3 += [[-1.8, rad, rad, rad, x_ear2[k], -sha, 0,
                             0, 0, 0]]
                    SE3 += [[-1.8, rad, rad, rad, x_ear1[k], -sha, sha,
                             0, 0, 0]]
                    SE3 += [[-1.8, rad, rad, rad, x_ear1[k], -sha, -sha,
                             0, 0, 0]]
                    
                    SE4 += [[-1.8, rad, rad, rad, x_ear1[k], 2 * sha, 0,
                             0, 0, 0]]
                    
                    SE5 += [[-1.8, rad, rad, rad, x_ear1[k], 2 * -sha, 0,
                             0, 0, 0]]
                if k < 7:
                    SE2 += [[-1.8, rad, rad, rad, x_ear2[k], sha, 2 * sha,
                             0, 0, 0]]
                    SE2 += [[-1.8, rad, rad, rad, x_ear2[k], sha, 2 * -sha,
                             0, 0, 0]]
                    
                    SE3 += [[-1.8, rad, rad, rad, x_ear2[k], -sha, 2 * sha,
                             0, 0, 0]]
                    SE3 += [[-1.8, rad, rad, rad, x_ear2[k], -sha, 2 * -sha,
                             0, 0, 0]]
                    
                    SE4 += [[-1.8, rad, rad, rad, x_ear1[k], 2 * sha, sha,
                             0, 0, 0]]
                    SE4 += [[-1.8, rad, rad, rad, x_ear1[k], 2 * sha, -sha,
                             0, 0, 0]]
                    SE4 += [[-1.8, rad, rad, rad, x_ear2[k], 2 * sha, 2 * sha,
                             0, 0, 0]]
                    SE4 += [[-1.8, rad, rad, rad, x_ear2[k], 2 * sha, 2 * -sha,
                             0, 0, 0]]
                    
                    SE5 += [[-1.8, rad, rad, rad, x_ear1[k], 2 * -sha, sha,
                             0, 0, 0]]
                    SE5 += [[-1.8, rad, rad, rad, x_ear1[k], 2 * -sha, -sha,
                             0, 0, 0]]
                    SE5 += [[-1.8, rad, rad, rad, x_ear2[k], 2 * -sha, 2 * sha,
                             0, 0, 0]]
                    SE5 += [[-1.8, rad, rad, rad, x_ear2[k], 2 * -sha, 2 * -sha,
                             0, 0, 0]]
                if k < 6:
                    SE1 += [[-1.8, rad, rad, rad, x_ear2[k], 0, 3 * sha,
                             0, 0, 0]]
                    SE1 += [[-1.8, rad, rad, rad, x_ear1[k], 0, 3 * -sha,
                             0, 0, 0]]
                    SE1 += [[-1.8, rad, rad, rad, x_ear1[k], 0, 0,
                             0, 0, 0]]
                    
                    SE2 += [[-1.8, rad, rad, rad, x_ear1[k], sha, 3 * sha,
                             0, 0, 0]]
                    SE2 += [[-1.8, rad, rad, rad, x_ear1[k], sha, 3 * -sha,
                             0, 0, 0]]
                    SE3 += [[-1.8, rad, rad, rad, x_ear1[k], -sha, 3 * sha,
                             0, 0, 0]]
                    SE3 += [[-1.8, rad, rad, rad, x_ear1[k], -sha, 3 * -sha,
                             0, 0, 0]]
                    
                    SE6 += [[-1.8, rad, rad, rad, x_ear1[k], 3 * sha, 0,
                             0, 0, 0]]
                    SE6 += [[-1.8, rad, rad, rad, x_ear1[k], 3 * sha, sha,
                             0, 0, 0]]
                    SE6 += [[-1.8, rad, rad, rad, x_ear1[k], 3 * sha, -sha,
                             0, 0, 0]]
                    
                    SE7 += [[-1.8, rad, rad, rad, x_ear1[k], 3 * -sha, 0,
                             0, 0, 0]]
                    SE7 += [[-1.8, rad, rad, rad, x_ear1[k], 3 * -sha, sha,
                             0, 0, 0]]
                    SE7 += [[-1.8, rad, rad, rad, x_ear1[k], 3 * -sha, -sha, 
                             0, 0, 0]]
                if k < 5:
                    SE4 += [[-1.8, rad, rad, rad, x_ear1[k], 2 * sha, 3 * sha,
                             0, 0, 0]]
                    SE4 += [[-1.8, rad, rad, rad, x_ear1[k], 2 * sha, 3 * -sha,
                             0, 0, 0]]
                    
                    SE5 += [[-1.8, rad, rad, rad, x_ear1[k], 2 * -sha, 3 * sha,
                             0, 0, 0]]
                    SE5 += [[-1.8, rad, rad, rad, x_ear1[k], 2 * -sha, 3 * -sha,
                             0, 0, 0]]
                    
                    SE6 += [[-1.8, rad, rad, rad, x_ear1[k], 3 * sha, 2 * sha,
                             0, 0, 0]]
                    SE6 += [[-1.8, rad, rad, rad, x_ear1[k], 3 * sha, 2 * -sha,
                             0, 0, 0]]
                    
                    SE7 += [[-1.8, rad, rad, rad, x_ear1[k], 3 * -sha, 2 * sha,
                             0, 0, 0]]
                    SE7 += [[-1.8, rad, rad, rad, x_ear1[k], 3 * -sha, 2 * -sha,
                             0, 0, 0]]
                if k < 4:
                    SE6 += [[-1.8, rad, rad, rad, x_ear1[k], 3 * sha, 3 * sha,
                             0, 0, 0]]
                    SE6 += [[-1.8, rad, rad, rad, x_ear1[k], 3 * sha, 3 * -sha,
                             0, 0, 0]]
                    
                    SE7 += [[-1.8, rad, rad, rad, x_ear1[k], 3 * -sha, 3 * sha,
                             0, 0, 0]]
                    SE7 += [[-1.8, rad, rad, rad, x_ear1[k], 3 * -sha, 3 * -sha,
                             0, 0, 0]]
                    
            ph += odl.phantom.ellipsoid_phantom(reco_space, [*SE1, *SE2, *SE3,
                                                             *SE4, *SE5,
                                                             *SE6, *SE7])
            # %% Make some bones with a for loop
            ph += po.addBonesFORBILD(voxels[0], volsize)
                        
            ph[np.asarray(ph) > 1.8] = 1.8
            ph[np.asarray(ph) < 0] = 0
            return reco_space, (ph * 0.1)
        elif self.PH == 'Low contrast':
            self.volumesize = np.array([12, 12, 12], dtype='float32')
            self.detecsize = np.array([2 * self.volumesize[0],
                                       self.volumesize[1]])
            # Make the reconstruction space
            reco_space = odl.uniform_discr(min_pt=-self.volumesize,
                                           max_pt=self.volumesize,
                                            shape=voxels, dtype='float32')

            A = np.array([87.708, 108.3346, 126.6693, 142.7121, 156.4631, 167.9223,
                          177.0896, 183.9651, 188.5487]) / 180 *  np.pi
            B = np.array([110.6265, 142.7121, 165.6304, 179.3814]) / 180 * np.pi
            
            size_xy = self.volumesize[0]
            Rfull = 10 / size_xy
            Rout = 5 / size_xy
            Rin =  2.5 / size_xy
            rout = np.array([.75, .45, .40, .35, .30, .25, .20, .15, .10]) / size_xy
            rin = np.array([.45, .35, .25, .15]) / size_xy
            
            v0 = 1
            v1 = 0.05
            v2 = 0.15
            v3 = 0.3
            # Elipse 2D = 'value', 'r', 'r', 'center_x', 'center_y', 'rotation'
            
            C0 = [v0, Rfull, Rfull, 0, 0, 0]
            C1, C2, C3 = [], [], [] 
            c1, c2, c3 = [], [], []
            
            for i in range(len(rout)):
                C1 += [[v1, rout[i], rout[i], Rout * np.cos(A[i]), Rout * np.sin(A[i]), 0]]
                C2 += [[v2, rout[i], rout[i], Rout * np.cos(A[i] + 2/3 * np.pi),
                        Rout * np.sin(A[i] + 2/3 * np.pi), 0]]
                C2 += [[v3, rout[i], rout[i], Rout * np.cos(A[i] + 4/3 * np.pi),
                        Rout * np.sin(A[i] + 4/3 * np.pi), 0]]
            for i in range(len(rin)):   
                c1 += [[v1, rin[i], rin[i], Rin * np.cos(B[i]), Rin * np.sin(B[i]), 0]]
                c2 += [[v1, rin[i], rin[i], Rin * np.cos(B[i] + 2/3 * np.pi),
                        Rin * np.sin(B[i] + 2/3 * np.pi), 0]]
                c3 += [[v1, rin[i], rin[i], Rin * np.cos(B[i] + 4/3 * np.pi),
                        Rin * np.sin(B[i] + 4/3 * np.pi), 0]]
            # %%
            ph = odl.phantom.ellipsoid_phantom(reco_space,
                    odl.phantom.phantom_utils.cylinders_from_ellipses([C0, *C1, *C2, *C3]),
                    min_pt=[-12, -12, -10], max_pt=[12, 12, 10])
            
            ph += odl.phantom.ellipsoid_phantom(reco_space,
                    odl.phantom.phantom_utils.cylinders_from_ellipses(c1),
                    min_pt=[-12, -12, -.7/ 2], max_pt=[12, 12, .7 /2])
            
            ph += odl.phantom.ellipsoid_phantom(reco_space,
                    odl.phantom.phantom_utils.cylinders_from_ellipses(c2),
                    min_pt=[-12, -12, -.3/2], max_pt=[12, 12, .3/2 ])
            
            ph += odl.phantom.ellipsoid_phantom(reco_space,
                    odl.phantom.phantom_utils.cylinders_from_ellipses(c3),
                    min_pt=[-12, -12, -.5/2], max_pt=[12, 12, .5/2 ])
            return reco_space, ph * 0.1
        
        elif self.PH == 'Delta':
            self.volumesize = np.array([12, 12, 12], dtype='float32')
            self.detecsize = np.array([2 * self.volumesize[0],
                                       self.volumesize[1]])
            # Make the reconstruction space
            reco_space = odl.uniform_discr(min_pt=-self.volumesize,
                                           max_pt=self.volumesize,
                                            shape=voxels, dtype='float32')
            f = reco_space.element(po.addDelta(voxels[0], a=1, sigma=1))
            return reco_space, f


        elif self.PH == 'Plane_yz':
            x = kwargs['offset_x']
            self.volumesize = np.array([12, 12, 12], dtype='float32')
            self.detecsize = np.array([2 * self.volumesize[0],
                                       self.volumesize[1]])
            # Make the reconstruction space
            reco_space = odl.uniform_discr(min_pt=-self.volumesize,
                                           max_pt=self.volumesize,
                                            shape=voxels, dtype='float32')
            f = np.zeros(voxels)
            mid = voxels[0] // 2
            f[mid + x, :, :] = 1
            f = reco_space.element(f)
            return reco_space, f

        elif type(self.PH) == dict:
            self.volumesize = np.array(self.PH['voxelsize'],
                                       dtype='float32')
            self.detecsize = np.array([2 * self.volumesize[0],
                                       self.volumesize[1]])
            reco_space = odl.uniform_discr(min_pt=-self.volumesize,
                                                max_pt=self.volumesize,
                                                shape=voxels, dtype='float32')
            f = reco_space.element(self.PH['PH'])
            return reco_space, f
        else:
            raise ValueError('unknown `Phantom name` ({})'
                             ''.format(self.PH))



