#!/usr/bin/env python

from shallow_water_geometry_config import ShallowWaterGeometryConfig
from shallow_water_model_config import ShallowWaterModelConfig
from shallow_water_geometry import ShallowWaterGeometry
from shallow_water_state import ShallowWaterState
import numpy as np
from mpi4py import MPI
from matplotlib import pyplot as plt, animation
#import sys
#import os
#import serialbox as ser
import gt4py.gtscript as gtscript
import gt4py.storage as gt_storage
#from gt4py.gtscript import stencil, function, PARALLEL, computation, interval, Field, parallel, region, I, J, K

backend="numpy"
F_TYPE = np.float64
#I_TYPE = np.int32
I_TYPE = int
FloatFieldIJ = gtscript.Field[gtscript.IJ, F_TYPE]

# Define constants
g = 9.81

class ShallowWaterModel:

    def __init__(self, config, geometry):

        self.config = config
        self.geometry = geometry
        self.dt = self.config.dt

        self.b = gt_storage.zeros(shape=(self.geometry.xme - self.geometry.xms + 1, self.geometry.yme - self.geometry.yms + 1),
                                  dtype=F_TYPE, backend=backend, default_origin=(1,1)) 

    def adv_nsteps(self, state, nsteps):

        # Get grid spacing
        _dx = self.geometry.dx
        _dy = self.geometry.dy

        # Sanity check for timestep
        if (state.max_wavespeed > 0.0):
            _maxdt = 0.68 * min(_dx, _dy) / state.max_wavespeed
            if (self.dt > _maxdt):
                print(f"WARNING: time step is too large, should be <= {_maxdt}")

        # Compute dxdt and dydt
        _dtdx = self.dt / _dx
        _dtdy = self.dt / _dy

        # Set up halo, origin, and domain
        _nhalo=1
        _nx = self.geometry.xte - self.geometry.xts + 1
        _ny = self.geometry.yte - self.geometry.yts + 1
        _origin=(_nhalo, _nhalo)
        _domain=(_nx, _ny, 1)

        # Create gt4py storages for the new model state
        _u_new = gt_storage.empty(shape=(self.geometry.xme - self.geometry.xms + 1, self.geometry.yme - self.geometry.yms + 1),
                                   dtype=F_TYPE, backend=backend, default_origin=_origin)
        _v_new = gt_storage.empty(shape=(self.geometry.xme - self.geometry.xms + 1, self.geometry.yme - self.geometry.yms + 1),
                                   dtype=F_TYPE, backend=backend, default_origin=_origin)
        _h_new = gt_storage.empty(shape=(self.geometry.xme - self.geometry.xms + 1, self.geometry.yme - self.geometry.yms + 1),
                                   dtype=F_TYPE, backend=backend, default_origin=_origin)

        for n in range(nsteps):
            # Exchange halo
            state.exchange_halo()

            # Get new boundaries
            boundary_update(u=state.u,
                            v=state.v,
                            h=state.h,
                            u_new=_u_new,
                            v_new=_v_new,
                            h_new=_h_new,
                            north=self.geometry.north,
                            south=self.geometry.south,
                            west=self.geometry.west,
                            east=self.geometry.east,
                            dtdx=_dtdx,
                            dtdy=_dtdy,
                            origin=(0,0,0),
                            domain=(self.geometry.xme - self.geometry.xms + 1, self.geometry.yme - self.geometry.yms + 1, 1))

            # Get new interior points
            interior_update(u=state.u,
                            v=state.v,
                            h=state.h,
                            b=self.b,
                            u_new=_u_new,
                            v_new=_v_new,
                            h_new=_h_new,
                            dtdx=_dtdx,
                            dtdy=_dtdy,
                            origin=_origin,
                            domain=_domain)

            # Update state with new state
            for i in range(self.geometry.xps - self.geometry.xms, self.geometry.xpe - self.geometry.xms + 1):
                for j in range(self.geometry.yps - self.geometry.yms, self.geometry.ype - self.geometry.yms + 1):
                    state.u[i,j] = _u_new[i,j]
                    state.v[i,j] = _v_new[i,j]
                    state.h[i,j] = _h_new[i,j]


@gtscript.stencil(backend=backend)
def boundary_update(u     : FloatFieldIJ,
                    v     : FloatFieldIJ,
                    h     : FloatFieldIJ,
                    u_new : FloatFieldIJ,
                    v_new : FloatFieldIJ,
                    h_new : FloatFieldIJ,
                    north : I_TYPE,
                    south : I_TYPE,
                    west  : I_TYPE,
                    east  : I_TYPE,
                    dtdx  : F_TYPE,
                    dtdy  : F_TYPE):
    # NOTE: FORWARD ordering is required here to disambiguate the missing k dimension
    #       for assignment into our 2D arrays.
    # NOTE: Requires origin/domain set such that I[0], I[-1], J[0], J[-1] are the
    #       first/last elements of the compute domain (which they are in the original
    #       Fortran).
    with computation(FORWARD), interval(...):
        # Update southern boundary if there is one
        with horizontal(region[:, J[0]]):
            if (south == -1):
                h_new = h[0, 1]
                u_new = u[0, 1]
                v_new = -v[0, 1]

        # Update northern boundary if there is one
        with horizontal(region[:, J[-1]]):
            if (north == -1):
                h_new = h[0,-1]
                u_new = u[0,-1]
                v_new = -v[0,-1]

        # Update western boundary if there is one
        with horizontal(region[I[0], :]):
            if (west == -1):
                h_new = h[1,0]
                u_new = -u[1,0]
                v_new = v[1,0]

        # Update eastern boundary if there is one
        with horizontal(region[I[-1], :]):
            if (east == -1):
                h_new = h[-1,0]
                u_new = -u[-1,0]
                v_new = v[-1,0]


@gtscript.stencil(backend=backend)
def interior_update(u     : FloatFieldIJ,
                    v     : FloatFieldIJ,
                    h     : FloatFieldIJ,
                    b     : FloatFieldIJ,
                    u_new : FloatFieldIJ,
                    v_new : FloatFieldIJ,
                    h_new : FloatFieldIJ,
                    dtdx  : F_TYPE,
                    dtdy  : F_TYPE):
    # NOTE: FORWARD ordering is required here to disambiguate the missing k dimension
    #       for assignment into our 2D arrays.
    with computation(FORWARD), interval(...):
        u_new = ((u[1,0] + u[-1,0] + u[0,1] + u[0,-1]) / 4.0)           \
                - 0.5 * dtdx * ((u[1,0]**2) / 2.0 - (u[-1,0]**2) / 2.0) \
                - 0.5 * dtdy * v * (u[0,1] - u[0,-1])                   \
                - 0.5 * g * dtdx * (h[1,0] - h[-1,0])

        v_new = ((v[1,0] + v[-1,0] + v[0,1] + v[0,-1]) / 4.0)           \
                - 0.5 * dtdy * ((v[0,1]**2) / 2.0 - (v[0,1]**2) / 2.0)  \
                - 0.5 * dtdx * u * (v[1,0] - v[-1,0])                   \
                - 0.5 * g * dtdy * (h[0,1] - h[0,-1])
  
        h_new = ((h[1,0] + h[-1,0] + h[0,1] + h[0,-1]) / 4.0)                \
                - 0.5 * dtdx * u * ((h[1,0] - b[1,0]) - (h[-1,0] - b[-1,0])) \
                - 0.5 * dtdy * v * ((h[0,1] - b[0,1]) - (h[0,-1] - b[0,-1])) \
                - 0.5 * dtdx * (h - b) * (u[1,0] - u[-1,0])                  \
                - 0.5 * dtdy * (h - b) * (v[0,1] - v[0,-1])


#comm = MPI.COMM_WORLD
#gc = ShallowWaterGeometryConfig.from_YAML_filename('shallow_water.yml')
#mc = ShallowWaterModelConfig.from_YAML_filename('shallow_water.yml')
#g = ShallowWaterGeometry(gc, comm)
#m = ShallowWaterModel(mc, g)
#
## Create a state with a tsunami pulse in it to initialize field h
#h = np.empty((g.npx, g.npy), dtype=float)
#xmid = g.xmax / 2.0
#ymid = g.ymax / 2.0
#sigma = np.floor(g.xmax / 20.0)
#for i in range(g.xps, g.xpe + 1):
#    for j in range(g.yps, g.ype + 1):
#        dsqr = (i * g.dx - xmid)**2 + (j * g.dy - ymid)**2
#        h[i - g.xps,j - g.yps] = 5000.0 + np.exp(-dsqr / sigma**2) * (mc.h0 - 5000.0)
#s = ShallowWaterState(g, h=h)
#
## Write the initial state
#s.write("state_0.nc")
#
## Advance the model state
#m.adv_nsteps(s, 200)
#
## Write the final state
#s.write("state_200.nc")
#
#def animate(n):
#    m.adv_nsteps(s, 1)
#    pc.set_array(s.h)
#
## Plot animation
#X, Y = np.meshgrid(np.linspace(0, g.xmax, g.npx), np.linspace(0, g.ymax, g.npy))
#fig, axs = plt.subplots()
#pc = axs.pcolormesh(X, Y, s.h, vmin=4990, vmax=5030, cmap='nipy_spectral')
#fig.colorbar(pc, ax=axs)
#fig.savefig('test0.png')
#anim = animation.FuncAnimation(fig, animate, interval=125, frames=500)
#anim.save('test.gif')
#fig.savefig('test500.png') 

