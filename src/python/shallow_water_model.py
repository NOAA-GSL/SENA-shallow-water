from mpi4py import MPI
import numpy as np
import gt4py.gtscript as gtscript
import gt4py.storage as gt_storage

from shallow_water_model_config import ShallowWaterModelConfig
from shallow_water_geometry import ShallowWaterGeometry
from shallow_water_state import ShallowWaterState


class ShallowWaterModel:

    backend = "numpy"
    F_TYPE = np.float64
    I_TYPE = np.int32
    # Use a 2D Field for the GT4Py Stencils
    FloatFieldIJ = gtscript.Field[gtscript.IJ, np.float64]

    def __init__(self, config : ShallowWaterModelConfig, geometry : ShallowWaterGeometry):
     
        self.config = config 
        self.geometry = geometry
        self.dt = config.dt

        # Initialize b (currently unused)
        self.b = gt_storage.zeros(shape=(self.geometry.xme - self.geometry.xms + 1, self.geometry.yme - self.geometry.yms + 1),
                                  dtype=self.F_TYPE, backend=self.backend, default_origin=(1,1)) 


    def adv_nsteps(self, state : ShallowWaterState, nsteps: np.int32):

        _dx = self.geometry.dx
        _dy = self.geometry.dy

        _dtdx = self.dt / _dx
        _dtdy = self.dt / _dy
        
        # Define gravitational constant
        _g  = 9.81

        # Sanity check for time step
        if(state.max_wavespeed > 0.0):
            _maxdt = .68 * min(_dx,_dy) / state.max_wavespeed
            if (self.dt > _maxdt):
                print("WARNING: time step is too large, should be <= ", _maxdt)
    
        # Get local bounds including halo
        _xms = self.geometry.xms
        _xme = self.geometry.xme
        _yms = self.geometry.yms
        _yme = self.geometry.yme
        
        # Get local bounds exluding the halo
        _xps = self.geometry.xps
        _xpe = self.geometry.xpe
        _yps = self.geometry.yps
        _ype = self.geometry.ype

        # Get local interior points 
        _xts = self.geometry.xts
        _xte = self.geometry.xte
        _yts = self.geometry.yts
        _yte = self.geometry.yte

        _u_new = gt_storage.zeros(shape=(_xme - _xms + 1, _yme - _yms + 1), backend=self.backend, default_origin=(1,1), dtype=self.F_TYPE )
        _v_new = gt_storage.zeros(shape=(_xme - _xms + 1, _yme - _yms + 1), backend=self.backend, default_origin=(1,1), dtype=self.F_TYPE )
        _h_new = gt_storage.zeros(shape=(_xme - _xms + 1, _yme - _yms + 1), backend=self.backend, default_origin=(1,1), dtype=self.F_TYPE )

        _b_gt = gt_storage.from_array(self.b, state.backend, default_origin=(0,0,0))

        # Move the model state n steps into the future
        for n in range(nsteps):

            # Exchange halos
            state.exchange_halo()   

            # Update the domain boundaries            
            self.update_boundaries(south  = self.geometry.south,
                                   north  = self.geometry.north,
                                   west   = self.geometry.west,
                                   east   = self.geometry.east,
                                   u      = state.u,
                                   v      = state.v,
                                   h      = state.h,
                                   u_new  = _u_new,
                                   v_new  = _v_new,
                                   h_new  = _h_new,
                                   origin = (0, 0),
                                   domain = (_xme - _xms + 1, _yme - _yms + 1, 1))

            # Update the domain interior
            self.update_interior(u      = state.u,
                                 v      = state.v,
                                 h      = state.h,
                                 b      = self.b,
                                 u_new  = _u_new,
                                 v_new  = _v_new,
                                 h_new  = _h_new,
                                 dtdx   = _dtdx,
                                 dtdy   = _dtdy,
                                 g      = _g,
                                 origin = (1, 1),
                                 domain = (_xte - _xts + 1, _yte - _yts + 1, 1))

            # Update state with new state
            for i in range(_xps - _xms, _xpe - _xms + 1):
                for j in range(_yps - _yms, _ype - _yms + 1):
                    state.u[i,j] = _u_new[i,j]
                    state.v[i,j] = _v_new[i,j]
                    state.h[i,j] = _h_new[i,j]

            # Update the model clock and step counter
            state.advance_clock(self.dt)


    # Get model state one step in the future for the domain interior
    @gtscript.stencil(backend=backend)
    def update_interior(u:     FloatFieldIJ,
                        v:     FloatFieldIJ,
                        h:     FloatFieldIJ,
                        b:     FloatFieldIJ,
                        u_new: FloatFieldIJ,
                        v_new: FloatFieldIJ,
                        h_new: FloatFieldIJ,
                        dtdx:  F_TYPE,
                        dtdy:  F_TYPE,
                        g:     F_TYPE):

        with computation(FORWARD), interval(...):
            u_new = ((u[1,0] + u[-1,0] + u[0,1] + u[0,-1]) / 4.0)                    \
                            - 0.5 * dtdx * ((u[1,0]**2) / 2.0 - (u[-1,0]**2) / 2.0)  \
                            - 0.5 * dtdy * (v[0,0]) * (u[0,1] - u[0,-1])             \
                            - 0.5 * g * dtdx * (h[1,0] - h[-1,0])
            v_new = ((v[1,0] + v[-1,0] + v[0,1] + v[0,-1]) / 4.0)                    \
                            - 0.5 * dtdy * ((v[0,1]**2) / 2.0 - (v[0,1]**2) / 2.0)   \
                            - 0.5 * dtdx * (u[0,0]) * (v[1,0] - v[-1,0])             \
                            - 0.5 * g * dtdy * (h[0,1] - h[0,-1])
            h_new = ((h[1,0] + h[-1,0] + h[0,1] + h[0,-1]) / 4.0)                                \
                            - 0.5 * dtdx * (u[0,0]) * ((h[1,0] - b[1,0]) - (h[-1,0] - b[-1,0]))  \
                            - 0.5 * dtdy * (v[0,0]) * ((h[0,1] - b[0,1]) - (h[0,-1] - b[0,-1]))  \
                            - 0.5 * dtdx * (h[0,0] - b[0,0]) * (u[1,0] - u[-1,0])                \
                            - 0.5 * dtdy * (h[0,0] - b[0,0]) * (v[0,1] - v[0,-1])

    # Get model state one step in the future for the domain boundaries
    @gtscript.stencil(backend=backend)
    def update_boundaries(north :  I_TYPE,
                          south :  I_TYPE,
                          east  :  I_TYPE,
                          west  :  I_TYPE,
                          u     :  FloatFieldIJ,
                          v     :  FloatFieldIJ,
                          h     :  FloatFieldIJ,
                          u_new :  FloatFieldIJ,
                          v_new :  FloatFieldIJ,
                          h_new :  FloatFieldIJ):

        with computation(FORWARD), interval(...):
            # Update southern boundary
            with horizontal(region[:, J[0]]):
                if (south == -1): 
                    h_new =  h[0,1]
                    u_new =  u[0,1]
                    v_new = -v[0,1]

            # Update northern boundary
            with horizontal(region[:, J[-1]]):
                if (north == -1):
                    h_new =  h[0,-1] 
                    u_new =  u[0,-1] 
                    v_new = -v[0,-1] 

            # Update western boundary
            with horizontal(region[I[0], :]):
                if (west == -1):
                    h_new =  h[1,0]
                    u_new = -u[1,0]
                    v_new =  v[1,0]

            # Update eastern boundary
            with horizontal(region[I[-1], :]):
                if (east == -1):
                    h_new =  h[-1,0]
                    u_new = -u[-1,0]
                    v_new =  v[-1,0]

