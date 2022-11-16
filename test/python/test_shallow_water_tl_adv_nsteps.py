#!/usr/bin/env python

import sys
import numpy as np
from mpi4py import MPI
import gt4py.gtscript as gtscript
import gt4py.storage as gt_storage

sys.path.append("../../src/python")

from shallow_water_geometry_config import ShallowWaterGeometryConfig
from shallow_water_geometry import ShallowWaterGeometry
from shallow_water_model_config import ShallowWaterModelConfig
from shallow_water_state import ShallowWaterState
from shallow_water_model import ShallowWaterModel


# Config Parameters
_nx = 101
_ny = 101
_xmax = 100000.0
_ymax = 100000.0
_u0 = 0.0
_v0 = 0.0
_b0 = 0.0
_h0 = 5030.0
_g = 9.81
_dt = 0.8

# Test parameters
_spinup_steps = 1000
_digits = 8

# Initialize lambda (scaling factor)
_lambda = 1.0

# Start MPI
comm = MPI.COMM_WORLD
# Get our MPI rank
_myrank = comm.Get_rank()
# Get number of MPI ranks
_nranks = comm.Get_size()

# Initialize error count to 0
_errors = 0

# Create a geometry configuration as specified by Config Parameters
geometry_config = ShallowWaterGeometryConfig(nx=_nx, ny=_ny, xmax=_xmax, ymax=_ymax)
# Create a geometry from configuration
geometry = ShallowWaterGeometry(geometry=geometry_config, mpi_comm=comm)

# Get index ranges from geometry
_xps = geometry.xps
_xpe = geometry.xpe
_yps = geometry.yps
_ype = geometry.ype

_npx = geometry.npx
_npy = geometry.npy

# Create a model configuration
model_config = ShallowWaterModelConfig(dt=_dt, u0=_u0, v0=_v0, b0=_b0, h0=_h0)

# Allocate space for model states
_u = np.zeros((_npx, _npy), dtype=np.float64)
_v = np.zeros((_npx, _npy), dtype=np.float64)
_h = np.zeros((_npx, _npy), dtype=np.float64)
_udelta        = np.zeros((_npx, _npy), dtype=np.float64)
_vdelta        = np.zeros((_npx, _npy), dtype=np.float64)
_hdelta        = np.zeros((_npx, _npy), dtype=np.float64)
_mu            = np.zeros((_npx, _npy), dtype=np.float64)
_mv            = np.zeros((_npx, _npy), dtype=np.float64)
_mh            = np.zeros((_npx, _npy), dtype=np.float64)
_m_udelta      = np.zeros((_npx, _npy), dtype=np.float64)
_m_vdelta      = np.zeros((_npx, _npy), dtype=np.float64)
_m_hdelta      = np.zeros((_npx, _npy), dtype=np.float64)
_mprime_udelta = np.zeros((_npx, _npy), dtype=np.float64)
_mprime_vdelta = np.zeros((_npx, _npy), dtype=np.float64)
_mprime_hdelta = np.zeros((_npx, _npy), dtype=np.float64)


# Create a state with a tsunami pulse in it to initialize field _h
xmid = geometry.xmax / 2.0
ymid = geometry.ymax / 2.0
sigma = np.floor(geometry.xmax / 20.0)
for i in range(_xps, _xpe + 1):
    for j in range(_yps, _ype + 1):
        dsqr = ((i-1) * geometry.dx - xmid)**2 + ((j-1) * geometry.dy - ymid)**2
        _h[i - _xps, j - _yps] = 5000.0 + np.exp(-dsqr / sigma**2) * (model_config.h0 - 5000.0)

state = ShallowWaterState(geometry, clock=0.0, backend=model_config.backend, h=_h)


# Create a model with the namelist configuration as specified by Config Parameters
shallow_water = ShallowWaterModel(model_config, geometry)

# Create a tangent linear model as specified by Config Parameters
shallow_water_tl = ShallowWaterModel(model_config, geometry)

# Spinup the forward model to avoid initial condition issues
print('Integrating forward model spinup steps: ', _spinup_steps)
shallow_water.adv_nsteps(state, _spinup_steps)

# Advance the forward model 100 steps and use it to compute a dx
_u[:,:] = state.get_u()
_v[:,:] = state.get_v()
_h[:,:] = state.get_h()

shallow_water.adv_nsteps(state, 100)

_udelta = state.get_u() - _u
_vdelta = state.get_v() - _v
_hdelta = state.get_h() - _h

_udelta = _udelta * 100000.0
_vdelta = _vdelta * 100000.0
_hdelta = _hdelta * 100000.0

uratio = np.zeros((_digits), dtype=np.float64)
vratio = np.zeros((_digits), dtype=np.float64)
hratio = np.zeros((_digits), dtype=np.float64)


# Loop over digits of precision to calculate metric ratio
for d in range(_digits):

    # Create a shallow_water state for M(x)
    state = ShallowWaterState(geometry, clock=0.0, backend=model_config.backend, u=_u, v=_v, h=_h)
    
    # Create a shallow_water state for M(x + lambda * dx)
    state_delta = ShallowWaterState(geometry, clock=0.0, backend=model_config.backend, u=_u + _lambda * _udelta, v=_v + _lambda * _vdelta, h=_h + _lambda * _hdelta)
    
    # Create a shallow_water_tl state and trajectory for M'(lambda * dx)
    trajectory = ShallowWaterState(geometry, clock=0.0, backend=model_config.backend, u=_u, v=_v, h=_h)
    state_tl = ShallowWaterState(geometry, clock=0.0, u=_lambda * _udelta, v=_lambda * _vdelta, h=_lambda * _hdelta)
   
    # Advance shallow_water, shallow_water_delta, and shallow_water_tl
    shallow_water.adv_nsteps(state, 1)
    shallow_water.adv_nsteps(state_delta, 1)
    shallow_water_tl.adv_nsteps_tl(state_tl, trajectory, 1)
    
    # Calculate and print test metric
    _mu = state.get_u()
    _mv = state.get_v()
    _mh = state.get_h()

    _m_udelta = state_delta.get_u()
    _m_vdelta = state_delta.get_v()
    _m_hdelta = state_delta.get_h()
    
    _mprime_udelta = state_tl.get_u()
    _mprime_vdelta = state_tl.get_v()
    _mprime_hdelta = state_tl.get_h()
    
    uratio[d] = (_m_udelta[(_xpe-_xps)//3,(_ype-_yps)//3] - _mu[(_xpe -_xps)//3,(_ype-_yps)//3]) / _mprime_udelta[(_xpe-_xps)//3, (_ype-_yps)//3]
    vratio[d] = (_m_vdelta[(_xpe-_xps)//3,(_ype-_yps)//3] - _mv[(_xpe -_xps)//3,(_ype-_yps)//3]) / _mprime_vdelta[(_xpe-_xps)//3, (_ype-_yps)//3]
    hratio[d] = (_m_hdelta[(_xpe-_xps)//3,(_ype-_yps)//3] - _mh[(_xpe -_xps)//3,(_ype-_yps)//3]) / _mprime_hdelta[(_xpe-_xps)//3, (_ype-_yps)//3]

    # Increase precision
    _lambda = _lambda / 10.0

# Loop over each MPI rank to check for proper increase in precision of ratio for decrease in lambda
for n in range(_nranks):
    
    if(_myrank == n):

        # Write dx info
        print("\n")
        print(f"{'min udelta':>15}{'max udelta':>15}{np.amin(_udelta):18.10f}{np.amax(_udelta):18.10f}")
        print(f"{'min vdelta':>15}{'max vdelta':>15}{np.amin(_vdelta):18.10f}{np.amax(_vdelta):18.10f}")
        print(f"{'min hdelta':>15}{'max hdelta':>15}{np.amin(_hdelta):18.10f}{np.amax(_hdelta):18.10f}")
        print("\n")

        # Write column headers
        print(f"{'Lambda':>13}{'':11}( M(x + lambda * dx) - M(x) ) / M'(lambda * dx)")
        print(f"{'':13}{'U':>18}{'V':>18}{'H':>18}")

        _lambda = 1.0
        _errors = 0
        
        for d in range(_digits):

            print(f"{_lambda:18.12f}{uratio[d]:18.12f}{vratio[d]:18.12f}{hratio[d]:18.12f}")
            
            if (d >= 1):
                # Check precision of ratios
                if(abs(uratio[d] - 1.0) > abs(uratio[d-1] - 1.0)):
                    print(f"ERROR: Precision of u ratio not decreasing as lambda decreases.  ")
                    _errors = _errors + 1
                if(abs(vratio[d] - 1.0) > abs(vratio[d-1] - 1.0)):
                    print(f"ERROR: Precision of v ratio not decreasing as lambda decreases.  ")
                    _errors = _errors + 1
                if(abs(hratio[d] - 1.0) > abs(hratio[d-1] - 1.0)):
                    print(f"ERROR: Precision of h ratio not decreasing as lambda decreases.  ")
                    _errors = _errors + 1

            # Increase precision
            _lambda = _lambda / 10.0
    
    comm.Barrier()

if (_errors > 0):
  comm.Abort(-1)

comm = MPI.Finalize()
