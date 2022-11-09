#!/usr/bin/env python

import sys
sys.path.append("../../src/python")

import numpy as np
from mpi4py import MPI
import gt4py.gtscript as gtscript
import gt4py.storage as gt_storage

from shallow_water_geometry_config import ShallowWaterGeometryConfig
from shallow_water_geometry import ShallowWaterGeometry
from shallow_water_model_config import ShallowWaterModelConfig
from shallow_water_state import ShallowWaterState
from shallow_water_model import ShallowWaterModel
from test_utilities import TestUtilities

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
_spinup_steps = 100
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

_xms = geometry.xms
_xme = geometry.xme
_yms = geometry.yms
_yme = geometry.yme

# Create a model configuration
model_config = ShallowWaterModelConfig(dt=_dt, u0=_u0, v0=_v0, b0=_b0, h0=_h0)

# Allocate space for model states
# _u = gt_storage.zeros(model_config.backend, default_origin=(1,1), shape=(_xme - _xms + 1, _yme - _yms + 1), dtype=model_config.F_TYPE)
# _v = gt_storage.zeros(model_config.backend, default_origin=(1,1), shape=(_xme - _xms + 1, _yme - _yms + 1), dtype=model_config.F_TYPE)
# _h = gt_storage.zeros(model_config.backend, default_origin=(1,1), shape=(_xme - _xms + 1, _yme - _yms + 1), dtype=model_config.F_TYPE)
_udelta        = gt_storage.zeros(model_config.backend, default_origin=(1,1), shape=(_xme - _xms + 1, _yme - _yms + 1), dtype=model_config.F_TYPE)
_vdelta        = gt_storage.zeros(model_config.backend, default_origin=(1,1), shape=(_xme - _xms + 1, _yme - _yms + 1), dtype=model_config.F_TYPE)
_hdelta        = gt_storage.zeros(model_config.backend, default_origin=(1,1), shape=(_xme - _xms + 1, _yme - _yms + 1), dtype=model_config.F_TYPE)
_mu            = gt_storage.zeros(model_config.backend, default_origin=(1,1), shape=(_xme - _xms + 1, _yme - _yms + 1), dtype=model_config.F_TYPE)
_mv            = gt_storage.zeros(model_config.backend, default_origin=(1,1), shape=(_xme - _xms + 1, _yme - _yms + 1), dtype=model_config.F_TYPE)
_mh            = gt_storage.zeros(model_config.backend, default_origin=(1,1), shape=(_xme - _xms + 1, _yme - _yms + 1), dtype=model_config.F_TYPE)
_m_udelta      = gt_storage.zeros(model_config.backend, default_origin=(1,1), shape=(_xme - _xms + 1, _yme - _yms + 1), dtype=model_config.F_TYPE)
_m_vdelta      = gt_storage.zeros(model_config.backend, default_origin=(1,1), shape=(_xme - _xms + 1, _yme - _yms + 1), dtype=model_config.F_TYPE)
_m_hdelta      = gt_storage.zeros(model_config.backend, default_origin=(1,1), shape=(_xme - _xms + 1, _yme - _yms + 1), dtype=model_config.F_TYPE)
_mprime_udelta = gt_storage.zeros(model_config.backend, default_origin=(1,1), shape=(_xme - _xms + 1, _yme - _yms + 1), dtype=model_config.F_TYPE)
_mprime_vdelta = gt_storage.zeros(model_config.backend, default_origin=(1,1), shape=(_xme - _xms + 1, _yme - _yms + 1), dtype=model_config.F_TYPE)
_mprime_hdelta = gt_storage.zeros(model_config.backend, default_origin=(1,1), shape=(_xme - _xms + 1, _yme - _yms + 1), dtype=model_config.F_TYPE)


# Create a state with a tsunami pulse in it to initialize field _h
_h = gt_storage.zeros(model_config.backend, default_origin=(1,1), shape=(_xme - _xms + 1, _yme - _yms + 1), dtype=model_config.F_TYPE)
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
_u = gt_storage.from_array(state.u, backend=model_config.backend, default_origin=(1,1), dtype=model_config.F_TYPE)
_v = gt_storage.from_array(state.v, backend=model_config.backend, default_origin=(1,1), dtype=model_config.F_TYPE)
_h = gt_storage.from_array(state.h, backend=model_config.backend, default_origin=(1,1), dtype=model_config.F_TYPE)

shallow_water.adv_nsteps(state, 100)

_udelta = state.u - _u
_vdelta = state.v - _v
_hdelta = state.h - _h

_udelta = _udelta * 100000.0
_vdelta = _vdelta * 100000.0
_hdelta = _hdelta * 100000.0

uratio = np.zeros(_digits)
vratio = np.zeros(_digits)
hratio = np.zeros(_digits)


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
    _mu = state.u
    _mv = state.v
    _mh = state.h

    _m_udelta = state_delta.u
    _m_vdelta = state_delta.v
    _m_hdelta = state_delta.h
    
    _mprime_udelta = state_tl.u
    _mprime_vdelta = state_tl.v
    _mprime_hdelta = state_tl.h
    
    uratio[d] = (_m_udelta[(_xpe - _xps) // 3, (_ype -  _yps) // 3] - _mu[(_xpe - _xps) // 3, (_ype - _yps) // 3]) / _mprime_udelta[(_xpe - _xps) // 3, (_ype - _yps) // 3]
    vratio[d] = (_m_vdelta[(_xpe - _xps) // 3, (_ype -  _yps) // 3] - _mv[(_xpe - _xps) // 3, (_ype - _yps) // 3]) / _mprime_vdelta[(_xpe - _xps) // 3, (_ype - _yps) // 3]
    hratio[d] = (_m_hdelta[(_xpe - _xps) // 3, (_ype -  _yps) // 3] - _mh[(_xpe - _xps) // 3, (_ype - _yps) // 3]) / _mprime_hdelta[(_xpe - _xps) // 3, (_ype - _yps) // 3]

    # Increase precision
    _lambda = _lambda / 10.0

# Loop over each MPI rank to check for proper increase in precision of ratio for decrease in lambda
for n in range(_nranks):
    
    if(_myrank == n):

        # Write dx info
        print("\n")
        print('{0:<10} {1:<35} {2:<32}'.format(" ", f"min udelta = {np.min(_udelta)}", f"max udelta = {np.max(_udelta)}"))
        print('{0:<10} {1:<35} {2:<32}'.format(" ", f"min vdelta = {np.min(_vdelta)}", f"max vdelta = {np.max(_vdelta)}"))
        print('{0:<10} {1:<35} {2:<32}'.format(" ", f"min hdelta = {np.min(_hdelta)}", f"max hdelta = {np.max(_hdelta)}"))
        print("\n")

        # Write column headers
        print('{0:<30} {1:>24}'.format("Lambda", "( M(x + lambda * dx) - M(x) ) / M'(lambda * dx)"))
        print('{0:<20} {1:>18} {2:>18} {3:>18}'.format(" ", "U", "V", "H"))

        _lambda = 1.0
        _errors = 0

        for d in range(_digits):
            print('{0:<26} {1:<22} {2:<22} {3:<22}'.format(f'{_lambda}', f'{uratio[d]}', f'{vratio[d]}', f'{hratio[d]}'))
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
