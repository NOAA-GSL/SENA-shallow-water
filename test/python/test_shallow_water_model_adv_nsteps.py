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
from test_utilities import TestUtilities

# Config Parameters
_step = 1000
_nx = 101
_ny = 101
_xmax = 100000.0
_ymax = 100000.0
_u0 = 0.0
_v0 = 0.0
_b0 = 0.0
_h0 = 10.06
_g = 9.81
_dt = 0.68 * (_xmax / (np.float64(_nx) - 1.0)) / (_u0 + np.sqrt(_g * (_h0 - _b0)))

comm = MPI.COMM_WORLD

# Initialize error count to 0
_errors = 0

geometry_config = ShallowWaterGeometryConfig(nx=_nx, ny=_ny, xmax=_xmax, ymax=_ymax)
geometry = ShallowWaterGeometry(geometry=geometry_config, mpi_comm=comm)

model_config = ShallowWaterModelConfig(dt=_dt, u0=_u0, v0=_v0, b0=_b0, h0=_h0)
model = ShallowWaterModel(model_config, geometry)

_u = gt_storage.zeros(model_config.backend, default_origin=(1,1), shape=(geometry.xme - geometry.xms + 1, geometry.yme - geometry.yms + 1), dtype=np.float64)
_v = gt_storage.zeros(model_config.backend, default_origin=(1,1), shape=(geometry.xme - geometry.xms + 1, geometry.yme - geometry.yms + 1), dtype=np.float64)
_h = gt_storage.zeros(model_config.backend, default_origin=(1,1), shape=(geometry.xme - geometry.xms + 1, geometry.yme - geometry.yms + 1), dtype=np.float64)

state = ShallowWaterState(geometry, u=_u, v=_v, h=_h, clock=_step * model.dt)

model.adv_nsteps(state,1)

#  Check clock
TestUtilities.check_real_scalar(state.clock, "clock", _step * model.dt + model.dt, 10E-12, _errors)
# TestUtilities.check_real_scalar(state.clock, "clock", _step * model.dt + model.dt, 200.0, _errors)

#  Check u
TestUtilities.check_min_max_real(state.u, "u", 0.0, 0.0, _errors)

#  Check v
TestUtilities.check_min_max_real(state.v, "v", 0.0, 0.0, _errors)

#  Check h
TestUtilities.check_min_max_real(state.h, "h", 0.0, 0.0, _errors)

#  Advance the model 2 steps
model.adv_nsteps(state, 2)

#  Check clock
TestUtilities.check_real_scalar(state.clock, "clock", _step * model.dt + model.dt + model.dt + model.dt, 10E-12, _errors)

#  Initialize shallow water state
_h[:,:] = _h[:,:] + 10.0
state = ShallowWaterState(geometry, clock=0.0, u=_u, v=_v, h=_h)

#  Initialize shallow water model
model = ShallowWaterModel(model_config, geometry)

#  Advance the model 1 step
model.adv_nsteps(state, 1)

#  Check u
TestUtilities.check_min_max_real(state.u, "u", 0.0, 0.0, _errors)

#  Check v
TestUtilities.check_min_max_real(state.v, "v", 0.0, 0.0, _errors)

#  Check h
TestUtilities.check_min_max_real(state.h, "h", 10.0, 10.0, _errors)

if (_errors > 0):
  comm.Abort(-1)


comm = MPI.Finalize()
