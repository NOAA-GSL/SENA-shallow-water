import sys
from numpy.testing import assert_almost_equal, assert_array_equal
import numpy as np
import pytest

sys.path.append("../../src/python")

from shallow_water_geometry_config import ShallowWaterGeometryConfig
from shallow_water_geometry import ShallowWaterGeometry
from shallow_water_state import ShallowWaterState
from shallow_water_model_config import ShallowWaterModelConfig
from shallow_water_gt4py_config import ShallowWaterGT4PyConfig
from shallow_water_model import ShallowWaterModel
from mpi4py import MPI

@pytest.mark.mpi(min_size=1)
def test_shallow_water_model_init():
    comm = MPI.COMM_WORLD
    nx = 101
    ny = 201
    xmax = 100000.0
    ymax = 100000.0
    dx = 1000.0
    dy = 500.0
    u0 = 0.0
    v0 = 0.0
    b0 = 0.0
    h0 = 5030.0

    g = 9.81
    dt = 0.68 * dx / (u0 + (g * (h0 - b0))**0.5)

    gc = ShallowWaterGeometryConfig(nx, ny, xmax, ymax)
    g = ShallowWaterGeometry(gc, comm)
    mc = ShallowWaterModelConfig(dt, u0, v0, b0, h0)
    gtc = ShallowWaterGT4PyConfig('numpy', np.float64)

    model = ShallowWaterModel(mc, gtc, g)

    geometry = model.geometry

    assert geometry.dx == dx
    assert geometry.dy == dy
    assert model.dt == dt
    assert model.backend == 'numpy'
    assert model.float_type == np.float64

@pytest.mark.mpi(min_size=1)
def test_shallow_water_model_adv_nsteps():
    comm = MPI.COMM_WORLD
    nx = 101
    ny = 201
    xmax = 100000.0
    ymax = 100000.0
    u0 = 0.0
    v0 = 0.0
    b0 = 0.0
    h0 = 5030.0
    g = 9.81
    dt = 0.68 * (xmax / float(nx - 1.0)) / (u0 + (g * (h0 - b0))**0.5)
    step = 1000

    gc = ShallowWaterGeometryConfig(nx, ny, xmax, ymax)
    g = ShallowWaterGeometry(gc, comm)

    npx = g.npx
    npy = g.npy
    xps = g.xps
    xpe = g.xpe
    yps = g.yps
    ype = g.ype
    xms = g.xms
    xme = g.xme
    yms = g.yms
    yme = g.yme

    mc = ShallowWaterModelConfig(dt, u0, v0, b0, h0)
    gtc = ShallowWaterGT4PyConfig('numpy', np.float64)

    model = ShallowWaterModel(mc, gtc, g)

    u = np.zeros((npx, npy))
    v = np.zeros((npx, npy))
    h = np.zeros((npx, npy))

    state = ShallowWaterState(g, gtc, u=u, v=v, h=h, clock = step * model.dt)

    model.adv_nsteps(state, 1)

    assert_almost_equal(state.clock, step * model.dt + model.dt)
    assert_array_equal(state.u.data[xps-xms:xpe-xms+1, yps-yms:ype-yms+1], 0.0)
    assert_array_equal(state.v.data[xps-xms:xpe-xms+1, yps-yms:ype-yms+1], 0.0)
    assert_array_equal(state.h.data[xps-xms:xpe-xms+1, yps-yms:ype-yms+1], 0.0)

    model.adv_nsteps(state, 2)

    assert_almost_equal(state.clock, step * model.dt + model.dt + model.dt + model.dt)

    h[:,:] = h[:,:] + 10.0
    state = ShallowWaterState(g, gtc, u=u, v=v, h=h)

    model.adv_nsteps(state, 1)

    assert_array_equal(state.u.data[xps-xms:xpe-xms+1, yps-yms:ype-yms+1], 0.0)
    assert_array_equal(state.v.data[xps-xms:xpe-xms+1, yps-yms:ype-yms+1], 0.0)
    assert_array_equal(state.h.data[xps-xms:xpe-xms+1, yps-yms:ype-yms+1], 10.0)

@pytest.mark.mpi(min_size=1)
def test_shallow_water_model_regression():
    comm = MPI.COMM_WORLD
    myrank = comm.Get_rank()
    nx = 11
    ny = 11
    xmax = 10000.0
    ymax = 10000.0
    u0 = 0.0
    v0 = 0.0
    b0 = 0.0
    h0 = 5030.0
    g = 9.81
    dt = 0.68 * (xmax / float(nx - 1.0)) / (u0 + (g * (h0 - b0))**0.5)

    u_rms_baseline = 0.00161019683016338
    v_rms_baseline = 0.00161183246804103
    h_rms_baseline = 5000.37196249264

    gc = ShallowWaterGeometryConfig(nx, ny, xmax, ymax)
    g = ShallowWaterGeometry(gc, comm)
    gtc = ShallowWaterGT4PyConfig('numpy', np.float64)

    h = np.empty((g.npx, g.npy), dtype=float)
    dx = g.dx
    dy = g.dy
    xmid = g.xmax / 2.0
    ymid = g.ymax / 2.0
    sigma = np.floor(g.xmax / 20.0)
    for i in range(g.xps, g.xpe + 1):
        for j in range(g.yps, g.ype + 1):
            dsqr = ((i-1) * g.dx - xmid)**2 + ((j-1) * g.dy - ymid)**2
            h[i - g.xps,j - g.yps] = 5000.0 + np.exp(-dsqr / sigma**2) * (h0 - 5000.0)
    s = ShallowWaterState(g, gtc, h=h)

    mc = ShallowWaterModelConfig(dt, u0, v0, b0, h0)

    model = ShallowWaterModel(mc, gtc, g)

    model.adv_nsteps(s, 100)

    if (myrank == 0):
        u_full = np.zeros((nx, ny))
        v_full = np.zeros((nx, ny))
        h_full = np.zeros((nx, ny))
    u_full, v_full, h_full = s.gather()

    if (myrank == 0):
        u_full = u_full * u_full
        u_rms = (np.sum(u_full) / (g.nx * g.ny))**0.5
        v_full = v_full * v_full
        v_rms = (np.sum(v_full) / (g.nx * g.ny))**0.5
        h_full = h_full * h_full
        h_rms = (np.sum(h_full) / (g.nx * g.ny))**0.5

        assert_almost_equal(u_rms, u_rms_baseline, 13)
        assert_almost_equal(v_rms, v_rms_baseline, 13)
        assert_almost_equal(h_rms, h_rms_baseline, 11)
