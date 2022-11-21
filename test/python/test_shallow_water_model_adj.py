import sys
import numpy as np
import pytest
from mpi4py import MPI

sys.path.append("../../src/python")

from shallow_water_geometry_config import ShallowWaterGeometryConfig
from shallow_water_geometry import ShallowWaterGeometry
from shallow_water_state import ShallowWaterState
from shallow_water_model_config import ShallowWaterModelConfig
from shallow_water_gt4py_config import ShallowWaterGT4PyConfig
from shallow_water_model import ShallowWaterModel
from shallow_water_model_tl import ShallowWaterModelTL
from shallow_water_model_adj import ShallowWaterModelADJ

@pytest.mark.mpi(min_size=1)
def test_shallow_water_model_adj_init():
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

    model = ShallowWaterModelADJ(mc, gtc, g)

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
    ny = 101
    xmax = 100000.0
    ymax = 100000.0
    u0 = 0.0
    v0 = 0.0
    b0 = 0.0
    h0 = 5030.0
    g = 9.81
    dt = 0.68 * (xmax / (nx - 1.0)) / (u0 + (g * (h0 - b0))**0.5)

    spinup_steps = 1000

    myrank = comm.Get_rank()
    
    gc = ShallowWaterGeometryConfig(nx, ny, xmax, ymax)
    g = ShallowWaterGeometry(gc, comm)
    gtc = ShallowWaterGT4PyConfig('numpy', np.float64)

    npx = g.npx
    npy = g.npy

    u = np.zeros((npx, npy))
    v = np.zeros((npx, npy))
    h = np.zeros((npx, npy))
    xu = np.zeros((npx, npy))
    xv = np.zeros((npx, npy))
    xh = np.zeros((npx, npy))
    yu = np.zeros((npx, npy))
    yv = np.zeros((npx, npy))
    yh = np.zeros((npx, npy))

    dx = g.dx
    dy = g.dy
    xmid = g.xmax / 2.0
    ymid = g.ymax / 2.0
    sigma = np.floor(g.xmax / 20.0)
    for i in range(g.xps, g.xpe + 1):
        for j in range(g.yps, g.ype + 1):
            dsqr = ((i-1) * dx - xmid)**2 + ((j-1) * dy - ymid)**2
            h[i - g.xps,j - g.yps] = 5000.0 + np.exp(-dsqr / sigma**2) * (h0 - 5000.0)
    state = ShallowWaterState(g, gtc, h=h)

    mc = ShallowWaterModelConfig(dt, u0, v0, b0, h0)
 
    shallow_water = ShallowWaterModel(mc, gtc, g)

    print(f"Integrating forward model spinup steps: {spinup_steps}")
    shallow_water.adv_nsteps(state, spinup_steps)

    u[:,:] = state.get_u()
    v[:,:] = state.get_v()
    h[:,:] = state.get_h()
    shallow_water.adv_nsteps(state, 100)
    xu = (state.get_u() - u)
    xv = (state.get_v() - v)
    xh = (state.get_h() - h)
    xu = xu * 1000.0
    xv = xv * 1000.0
    xh = xh * 1000.0
    statex = ShallowWaterState(g, gtc, u=xu, v=xv, h=xh)
    stated = ShallowWaterState(g, gtc, u=xu, v=xv, h=xh)

    shallow_water.adv_nsteps(state, 100)
    yu = (state.get_u() - u)
    yv = (state.get_v() - v)
    yh = (state.get_h() - h)
    yu = yu * 1000.0
    yv = yv * 1000.0
    yh = yh * 1000.0
    statey = ShallowWaterState(g, gtc, u=yu, v=yv, h=yh)
    stateb = ShallowWaterState(g, gtc, u=yu, v=yv, h=yh)

    shallow_water_tl = ShallowWaterModelTL(mc, gtc, g)
    shallow_water_tl.adv_nsteps(state=stated, trajectory=state, nsteps=1)

    shallow_water_adj = ShallowWaterModelADJ(mc, gtc, g)
    shallow_water_adj.adv_nsteps(state=stateb, trajectory=state, nsteps=1)

    global_ud = np.zeros((g.nx, g.ny))
    global_vd = np.zeros((g.nx, g.ny))
    global_hd = np.zeros((g.nx, g.ny))
    global_ud, global_vd, global_hd = stated.gather()

    global_ub = np.zeros((g.nx, g.ny))
    global_vb = np.zeros((g.nx, g.ny))
    global_hb = np.zeros((g.nx, g.ny))
    global_ub, global_vb, global_hb = stateb.gather()

    global_xu = np.zeros((g.nx, g.ny))
    global_xv = np.zeros((g.nx, g.ny))
    global_xh = np.zeros((g.nx, g.ny))
    global_xu, global_xv, global_xh = statex.gather()

    global_yu = np.zeros((g.nx, g.ny))
    global_yv = np.zeros((g.nx, g.ny))
    global_yh = np.zeros((g.nx, g.ny))
    global_yu, global_yv, global_yh = statey.gather()

    if (myrank == 0):
        dot_ratio = ((np.dot(global_ud.flatten(), global_yu.flatten()) +  \
                      np.dot(global_vd.flatten(), global_yv.flatten()) +  \
                      np.dot(global_hd.flatten(), global_yh.flatten())) / \
                     (np.dot(global_xu.flatten(), global_ub.flatten()) +  \
                      np.dot(global_xv.flatten(), global_vb.flatten()) +  \
                      np.dot(global_xh.flatten(), global_hb.flatten())))
        print(f"{'<M(x),y> / <x, M*(y)> = ':>25}{dot_ratio:20.15f}")

        assert abs(dot_ratio - 1.0) <= 1.0E-14

