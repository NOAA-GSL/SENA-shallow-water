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
from shallow_water_model_tl import ShallowWaterModelTL
from mpi4py import MPI

@pytest.mark.mpi(min_size=1)
def test_shallow_water_model_tl_init():
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

    model = ShallowWaterModelTL(mc, gtc, g)

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
    dt = 0.8

    spinup_steps = 1000
    digits = 8

    myrank = comm.Get_rank()
    nranks = comm.Get_size()

    lambda_p = 1.0

    gc = ShallowWaterGeometryConfig(nx, ny, xmax, ymax)
    g = ShallowWaterGeometry(gc, comm)
    gtc = ShallowWaterGT4PyConfig('numpy', np.float64)

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

    u = np.zeros((npx, npy))
    v = np.zeros((npx, npy))
    h = np.zeros((npx, npy))
    udelta = np.zeros((npx, npy))
    vdelta = np.zeros((npx, npy))
    hdelta = np.zeros((npx, npy))
    mu = np.zeros((npx, npy))
    mv = np.zeros((npx, npy))
    mh = np.zeros((npx, npy))
    m_udelta = np.zeros((npx, npy))
    m_vdelta = np.zeros((npx, npy))
    m_hdelta = np.zeros((npx, npy))
    mprime_udelta = np.zeros((npx, npy))
    mprime_vdelta = np.zeros((npx, npy))
    mprime_hdelta = np.zeros((npx, npy))

    dx = g.dx
    dy = g.dy
    xmid = g.xmax / 2.0
    ymid = g.ymax / 2.0
    sigma = np.floor(g.xmax / 20.0)
    for i in range(g.xps, g.xpe + 1):
        for j in range(g.yps, g.ype + 1):
            dsqr = ((i-1) * g.dx - xmid)**2 + ((j-1) * g.dy - ymid)**2
            h[i - g.xps,j - g.yps] = 5000.0 + np.exp(-dsqr / sigma**2) * (h0 - 5000.0)
    state = ShallowWaterState(g, gtc, h=h)

    mc = ShallowWaterModelConfig(dt, u0, v0, b0, h0)
 
    shallow_water = ShallowWaterModel(mc, gtc, g)
    shallow_water_tl = ShallowWaterModelTL(mc, gtc, g)

    print(f"Integrating forward model spinup steps: {spinup_steps}")
    shallow_water.adv_nsteps(state, spinup_steps)

    u[:,:] = state.u[xps-xms:xpe-xms+1, yps-yms:ype-yms+1]
    v[:,:] = state.v[xps-xms:xpe-xms+1, yps-yms:ype-yms+1]
    h[:,:] = state.h[xps-xms:xpe-xms+1, yps-yms:ype-yms+1]
    shallow_water.adv_nsteps(state, 100)
    udelta = (state.u[xps-xms:xpe-xms+1, yps-yms:ype-yms+1] - u)
    vdelta = (state.v[xps-xms:xpe-xms+1, yps-yms:ype-yms+1] - v)
    hdelta = (state.h[xps-xms:xpe-xms+1, yps-yms:ype-yms+1] - h)
    udelta = udelta * 100000.0
    vdelta = vdelta * 100000.0
    hdelta = hdelta * 100000.0

    uratio = np.zeros((digits), dtype=np.float64)
    vratio = np.zeros((digits), dtype=np.float64)
    hratio = np.zeros((digits), dtype=np.float64)

    for d in range(digits):

        state = ShallowWaterState(g, gtc, u=u, v=v, h=h)

        state_delta = ShallowWaterState(g, gtc,  u=u + lambda_p * udelta, v=v + lambda_p * vdelta, h=h + lambda_p * hdelta)

        trajectory = ShallowWaterState(g, gtc, u=u, v=v, h=h)
        state_tl = ShallowWaterState(g, gtc, u=lambda_p * udelta, v=lambda_p * vdelta, h=lambda_p * hdelta)

        shallow_water.adv_nsteps(state, 1)
        shallow_water.adv_nsteps(state_delta, 1)
        shallow_water_tl.adv_nsteps(state_tl, trajectory, 1)

        mu = state.u[xps-xms:xpe-xms+1, yps-yms:ype-yms+1]
        mv = state.v[xps-xms:xpe-xms+1, yps-yms:ype-yms+1]
        mh = state.h[xps-xms:xpe-xms+1, yps-yms:ype-yms+1]

        m_udelta = state_delta.u[xps-xms:xpe-xms+1, yps-yms:ype-yms+1]
        m_vdelta = state_delta.v[xps-xms:xpe-xms+1, yps-yms:ype-yms+1]
        m_hdelta = state_delta.h[xps-xms:xpe-xms+1, yps-yms:ype-yms+1]

        mprime_udelta = state_tl.u[xps-xms:xpe-xms+1, yps-yms:ype-yms+1]
        mprime_vdelta = state_tl.v[xps-xms:xpe-xms+1, yps-yms:ype-yms+1]
        mprime_hdelta = state_tl.h[xps-xms:xpe-xms+1, yps-yms:ype-yms+1]

        uratio[d] = (m_udelta[(xpe-xps)//3,(ype-yps)//3] - mu[(xpe-xps)//3,(ype-yps)//3]) / mprime_udelta[(xpe-xps)//3,(ype-yps)//3]
        vratio[d] = (m_vdelta[(xpe-xps)//3,(ype-yps)//3] - mv[(xpe-xps)//3,(ype-yps)//3]) / mprime_vdelta[(xpe-xps)//3,(ype-yps)//3]
        hratio[d] = (m_hdelta[(xpe-xps)//3,(ype-yps)//3] - mh[(xpe-xps)//3,(ype-yps)//3]) / mprime_hdelta[(xpe-xps)//3,(ype-yps)//3]

        lambda_p = lambda_p / 10.0

    for n in range(nranks):

        if (myrank == n):

            print("\n")
            print(f"{'min udelta':>15}{'max udelta':>15}{np.amin(udelta):18.10f}{np.amax(udelta):18.10f}")
            print(f"{'min vdelta':>15}{'max vdelta':>15}{np.amin(vdelta):18.10f}{np.amax(vdelta):18.10f}")
            print(f"{'min hdelta':>15}{'max hdelta':>15}{np.amin(hdelta):18.10f}{np.amax(hdelta):18.10f}")
            print("\n")

            print(f"{'lambda':>13}{'':11}( M(x + lambda * dx) - M(x) ) / M'(lambda * dx)")
            print(f"{'':13}{'U':>18}{'V':>18}{'H':>18}")
            lambda_p = 1.0
            errors = 0

            for d in range(digits):

                print(f"{lambda_p:18.12f}{uratio[d]:18.12f}{vratio[d]:18.12f}{hratio[d]:18.12f}")

                if (d > 1):
                    if ((abs(uratio[d] - 1.0)) > (abs(uratio[d-1] - 1.0))):
                        print("ERROR: Precision of u ratio not decreasing as lambda decreases.  ")
                        errors += 1
                    if ((abs(vratio[d] - 1.0)) > (abs(vratio[d-1] - 1.0))):
                        print("ERROR: Precision of u ratio not decreasing as lambda decreases.  ")
                        errors += 1
                    if ((abs(hratio[d] - 1.0)) > (abs(hratio[d-1] - 1.0))):
                        print("ERROR: Precision of u ratio not decreasing as lambda decreases.  ")
                        errors += 1

                lambda_p = lambda_p / 10.0

        comm.Barrier()

    assert errors == 0
