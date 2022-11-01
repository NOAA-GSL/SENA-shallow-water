import sys
from numpy.testing import assert_almost_equal, assert_array_equal
import numpy as np
import pytest

sys.path.append("../../src/python")

from shallow_water_geometry_config import ShallowWaterGeometryConfig 
from shallow_water_geometry import ShallowWaterGeometry
from shallow_water_state import ShallowWaterState
from mpi4py import MPI

@pytest.fixture
def input_geometry():
    comm = MPI.COMM_WORLD  
    nranks = comm.Get_size()
    myrank = comm.Get_rank()
    nx = 101
    ny = 201
    xmax = 100000.0
    ymax = 100000.0
    config = ShallowWaterGeometryConfig(nx, ny, xmax, ymax)
    geometry = ShallowWaterGeometry(config, comm)
    return geometry

@pytest.fixture
def input_io_geometry():
    comm = MPI.COMM_WORLD  
    nranks = comm.Get_size()
    myrank = comm.Get_rank()
    nx = 10
    ny = 10
    xmax = 10000.0
    ymax = 10000.0
    config = ShallowWaterGeometryConfig(nx, ny, xmax, ymax)
    geometry = ShallowWaterGeometry(config, comm)
    return geometry


@pytest.mark.mpi(min_size=1)
def test_shallow_water_state_init_default(input_geometry):
    state = ShallowWaterState(input_geometry)
    assert_array_equal(state.u, 0.0)
    assert_array_equal(state.v, 0.0)
    assert_array_equal(state.h, 0.0)

@pytest.mark.mpi(min_size=1)
def test_shallow_water_state_init_optional(input_geometry):
    npx = input_geometry.npx
    npy = input_geometry.npy
    xps = input_geometry.xps
    xpe = input_geometry.xpe
    yps = input_geometry.yps
    ype = input_geometry.ype
    xms = input_geometry.xms
    xme = input_geometry.xme
    yms = input_geometry.yms
    yme = input_geometry.yme
    u = np.full((npx, npy), 1.0)
    v = np.full((npx, npy), 2.0)
    h = np.full((npx, npy), 3.0)
    state = ShallowWaterState(input_geometry, u=u, v=v, h=h)

    assert_array_equal(state.u.data[xps-xms:xpe-xms+1, yps-yms:ype-yms+1], u)
    assert_array_equal(state.v.data[xps-xms:xpe-xms+1, yps-yms:ype-yms+1], v)
    assert_array_equal(state.h.data[xps-xms:xpe-xms+1, yps-yms:ype-yms+1], h)

@pytest.mark.mpi(min_size=1)
def test_shallow_water_state_halo(input_geometry):
    myrank = input_geometry.communicator.Get_rank()
    npx = input_geometry.npx
    npy = input_geometry.npy
    xps = input_geometry.xps
    xpe = input_geometry.xpe
    yps = input_geometry.yps
    ype = input_geometry.ype
    xms = input_geometry.xms
    xme = input_geometry.xme
    yms = input_geometry.yms
    yme = input_geometry.yme
    north = input_geometry.north
    south = input_geometry.south
    west = input_geometry.west
    east = input_geometry.east
    u = np.full((npx, npy), 10.0 * myrank)
    v = np.full((npx, npy), 20.0 * myrank)
    h = np.full((npx, npy), 30.0 * myrank)

    state = ShallowWaterState(input_geometry, u=u, v=v, h=h)

    state.exchange_halo()

    assert_array_equal(state.u.data[xps-xms:xpe-xms+1, yps-yms:ype-yms+1], u)
    assert_array_equal(state.v.data[xps-xms:xpe-xms+1, yps-yms:ype-yms+1], v)
    assert_array_equal(state.h.data[xps-xms:xpe-xms+1, yps-yms:ype-yms+1], h)
    
    if (north != -1):
        assert_array_equal(state.u.data[xps-xms:xpe-xms+1, yme-yms], 10.0 * north)
        assert_array_equal(state.v.data[xps-xms:xpe-xms+1, yme-yms], 20.0 * north)
        assert_array_equal(state.h.data[xps-xms:xpe-xms+1, yme-yms], 30.0 * north)
    if (south != -1):
        assert_array_equal(state.u.data[xps-xms:xpe-xms+1, yms-yms], 10.0 * south)
        assert_array_equal(state.v.data[xps-xms:xpe-xms+1, yms-yms], 20.0 * south)
        assert_array_equal(state.h.data[xps-xms:xpe-xms+1, yms-yms], 30.0 * south)
    if (west != -1):
        assert_array_equal(state.u.data[xms-xms, yps-yms:ype-yms+1],  10.0 * west)
        assert_array_equal(state.v.data[xms-xms, yps-yms:ype-yms+1],  20.0 * west)
        assert_array_equal(state.h.data[xms-xms, yps-yms:ype-yms+1],  30.0 * west)
    if (east != -1):
        assert_array_equal(state.u.data[xme-xms, yps-yms:ype-yms+1], 10.0 * east)
        assert_array_equal(state.v.data[xme-xms, yps-yms:ype-yms+1], 20.0 * east)
        assert_array_equal(state.h.data[xme-xms, yps-yms:ype-yms+1], 30.0 * east)

@pytest.mark.mpi(min_size=1)
def test_shallow_water_state_read(input_io_geometry):
    npx = input_io_geometry.npx
    npy = input_io_geometry.npy
    xps = input_io_geometry.xps
    xpe = input_io_geometry.xpe
    yps = input_io_geometry.yps
    ype = input_io_geometry.ype
    xms = input_io_geometry.xms
    xme = input_io_geometry.xme
    yms = input_io_geometry.yms
    yme = input_io_geometry.yme
    clock = 182850.615187004
    u = np.full((npx, npy), 5.0)
    v = np.full((npx, npy), 6.0)
    h = np.full((npx, npy), 7.0)

    state = ShallowWaterState(input_io_geometry)
    state.read("../test_input/test_shallow_water_reader.nc")

    assert_almost_equal(state.clock, clock, decimal=9)
    assert_array_equal(state.u.data[xps-xms:xpe-xms+1, yps-yms:ype-yms+1], u)
    assert_array_equal(state.v.data[xps-xms:xpe-xms+1, yps-yms:ype-yms+1], v)
    assert_array_equal(state.h.data[xps-xms:xpe-xms+1, yps-yms:ype-yms+1], h)

@pytest.mark.mpi(min_size=1)
def test_shallow_water_state_write(input_io_geometry):
    npx = input_io_geometry.npx
    npy = input_io_geometry.npy
    xps = input_io_geometry.xps
    xpe = input_io_geometry.xpe
    yps = input_io_geometry.yps
    ype = input_io_geometry.ype
    xms = input_io_geometry.xms
    xme = input_io_geometry.xme
    yms = input_io_geometry.yms
    yme = input_io_geometry.yme
    clock = 1000.0
    u = np.full((npx, npy), 5.0)
    v = np.full((npx, npy), 6.0)
    h = np.full((npx, npy), 7.0)

    state = ShallowWaterState(input_io_geometry, u=u, v=v, h=h, clock=clock)
    state.write("test_shallow_water_writer.nc")

    state_read = ShallowWaterState(input_io_geometry)
    state_read.read("test_shallow_water_writer.nc")

    assert_almost_equal(state.clock, state_read.clock, decimal=9)

    assert_array_equal(state_read.u.data[xps-xms:xpe-xms+1, yps-yms:ype-yms+1], u)
    assert_array_equal(state_read.v.data[xps-xms:xpe-xms+1, yps-yms:ype-yms+1], v)
    assert_array_equal(state_read.h.data[xps-xms:xpe-xms+1, yps-yms:ype-yms+1], h)
