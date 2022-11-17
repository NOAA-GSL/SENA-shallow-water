import sys
import pytest
from mpi4py import MPI

sys.path.append("../../src/python")

from shallow_water_geometry_config import ShallowWaterGeometryConfig 
from shallow_water_geometry import ShallowWaterGeometry

@pytest.fixture
def answer_key():
    comm = MPI.COMM_WORLD  
    nranks = comm.Get_size()
    myrank = comm.Get_rank()

    nx = 11
    ny = 11
    xmax = 1000.0
    ymax = 1000.0

    answers = {
        'communicator' : comm,
        'nranks' : nranks,
        'myrank' : myrank,
        'nx' : nx,
        'ny' : ny,
        'xmax' : xmax,
        'ymax' : ymax
    }
    answers['dx'] = xmax / float(nx - 1)
    answers['dy'] = ymax / float(ny - 1)

    if nranks == 1:
        answers['north'] = -1
        answers['south'] = -1
        answers['west'] = -1
        answers['east'] = -1
        answers['npx'] = nx
        answers['npy'] = ny
        answers['xps'] = 1
        answers['xpe'] = nx
        answers['yps'] = 1
        answers['ype'] = ny
        answers['xts'] = 2
        answers['xte'] = nx -1
        answers['yts'] = 2
        answers['yte'] = ny - 1
        answers['xms'] = 1
        answers['xme'] = nx
        answers['yms'] = 1
        answers['yme'] = ny

    elif nranks == 2:
        if myrank == 0:
            answers['north'] = 1
            answers['south'] = -1
            answers['west'] = -1
            answers['east'] = -1
            answers['npx'] = nx
            answers['npy'] = 5
            answers['xps'] = 1
            answers['xpe'] = nx
            answers['yps'] = 1
            answers['ype'] = 5
            answers['xts'] = 2
            answers['xte'] = nx -1
            answers['yts'] = 2
            answers['yte'] = 5
            answers['xms'] = 1
            answers['xme'] = nx
            answers['yms'] = 1
            answers['yme'] = 6
        elif myrank == 1:
            answers['north'] = -1
            answers['south'] = 0
            answers['west'] = -1
            answers['east'] = -1
            answers['npx'] = nx
            answers['npy'] = 6
            answers['xps'] = 1
            answers['xpe'] = nx
            answers['yps'] = 6
            answers['ype'] = ny
            answers['xts'] = 2
            answers['xte'] = nx -1
            answers['yts'] = 6
            answers['yte'] = ny - 1
            answers['xms'] = 1
            answers['xme'] = nx
            answers['yms'] = 5
            answers['yme'] = ny

    elif nranks == 4:
        if myrank == 0:
            answers['north'] = 2
            answers['south'] = -1
            answers['west'] = -1
            answers['east'] = 1
            answers['npx'] = 5
            answers['npy'] = 5
            answers['xps'] = 1
            answers['xpe'] = 5
            answers['yps'] = 1
            answers['ype'] = 5
            answers['xts'] = 2
            answers['xte'] = 5
            answers['yts'] = 2
            answers['yte'] = 5
            answers['xms'] = 1
            answers['xme'] = 6
            answers['yms'] = 1
            answers['yme'] = 6
        elif myrank == 1:
            answers['north'] = 3
            answers['south'] = -1
            answers['west'] = 0
            answers['east'] = -1
            answers['npx'] = 6
            answers['npy'] = 5
            answers['xps'] = 6
            answers['xpe'] = nx
            answers['yps'] = 1
            answers['ype'] = 5
            answers['xts'] = 6
            answers['xte'] = nx -1
            answers['yts'] = 2
            answers['yte'] = 5
            answers['xms'] = 5
            answers['xme'] = nx
            answers['yms'] = 1
            answers['yme'] = 6
        elif myrank == 2:
            answers['north'] = -1
            answers['south'] = 0
            answers['west'] = -1
            answers['east'] = 3
            answers['npx'] = 5
            answers['npy'] = 6
            answers['xps'] = 1
            answers['xpe'] = 5
            answers['yps'] = 6
            answers['ype'] = ny
            answers['xts'] = 2
            answers['xte'] = 5
            answers['yts'] = 6
            answers['yte'] = ny-1
            answers['xms'] = 1
            answers['xme'] = 6
            answers['yms'] = 5
            answers['yme'] = ny
        elif myrank == 3:
            answers['north'] = -1
            answers['south'] = 1
            answers['west'] = 2
            answers['east'] = -1
            answers['npx'] = 6
            answers['npy'] = 6
            answers['xps'] = 6
            answers['xpe'] = nx
            answers['yps'] = 6
            answers['ype'] = ny
            answers['xts'] = 6
            answers['xte'] = nx -1
            answers['yts'] = 6
            answers['yte'] = ny - 1
            answers['xms'] = 5
            answers['xme'] = nx
            answers['yms'] = 5
            answers['yme'] = ny

    elif nranks == 9:
        if myrank == 0:
            answers['north'] = 3
            answers['south'] = -1
            answers['west'] = -1
            answers['east'] = 1
            answers['npx'] = 3
            answers['npy'] = 3
            answers['xps'] = 1
            answers['xpe'] = 3
            answers['yps'] = 1
            answers['ype'] = 3
            answers['xts'] = 2
            answers['xte'] = 3
            answers['yts'] = 2
            answers['yte'] = 3
            answers['xms'] = 1
            answers['xme'] = 4
            answers['yms'] = 1
            answers['yme'] = 4
        elif myrank ==1:
            answers['north'] = 4
            answers['south'] = -1
            answers['west'] = 0
            answers['east'] = 2
            answers['npx'] = 4
            answers['npy'] = 3
            answers['xps'] = 4
            answers['xpe'] = 7
            answers['yps'] = 1
            answers['ype'] = 3
            answers['xts'] = 4
            answers['xte'] = 7
            answers['yts'] = 2
            answers['yte'] = 3
            answers['xms'] = 3
            answers['xme'] = 8
            answers['yms'] = 1
            answers['yme'] = 4
        elif myrank == 2:
            answers['north'] = 5
            answers['south'] = -1
            answers['west'] = 1
            answers['east'] = -1
            answers['npx'] = 4
            answers['npy'] = 3
            answers['xps'] = 8
            answers['xpe'] = nx
            answers['yps'] = 1
            answers['ype'] = 3
            answers['xts'] = 8
            answers['xte'] = nx-1
            answers['yts'] = 2
            answers['yte'] = 3
            answers['xms'] = 7
            answers['xme'] = nx
            answers['yms'] = 1
            answers['yme'] = 4
        elif myrank == 3:
            answers['north'] = 6
            answers['south'] = 0
            answers['west'] = -1
            answers['east'] = 4
            answers['npx'] = 3
            answers['npy'] = 4
            answers['xps'] = 1
            answers['xpe'] = 3
            answers['yps'] = 4
            answers['ype'] = 7
            answers['xts'] = 2
            answers['xte'] = 3
            answers['yts'] = 4
            answers['yte'] = 7
            answers['xms'] = 1
            answers['xme'] = 4
            answers['yms'] = 3
            answers['yme'] = 8
        elif myrank == 4:
            answers['north'] = 7
            answers['south'] =1
            answers['west'] = 3
            answers['east'] = 5
            answers['npx'] = 4
            answers['npy'] = 4
            answers['xps'] = 4
            answers['xpe'] = 7
            answers['yps'] = 4
            answers['ype'] = 7
            answers['xts'] = 4
            answers['xte'] = 7
            answers['yts'] = 4
            answers['yte'] = 7
            answers['xms'] = 3
            answers['xme'] = 8
            answers['yms'] = 3
            answers['yme'] = 8
        elif myrank == 5:
            answers['north'] = 8
            answers['south'] = 2
            answers['west'] = 4
            answers['east'] = -1
            answers['npx'] = 4
            answers['npy'] = 4
            answers['xps'] = 8
            answers['xpe'] = nx
            answers['yps'] = 4
            answers['ype'] = 7
            answers['xts'] = 8
            answers['xte'] = nx -1
            answers['yts'] = 4
            answers['yte'] = 7
            answers['xms'] = 7
            answers['xme'] = nx
            answers['yms'] = 3
            answers['yme'] = 8
        elif myrank == 6:
            answers['north'] = -1
            answers['south'] = 3
            answers['west'] = -1
            answers['east'] = 7
            answers['npx'] = 3
            answers['npy'] = 4
            answers['xps'] = 1
            answers['xpe'] = 3
            answers['yps'] = 8
            answers['ype'] = ny
            answers['xts'] = 2
            answers['xte'] = 3
            answers['yts'] = 8
            answers['yte'] = ny-1
            answers['xms'] = 1
            answers['xme'] = 4
            answers['yms'] = 7
            answers['yme'] = ny
        elif myrank == 7:
            answers['north'] = -1
            answers['south'] = 4
            answers['west'] = 6
            answers['east'] = 8
            answers['npx'] = 4
            answers['npy'] = 4
            answers['xps'] = 4
            answers['xpe'] = 7
            answers['yps'] = 8
            answers['ype'] = ny
            answers['xts'] = 4
            answers['xte'] = 7
            answers['yts'] = 8
            answers['yte'] = ny-1
            answers['xms'] = 3
            answers['xme'] = 8
            answers['yms'] = 7
            answers['yme'] = ny
        elif myrank == 8:
            answers['north'] = -1
            answers['south'] = 5
            answers['west'] = 7
            answers['east'] = -1
            answers['npx'] = 4
            answers['npy'] = 4
            answers['xps'] = 8
            answers['xpe'] = nx
            answers['yps'] = 8
            answers['ype'] = ny
            answers['xts'] = 8
            answers['xte'] = nx -1
            answers['yts'] = 8
            answers['yte'] = ny -1
            answers['xms'] = 7
            answers['xme'] = nx
            answers['yms'] = 7
            answers['yme'] = ny

    return answers


@pytest.mark.mpi(min_size=1)
def test_shallow_water_geometry(answer_key):
    comm = MPI.COMM_WORLD
    gc = ShallowWaterGeometryConfig(answer_key['nx'], answer_key['ny'], answer_key['xmax'], answer_key['ymax'])
    g = ShallowWaterGeometry(gc, comm)
    
    assert g.communicator == answer_key['communicator']
    assert g.nranks == answer_key['nranks']
    assert g.rank == answer_key['myrank']
    assert g.nx == answer_key['nx']
    assert g.ny == answer_key['ny']
    assert g.xmax == answer_key['xmax']
    assert g.ymax == answer_key['ymax']
    assert g.dx == answer_key['dx']
    assert g.dy == answer_key['dy']
    assert g.north == answer_key['north']
    assert g.south == answer_key['south']
    assert g.west == answer_key['west']
    assert g.east == answer_key['east']
    assert g.npx == answer_key['npx']
    assert g.npy == answer_key['npy']
    assert g.xps == answer_key['xps']
    assert g.xpe == answer_key['xpe']
    assert g.yps == answer_key['yps']
    assert g.ype == answer_key['ype']
    assert g.xts == answer_key['xts']
    assert g.xte == answer_key['xte']
    assert g.yts == answer_key['yts']
    assert g.yte == answer_key['yte']
    assert g.xms == answer_key['xms']
    assert g.xme == answer_key['xme']
    assert g.yms == answer_key['yms']
    assert g.yme == answer_key['yme']

