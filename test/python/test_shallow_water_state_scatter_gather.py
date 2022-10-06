from mpi4py import MPI
import numpy as np

import sys
sys.path.append("../../src/python")

from shallow_water_geometry_config import ShallowWaterGeometryConfig
from shallow_water_geometry import ShallowWaterGeometry
from shallow_water_state import ShallowWaterState

# To test the MPI code contained in this file, execute on multiple process:
#  ex:    mpiexec -n 4 python test_shallow_water_state_scatter_gather.py

comm = MPI.COMM_WORLD 

gc = ShallowWaterGeometryConfig(yamlpath="../test_input/test_shallow_water_config.yml")

g = ShallowWaterGeometry(geometry=gc, mpi_comm=comm)

print(g.__dict__)

test_shallow_water_state = ShallowWaterState(geometry=g, clock=0.0) # u=np.full((g.npx,g.npx), comm.Get_rank())

test_shallow_water_state.exchange_halo()
for n in range(comm.Get_size()):
    if (comm.Get_rank() == n):
        print(comm.Get_rank())
        print(g.xms, g.xme, g.yms, g.yme)
        print(g.xps, g.xpe, g.yps, g.ype)
        print(g.xts, g.xte, g.yts, g.yte)
        print(np.rot90(test_shallow_water_state.u))
    comm.Barrier()
comm.Barrier()

u_full, v_full, h_full = test_shallow_water_state.gather()

comm.Barrier()
if (comm.Get_rank() == 0):
    print(np.rot90(u_full))
test_shallow_water_state.scatter(u_full, v_full, h_full)
comm.Barrier()
for n in range(comm.Get_size()):
    if (comm.Get_rank() == n):
        print(comm.Get_rank())
        print(g.xms, g.xme, g.yms, g.yme)
        print(g.xps, g.xpe, g.yps, g.ype)
        print(g.xts, g.xte, g.yts, g.yte)
        print(np.rot90(test_shallow_water_state.u))
    comm.Barrier()

