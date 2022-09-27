from mpi4py import MPI
import sys
sys.path.append("../../src/python")

from shallow_water_geometry_config import shallow_water_geometry_config
from shallow_water_geometry import shallow_water_geometry

comm = MPI.COMM_WORLD

geom_config = shallow_water_geometry_config(yamlpath="../../parm/shallow_water.yml")

print( geom_config.__dict__)

geom = shallow_water_geometry(geometry=geom_config, mpi_comm=comm)

print(geom.__dict__)

print(geom.get_communicator())

print(getattr(geom, "east"))

print(type(geom))

