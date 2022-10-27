from mpi4py import MPI
import sys
sys.path.append("../../src/python")

from shallow_water_geometry_config import ShallowWaterGeometryConfig
from shallow_water_geometry import ShallowWaterGeometry

comm = MPI.COMM_WORLD

geom_config = ShallowWaterGeometryConfig(yamlpath="../../parm/shallow_water.yml")

print( geom_config.__dict__)

geom = ShallowWaterGeometry(geometry=geom_config, mpi_comm=comm)

print(geom.__dict__)

print(geom.get_communicator())

print(getattr(geom, "east"))

print(type(geom))

comm = MPI.Finalize()
