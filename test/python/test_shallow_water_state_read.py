import importlib
from mpi4py import MPI
import numpy as np

import sys
sys.path.append("../../src/python")

from shallow_water_geometry_config import ShallowWaterGeometryConfig
from shallow_water_geometry import ShallowWaterGeometry
from shallow_water_state import ShallowWaterState


comm = MPI.COMM_WORLD 

comm.Get_rank()

geom_config = ShallowWaterGeometryConfig(yamlpath="../test_input/test_shallow_water_config.yml")

geom = ShallowWaterGeometry(geometry=geom_config, mpi_comm=comm)

print(geom.__dict__)

test_shallow_water_state = ShallowWaterState(geometry=geom, clock=0.0)

filepath = "../test_input/test_shallow_water_reader.nc"

test_shallow_water_state.read_NetCDF(filepath)

comm = MPI.Finalize()
