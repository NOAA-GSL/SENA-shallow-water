from mpi4py import MPI
from netCDF4 import Dataset
import numpy as np
import sys
import os
sys.path.append("../../src/python")

from shallow_water_geometry_config import ShallowWaterGeometryConfig
from shallow_water_geometry import ShallowWaterGeometry
from shallow_water_state import ShallowWaterState

comm = MPI.COMM_WORLD 

comm.Get_rank()

geom_config = ShallowWaterGeometryConfig(yamlpath="../test_input/test_shallow_water_config.yml")

geom = ShallowWaterGeometry(geometry=geom_config, mpi_comm=comm)

test_shallow_water_state = ShallowWaterState(geometry=geom, clock=0.0)

path = "../test_output"

filename = "../test_output/test_shallow_water_writer.nc"

if not os.path.exists(path):
    os.makedirs(path, exist_ok=True)

test_shallow_water_state.write_NetCDF(filename)

comm = MPI.Finalize()


#############
# Read the netCDF file as well to verify the output fields are correct
#############
dataset = Dataset(filename, 'r', diskless=True, persist=True)

print(dataset)

dataset.close()
