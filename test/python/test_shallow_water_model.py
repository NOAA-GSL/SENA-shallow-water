from mpi4py import MPI
import numpy as np

import sys
sys.path.append("../../src/python")

from shallow_water_geometry_config import ShallowWaterGeometryConfig
from shallow_water_geometry import ShallowWaterGeometry
from shallow_water_state import ShallowWaterState
from shallow_water_model_config import ShallowWaterModelConfig
from shallow_water_model import ShallowWaterModel


comm = MPI.COMM_WORLD 

comm.Get_rank()

geom_config = ShallowWaterGeometryConfig(yamlpath="../test_input/test_shallow_water_config.yml")

geom = ShallowWaterGeometry(geometry=geom_config, mpi_comm=comm)

state = ShallowWaterState(geometry=geom, clock=0.0)

filepath = "../test_input/test_shallow_water_reader.nc"

state.read_NetCDF(filepath)

model_config = ShallowWaterModelConfig(yamlpath="../test_input/test_shallow_water_config.yml")

model = ShallowWaterModel(config=model_config, geometry=geom)

print(model.__dict__)

model.adv_nsteps_model(state=state, nsteps=np.int32(1))

comm = MPI.Finalize()
