"""Main driver for the Shallow Water Model."""

from mpi4py import MPI
import numpy as np
from netCDF4 import Dataset
import gt4py.gtscript as gtscript
import gt4py.storage as gt_storage

from shallow_water_model_config import ShallowWaterModelConfig
from shallow_water_geometry import ShallowWaterGeometry
from shallow_water_state import ShallowWaterState
from shallow_water_model import ShallowWaterModel

