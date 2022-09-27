from mpi4py import MPI
import numpy as np
# remove yaml import after testing 
import yaml

from shallow_water_geometry import shallow_water_geometry

# remove below line after testing
from shallow_water_geometry_config import shallow_water_geometry_config

# remove below line after testing
with open('../../parm/shallow_water.yml', 'r') as file:
    param = yaml.safe_load(file)


mpi_comm = MPI.COMM_WORLD 

class shallow_water_state(shallow_water_geometry):

    def __init__(self):

        print(self)
        

    def __call(self):
        pass


geom_config = shallow_water_geometry_config(param)

print(geom_config)

geom = shallow_water_geometry(param)

print(geom)

test_shallow_water_state = shallow_water_state()

print(test_shallow_water_state.__dict__)

