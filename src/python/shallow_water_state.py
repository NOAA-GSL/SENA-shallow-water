from time import time
from mpi4py import MPI
import numpy as np

# remove yaml import after testing 
import yaml

from shallow_water_geometry import shallow_water_geometry

# remove below line after testing
from shallow_water_geometry_config import shallow_water_geometry_config


class shallow_water_state:

    def __init__(self, geometry: shallow_water_geometry, clock: np.float64, u=None, v=None, h=None):
        """
        Initialized a shallow water state class 

        Arguments: 
            geometry   
            clock   

        Return: 
            An initialized shallow_water_state object 
        """
        # Set the physical constant of gravity
        self.g = np.float64(9.81)
        
        # Set the geometry associated with this state
        self.geometry = geometry
        
        # Get the domain index range for this patch from the geometry
        self.xps = self.geometry.get_xps()
        self.xpe = self.geometry.get_xpe()
        self.yps = self.geometry.get_yps()
        self.ype = self.geometry.get_ype()

        # Get the memory allocation index range for this patch from the geometry
        self.xms = self.geometry.get_xms()
        self.xme = self.geometry.get_xme()
        self.yms = self.geometry.get_yms()
        self.yme = self.geometry.get_yme()

        # Allocate u, v, h 
        self.u = np.zeros((self.xme - self.xms + 1, self.yme - self.yms + 1))
        self.v = np.zeros((self.xme - self.xms + 1, self.yme - self.yms + 1))
        self.h = np.zeros((self.xme - self.xms + 1, self.yme - self.yms + 1))

        # Initialize u 
        if (u):
            for j in range(self.yps, self.ype + 1):
                for i in range(self.xps, self.xpe + 1):
                    self.u[i - self.xms, j - self.yms] = u[i - self.xps, j - self.yps]

        # Initialize v 
        if (v):
            for j in range(self.yps, self.ype + 1):
                for i in range(self.xps, self.xpe + 1):
                    self.v[i - self.xms, j - self.yms] = v[i - self.xps, j - self.yps]

        # Initialize h
        if (h):
            for j in range(self.yps, self.ype + 1):
                for i in range(self.xps, self.xpe + 1):
                    self.h[i - self.xms, j - self.yms] = h[i - self.xps, j - self.yps]



            
        

comm = MPI.COMM_WORLD 

geom_config = shallow_water_geometry_config(yamlpath="../../parm/shallow_water_test.yml")

# geom_config(yamlpath="../../parm/shallow_water_test.yml")

print("geom_config :", geom_config)

geom = shallow_water_geometry(geometry=geom_config, mpi_comm=comm)

print(geom.__dict__)

test_shallow_water_state = shallow_water_state(geometry=geom, clock=0)

print(test_shallow_water_state.geometry.__dict__)

print(test_shallow_water_state.u)
