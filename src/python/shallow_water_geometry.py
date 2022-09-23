from mpi4py import MPI
import numpy as np
import yaml

from shallow_water_geometry_config import shallow_water_geometry_config

g  = np.float64(9.81)

with open('../../parm/shallow_water.yml', 'r') as file:
    parm = yaml.safe_load(file)

mpi_comm = MPI.COMM_WORLD 

class shallow_water_geometry(shallow_water_geometry_config):
    """Class for the geometry, extending the shallow_water_geometry_config class.
        Must be passed the loaded data from a yaml file as an argument to call the super()
        method of initializing the parent class. 
    """

    def __init__(self, data):
        """
            Initialize this class as a child of the shallow_water_geometry_config class. 
        """
        super().__init__(data)

        self.dx       = None  # Grid spacing in the x direction 
        self.dy       = None  # Grid spacing in the y direction  

        self.mpi_comm = None  # MPI communicator
        self.ierr     = None
        self.factor   = None
        self.nranks   = None  # Total number of MPI ranks
        self.rank     = None  # MPI rank of this task
        self.nxprocs  = None  # Size of the processor grid in the x direction
        self.nyprocs  = None  # Size of the processor grid in the y direction
        self.xproc    = None  # Processor grid coordinate of this MPI task in the x direction
        self.yproc    = None  # Processor grid coordinate of this MPI task in the y direction
        
        self.north    = None  # MPI rank of northern neighbor
        self.south    = None  # MPI rank of southern neighbor
        self.west     = None  # MPI rank of western neighbor
        self.east     = None  # MPI rank of eastern neighbor
        self.npx      = None  # Extent of the domain for this patch in x/y directions
        self.npy      = None  # Extent of the domain for this patch in x/y directions
        self.xps      = None  # Start indices of this grid patch in the x direction
        self.xpe      = None  # End indices of this grid patch in the x direction
        self.yps      = None  # Start indices of this grid patch in the y direction
        self.ype      = None  # End indices of this grid patch in the y direction
        self.xts      = None  # Start indices of interior points for this grid patch in the x direction
        self.xte      = None  # End indices of interior points for this grid patch in the x direction
        self.yts      = None  # Start indices of interior points for this grid patch in the y direction
        self.yte      = None  # End indices of interior points for this grid patch in the y direction
        self.xms      = None  # Start indices of the memory allocated for this grid patch in the x direction
        self.xme      = None  # End indices of the memory allocated for this grid patch in the x direction
        self.yms      = None  # Start indices of the memory allocated for this grid patch in the y direction
        self.yme      = None  # End indices of the memory allocated for this grid patch in the y direction


    # TODO can all of these functions be substituted by Python's getattr() method? 
    def get_communicator(self):
        return self.mpi_comm

    def get_rank(self):
        return self.rank
    
    def get_nranks(self):
        return self.nranks

    def get_nx(self):
        return self.nx
    
    def get_ny(self):
        return self.ny
    
    def get_xmax(self):
        return self.xmax
    
    def get_ymax(self):
        return self.ymax

    def get_dx(self):
        return self.dx
    
    def get_dy(self):
        return self.dy

    def get_north(self):
        return self.north

    def get_south(self):
        return self.south

    def get_west(self):
        return self.west

    def get_east(self):
        return self.east

    def get_npx(self):
        return self.npx
    
    def get_npy(self):
        return self.npy
    
    def get_xps(self):
        return self.xps

    def get_xpe(self):
        return self.xpe
    
    def get_yps(self):
        return self.yps

    def get_ype(self):
        return self.ype

    def get_xts(self):
        return self.xts

    def get_xte(self):
        return self.xte
    
    def get_yts(self):
        return self.yts
    
    def get_yte(self):
        return self.yte
    
    def get_xms(self):
        return self.xms
    
    def get_xme(self):
        return self.xme    
    
    def get_yms(self):
        return self.yms
    
    def get_yme(self):
        return self.yme


    def __call__(self, mpi_comm):

        # Define the geometry grid spacing
        self.dx       = self.xmax / (np.float64(self.nx) - np.float64(1.0))
        self.dy       = self.ymax / (np.float64(self.ny) - np.float64(1.0))
        
        # Initialize the MPI communicator
        self.mpi_comm = mpi_comm
        # Get number of MPI ranks and the rank of this MPI Process
        self.nranks = self.mpi_comm.Get_size()
        self.rank   = self.mpi_comm.Get_rank()
        self.factor = self.nranks**0.5 + 0.5
        
        # Compute the size of the processor grid
        while (self.nranks % self.factor != 0):
            self.factor = self.factor - 1
        if (self.nx >= self.ny):
            self.nxprocs = self.factor
        else:
            self.nxprocs = self.nranks / self.factor

        self.nyprocs = self.nranks / self.nxprocs
        
        # Compute the processor coordinate for this rank
        self.xproc = self.rank % self.nxprocs
        self.yproc = self.rank / self.nxprocs

        # Compute the ranks of our neighbors
        if (self.yproc == self.nyprocs -1):
            self.north = -1
        else:
            self.north = (self.yproc +1) * self.nxprocs + self.xproc
        if (self.yproc == 0):
            self.south = -1
        else:
            self.south    = (self.yproc - 1) * self.nxprocs + self.xproc
        if (self.xproc == 0):
            self.west = -1
        else:
            self.west  = self.yproc * self.nxprocs + self.xproc - 1
        if (self.xproc == self.nxprocs - 1):
            self.east = -1
        else:
            self.east = self.yproc * self.nxprocs + self.xproc + 1
        
        # Compute the size of the x and y extents for this patch of the domain
        self.npx   = self.nx / self.nxprocs
        if (self.xproc >= (self.nxprocs - (self.nx % self.nxprocs))):
          self.npx = self.npx + 1
        
        self.npy = self.ny / self.nyprocs
        if (self.yproc >= (self.nyprocs - (self.ny % self.nyprocs))):
          self.npy = self.npy + 1
        
        # Compute the start/end indices for this patch of the domain
        self.xps = self.nx / self.nxprocs * self.xproc + 1 + max(0, self.xproc - (self.nxprocs - (self.nx % self.nxprocs)))
        self.xpe = self.xps + self.npx - 1
        self.yps = self.ny / self.nyprocs * self.yproc + 1 + max(0, self.yproc - (self.nyprocs - (self.ny % self.nyprocs)))
        self.ype = self.yps + self.npy - 1

        # Compute the start/end indices for the interior points and memory allocated for this patch
        if (self.north == -1):
            self.yte = self.ype - 1
            self.yme = self.ype
        else:
            self.yte = self.ype
            self.yme = self.ype + 1

        if (self.south == -1):
            self.yts = self.yps + 1
            self.yms = self.yps
        else:
            self.yts = self.yps
            self.yms = self.yps - 1

        if (self.west == -1):
            self.xts = self.xps + 1
            self.xms = self.xps
        else:
            self.xts = self.xps
            self.xms = self.xps - 1

        if (self.east == -1):
            self.xte = self.xpe - 1
            self.xme = self.xpe
        else:
            self.xte = self.xpe
            self.xme = self.xpe + 1

        

geom_config = shallow_water_geometry_config(parm)

print(geom_config)

geom = shallow_water_geometry(parm)

print(geom)

print(geom.__dict__)

geom(mpi_comm)

print(geom.get_communicator())

print(getattr(geom, "east"))

print(geom.__dict__)



