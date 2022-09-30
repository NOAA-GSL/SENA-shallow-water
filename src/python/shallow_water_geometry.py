from mpi4py import MPI
import numpy as np

from shallow_water_geometry_config import shallow_water_geometry_config


class shallow_water_geometry:
    """
    Class for the geometry, extending the shallow_water_geometry_config class.
    """

    def __init__(self, geometry: shallow_water_geometry_config, mpi_comm: MPI.Comm):
        """
        Initialize this class based off of the shallow_water_geometry_config class.

            Arguments:
                geometry:  class,   Geometry consisting of nx, ny, xmax, ymax
                mpi_comm:  class,   MPI Communication

            Return:
                An initialized geometry class based off a geometry configuration.
        """

        self.nx   = geometry.nx   
        self.ny   = geometry.ny   
        self.xmax = geometry.xmax 
        self.ymax = geometry.ymax 

        # Define the geometry grid spacing
        self.dx       = self.xmax / (np.float64(self.nx) - np.float64(1.0))
        self.dy       = self.ymax / (np.float64(self.ny) - np.float64(1.0))
        
        # Initialize the MPI communicator
        self.mpi_comm = mpi_comm
        # Get the total number of MPI ranks and the rank of this MPI Process
        self.nranks = self.mpi_comm.Get_size()
        self.rank   = self.mpi_comm.Get_rank()
        #
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

        # Compute the MPI ranks of our neighbors
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

    # TODO - Can all of these functions be substituted by Python's getattr() method? 
    def get_communicator(self):
        return self.mpi_comm

    def get_rank(self):
        return np.int32(self.rank)
    
    def get_nranks(self):
        return np.int32(self.nranks)

    def get_nx(self):
        return np.int32(self.nx)
    
    def get_ny(self):
        return np.int32(self.ny)
    
    def get_xmax(self):
        return np.float64(self.xmax)
    
    def get_ymax(self):
        return np.float64(self.ymax)

    def get_dx(self):
        return np.float64(self.dx)
    
    def get_dy(self):
        return np.float64(self.dy)

    def get_north(self):
        return np.int32(self.north)

    def get_south(self):
        return np.int32(self.south)

    def get_west(self):
        return np.int32(self.west)

    def get_east(self):
        return np.int32(self.east)

    def get_npx(self):
        return np.int32(self.npx)
    
    def get_npy(self):
        return np.int32(self.npy)
    
    def get_xps(self):
        return np.int32(self.xps)

    def get_xpe(self):
        return np.int32(self.xpe)
    
    def get_yps(self):
        return np.int32(self.yps)

    def get_ype(self):
        return np.int32(self.ype)

    def get_xts(self):
        return np.int32(self.xts)

    def get_xte(self):
        return np.int32(self.xte)
    
    def get_yts(self):
        return np.int32(self.yts)
    
    def get_yte(self):
        return np.int32(self.yte)
    
    def get_xms(self):
        return np.int32(self.xms)
    
    def get_xme(self):
        return np.int32(self.xme)
    
    def get_yms(self):
        return np.int32(self.yms)
    
    def get_yme(self):
        return np.int32(self.yme)

