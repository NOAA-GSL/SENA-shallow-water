from mpi4py import MPI
import numpy as np

from shallow_water_geometry_config import ShallowWaterGeometryConfig


class ShallowWaterGeometry:
    """
    Class for the geometry, extending the ShallowWaterGeometryConfig class.
    """

    def __init__(self, geometry: ShallowWaterGeometryConfig, mpi_comm: MPI.Comm):
        """
        Initialize this class based off of the ShallowWaterGeometryConfig class.

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
        self.dx = self.xmax // (np.float64(self.nx) - np.float64(1.0))
        self.dy = self.ymax // (np.float64(self.ny) - np.float64(1.0))
        
        # Initialize the MPI communicator
        self.mpi_comm = mpi_comm

        # Get the total number of MPI ranks and the rank of this MPI Process
        self.nranks = self.mpi_comm.Get_size()
        self.rank   = self.mpi_comm.Get_rank()
        
        _factor = np.int32(self.nranks**0.5 + 0.5)
        
        # Compute the size of the processor grid 
        while (self.nranks % _factor != 0):
            _factor = _factor - 1
        if (self.nx >= self.ny):
            self.nxprocs = _factor
        else:
            self.nxprocs = np.int32(self.nranks // _factor)

        self.nyprocs = np.int32(self.nranks // self.nxprocs)
        
        # Compute the processor coordinate for this rank
        self.xproc = np.int32(self.rank % self.nxprocs)
        self.yproc = np.int32(self.rank // self.nxprocs)

        # Compute the MPI ranks of our neighbors
        if (self.yproc == self.nyprocs -1):
            self.north = np.int32(-1)
        else:
            self.north = np.int32((self.yproc +1) * self.nxprocs + self.xproc)
        if (self.yproc == 0):
            self.south = np.int32(-1)
        else:
            self.south = np.int32((self.yproc - 1) * self.nxprocs + self.xproc)
        if (self.xproc == 0):
            self.west = np.int32(-1)
        else:
            self.west  = np.int32(self.yproc * self.nxprocs + self.xproc - 1)
        if (self.xproc == self.nxprocs - 1):
            self.east = np.int32(-1)
        else:
            self.east = np.int32(self.yproc * self.nxprocs + self.xproc + 1)
        
        # Compute the size of the x and y extents for this patch of the domain
        self.npx = np.int32(self.nx // self.nxprocs)
        if (self.xproc >= (self.nxprocs - (self.nx % self.nxprocs))):
          self.npx = self.npx + 1
        
        self.npy = np.int32(self.ny // self.nyprocs)
        if (self.yproc >= (self.nyprocs - (self.ny % self.nyprocs))):
          self.npy = self.npy + 1
        
        # Compute the start/end indices for this patch of the domain
        self.xps = np.int32(self.nx / self.nxprocs * self.xproc + 1 + max(0, self.xproc - (self.nxprocs - (self.nx % self.nxprocs))))
        self.xpe = np.int32(self.xps + self.npx - 1)
        self.yps = np.int32(self.ny / self.nyprocs * self.yproc + 1 + max(0, self.yproc - (self.nyprocs - (self.ny % self.nyprocs))))
        self.ype = np.int32(self.yps + self.npy - 1)

        # Compute the start/end indices for the interior points and memory allocated for this patch
        if (self.north == -1):
            self.yte = np.int32(self.ype - 1)
            self.yme = np.int32(self.ype)
        else:
            self.yte = np.int32(self.ype)
            self.yme = np.int32(self.ype + 1)

        if (self.south == -1):
            self.yts = np.int32(self.yps + 1)
            self.yms = np.int32(self.yps)
        else:
            self.yts = np.int32(self.yps)
            self.yms = np.int32(self.yps - 1)

        if (self.west == -1):
            self.xts = np.int32(self.xps + 1)
            self.xms = np.int32(self.xps)
        else:
            self.xts = np.int32(self.xps)
            self.xms = np.int32(self.xps - 1)

        if (self.east == -1):
            self.xte = np.int32(self.xpe - 1)
            self.xme = np.int32(self.xpe)
        else:
            self.xte = np.int32(self.xpe)
            self.xme = np.int32(self.xpe + 1)
