from shallow_water_geometry_config import ShallowWaterGeometryConfig 
from mpi4py import MPI

class ShallowWaterGeometry:
    """
    Class for the Shallow Water Geometry, extending the ShallowWaterGeometryConfig class.
    """

    def __init__(self, config: ShallowWaterGeometryConfig, mpi_comm: MPI.Intracomm):
        """
        Initialize this class based off of the ShallowWaterGeometryConfig class.
            Arguments:
                geometry:  class,   Geometry consisting of nx, ny, xmax, ymax
                mpi_comm:  class,   MPI Communication
            Return:
                An initialized geometry class based off a geometry configuration.
        """

        # Get geometry grid dimensions from config
        self.nx = config.nx
        self.ny = config.ny

        # Get geometry domain dimensions from config 
        self.xmax = config.xmax
        self.ymax = config.ymax

        # Define the geometry grid spacing
        self.dx = self.xmax / (self.nx - 1.0)
        self.dy = self.ymax / (self.ny - 1.0)

        # Initialize the MPI communicator
        self.communicator = mpi_comm

        # Get the number of MPI ranks
        self.nranks = mpi_comm.Get_size()

        # Get the rank of this MPI process
        self.rank = mpi_comm.Get_rank()

        # Compute the size of the processor grid
        _factor = int(self.nranks**0.5 + 0.5)

        while (self.nranks % _factor != 0 ):
            _factor = _factor - 1

        if (self.nx >= self.ny):
            self.nxprocs = _factor
        else:
            self.nxprocs = self.nranks // _factor

        self.nyprocs = self.nranks // self.nxprocs

        # Compute the processor coordinate for this rank
        self.xproc = self.rank % self.nxprocs
        self.yproc = self.rank // self.nxprocs

        # Compute the ranks of our neighbors
        if (self.yproc == self.nyprocs - 1):
            self.north = -1
        else:
            self.north = (self.yproc + 1) * self.nxprocs + self.xproc
        if (self.yproc == 0):
            self.south = -1
        else:
            self.south = (self.yproc - 1) * self.nxprocs + self.xproc
        if (self.xproc == 0):
            self.west = -1
        else:
            self.west  = self.yproc * self.nxprocs + self.xproc - 1
        if (self.xproc == self.nxprocs - 1):
            self.east = -1
        else:
            self.east = self.yproc * self.nxprocs + self.xproc + 1
  
        # Compute the size of the x and y extents for this patch of the domain
        self.npx = self.nx // self.nxprocs
        if (self.xproc >= (self.nxprocs - (self.nx % self.nxprocs))):
            self.npx = self.npx + 1

        self.npy = self.ny // self.nyprocs
        if (self.yproc >= (self.nyprocs - (self.ny % self.nyprocs))):
            self.npy = self.npy + 1
  
        # Compute the start/end indices for this patch of the domain
        self.xps = self.nx // self.nxprocs * self.xproc + 1 + max(0, self.xproc - (self.nxprocs - (self.nx % self.nxprocs)))
        self.xpe = self.xps + self.npx - 1
        self.yps = self.ny // self.nyprocs * self.yproc + 1 + max(0, self.yproc - (self.nyprocs - (self.ny % self.nyprocs)))
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
