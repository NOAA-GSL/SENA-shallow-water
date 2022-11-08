from shallow_water_geometry_config import ShallowWaterGeometryConfig
from shallow_water_geometry import ShallowWaterGeometry
import numpy as np
from mpi4py import MPI
import gt4py.storage as gt_storage

backend="numpy"

class ShallowWaterState:

    def __init__(self, geometry, config, u=None, v=None, h=None, clock=0):

        # Physical constants
        _g  = 9.81

        # Set the geometry associated with this state
        self.geometry = geometry

        # Set the config for use in GT4Py storage creation
        self.config = config

        # Initialize u
        self.u = gt_storage.zeros(shape=(self.geometry.xme - self.geometry.xms + 1, self.geometry.yme - self.geometry.yms + 1),
                                  dtype=np.float64, backend=self.config.backend, default_origin=(1,1))
        if (u) is not None:
            for i in range(geometry.xps, geometry.xpe + 1):
                for j in range(geometry.yps, geometry.ype + 1):
                    self.u[i - geometry.xms, j - geometry.yms] = u[i - geometry.xps, j - geometry.yps]

        # Initialize v
        self.v = gt_storage.zeros(shape=(self.geometry.xme - self.geometry.xms + 1, self.geometry.yme - self.geometry.yms + 1),
                                  dtype=np.float64, backend=self.config.backend, default_origin=(1,1))
        if (v) is not None:
            for i in range(geometry.xps, geometry.xpe + 1):
                for j in range(geometry.yps, geometry.ype + 1):
                    self.v[i - geometry.xms, j - geometry.yms] = v[i - geometry.xps, j - geometry.yps]

        # Initialize h
        self.h = gt_storage.zeros(shape=(self.geometry.xme - self.geometry.xms + 1, self.geometry.yme - self.geometry.yms + 1),
                                  dtype=np.float64, backend=self.config.backend, default_origin=(1,1))
        if (h) is not None:
            for i in range(geometry.xps, geometry.xpe + 1):
                for j in range(geometry.yps, geometry.ype + 1):
                    self.h[i - geometry.xms, j - geometry.yms] = h[i - geometry.xps, j - geometry.yps]

        # ! Calculate the maximum wave speed from h
        _max_h = np.zeros(1, np.float64)
        _local_max = np.full(1, np.amax(self.h))
        geometry.communicator.Allreduce(_local_max, _max_h, op=MPI.MAX)
        self.max_wavespeed = (_g * _max_h)**0.5

        # Initialize clock
        if (clock):
            self.clock = clock
        else:
            self.clock = 0.0


    def exchange_halo(self):

        # Get the MPI communicator from the geometry
        _communicator = self.geometry.communicator

        # Get the index ranges for this patch
        _xms = self.geometry.xms
        _xme = self.geometry.xme
        _yms = self.geometry.yms
        _yme = self.geometry.yme
        _xps = self.geometry.xps
        _xpe = self.geometry.xpe
        _yps = self.geometry.yps
        _ype = self.geometry.ype

        # Get the extents of the domain
        _npx = self.geometry.npx
        _npy = self.geometry.npy

        # Get MPI ranks of the neighbors of this patch
        _north = self.geometry.north
        _south = self.geometry.south
        _west = self.geometry.west
        _east = self.geometry.east

        # Set MPI send/recv tags
        _ntag = 1
        _stag = 2
        _wtag = 3
        _etag = 4

        # Post the non-blocking receive half of the exhange first to reduce overhead
        _nrequests = 0
        _irequests = []
        _nrecvbuffer = np.zeros((_npx, 3))
        if (_north != -1):
            _irequests.append(_communicator.Irecv(_nrecvbuffer, _north, _stag))
            _nrequests = _nrequests + 1

        _srecvbuffer = np.zeros((_npx, 3))
        if (_south != -1):
            _irequests.append(_communicator.Irecv(_srecvbuffer, _south, _ntag))
            _nrequests = _nrequests + 1

        _wrecvbuffer = np.zeros((_npy, 3))
        if (_west != -1):
            _irequests.append(_communicator.Irecv(_wrecvbuffer, _west, _etag))
            _nrequests = _nrequests + 1

        _erecvbuffer = np.zeros((_npy, 3))
        if (_east != -1):
            _irequests.append(_communicator.Irecv(_erecvbuffer, _east, _wtag))
            _nrequests = _nrequests + 1

        # Pack the send buffers
        _nsendbuffer = np.zeros((_npx, 3))
        if (_north != -1):
            for i in range(_xps, _xpe + 1):
                _nsendbuffer[i - _xps, 0] = self.u[i - _xms, _ype - _yms]
                _nsendbuffer[i - _xps, 1] = self.v[i - _xms, _ype - _yms]
                _nsendbuffer[i - _xps, 2] = self.h[i - _xms, _ype - _yms]
        _ssendbuffer = np.zeros((_npx, 3))
        if (_south != -1):
            for i in range(_xps, _xpe + 1):
                _ssendbuffer[i - _xps, 0] = self.u[i - _xms, _yps - _yms]
                _ssendbuffer[i - _xps, 1] = self.v[i - _xms, _yps - _yms]
                _ssendbuffer[i - _xps, 2] = self.h[i - _xms, _yps - _yms]
        _wsendbuffer = np.zeros((_npy, 3))
        if (_west != -1):
            for j in range(_yps, _ype + 1):
                _wsendbuffer[j - _yps, 0] = self.u[_xps - _xms, j - _yms]
                _wsendbuffer[j - _yps, 1] = self.v[_xps - _xms, j - _yms]
                _wsendbuffer[j - _yps, 2] = self.h[_xps - _xms, j - _yms]
        _esendbuffer = np.zeros((_npy, 3))
        if (_east != -1):
            for j in range(_yps, _ype + 1):
                _esendbuffer[j - _yps, 0] = self.u[_xpe - _xms, j - _yms]
                _esendbuffer[j - _yps, 1] = self.v[_xpe - _xms, j - _yms]
                _esendbuffer[j - _yps, 2] = self.h[_xpe - _xms, j - _yms]

        # Now post the non-blocking send half of the exchange
        if (_north != -1):
            _irequests.append(_communicator.Isend(_nsendbuffer, _north, _ntag))
            _nrequests = _nrequests + 1

        if (_south != -1):
            _irequests.append(_communicator.Isend(_ssendbuffer, _south, _stag))
            _nrequests = _nrequests + 1

        if (_west != -1):
            _irequests.append(_communicator.Isend(_wsendbuffer, _west, _wtag))
            _nrequests = _nrequests + 1

        if (_east != -1):
            _irequests.append(_communicator.Isend(_esendbuffer, _east, _etag))
            _nrequests = _nrequests + 1

        # Wait for the exchange to complete
        if (_nrequests > 0):
            MPI.Request.Waitall(_irequests)

        # Unpack the receive buffers
        if (_north != -1):
            for i in range(_xps, _xpe + 1):
                self.u[i - _xms, _yme - _yms] = _nrecvbuffer[i - _xps, 0]
                self.v[i - _xms, _yme - _yms] = _nrecvbuffer[i - _xps, 1]
                self.h[i - _xms, _yme - _yms] = _nrecvbuffer[i - _xps, 2]
        if (_south != -1):
            for i in range(_xps, _xpe + 1):
                self.u[i - _xms, _yms - _yms] = _srecvbuffer[i - _xps, 0]
                self.v[i - _xms, _yms - _yms] = _srecvbuffer[i - _xps, 1]
                self.h[i - _xms, _yms - _yms] = _srecvbuffer[i - _xps, 2]
        if (_west != -1):
            for j in range(_yps, _ype + 1):
                self.u[_xms - _xms, j - _yms] = _wrecvbuffer[j - _yps, 0]
                self.v[_xms - _xms, j - _yms] = _wrecvbuffer[j - _yps, 1]
                self.h[_xms - _xms, j - _yms] = _wrecvbuffer[j - _yps, 2]
        if (_east != -1):
            for j in range(_yps, _ype + 1):
                self.u[_xme - _xms, j - _yms] = _erecvbuffer[j - _yps, 0]
                self.v[_xme - _xms, j - _yms] = _erecvbuffer[j - _yps, 1]
                self.h[_xme - _xms, j - _yms] = _erecvbuffer[j - _yps, 2]


    def scatter(self, u_full, v_full, h_full):

        # Get the MPI communicator from the geometry
        _communicator = self.geometry.communicator

        # Get the number of MPI ranks from the geometry
        _nranks = self.geometry.nranks

        # Get the MPI rank of this process from the geometry
        _myrank = self.geometry.rank

        # Get the local indices (including the halo) from the geometry
        _xms_local = np.full(1, self.geometry.xms, dtype=int)
        _xme_local = np.full(1, self.geometry.xme, dtype=int)
        _yms_local = np.full(1, self.geometry.yms, dtype=int)
        _yme_local = np.full(1, self.geometry.yme, dtype=int)

        # Calculate the local number of elements
        _nelements_local = (_xme_local - _xms_local + 1) * (_yme_local - _yms_local + 1)

        # Allocate space for the indices of each rank
        if (_myrank == 0):
            _xms = np.empty((_nranks), dtype=int)
            _xme = np.empty((_nranks), dtype=int)
            _yms = np.empty((_nranks), dtype=int)
            _yme = np.empty((_nranks), dtype=int)
        else:
            _xms = np.empty((1), dtype=int)
            _xme = np.empty((1), dtype=int)
            _yms = np.empty((1), dtype=int)
            _yme = np.empty((1), dtype=int)

        # Gather the local indices for each rank
        _communicator.Gather(_xms_local, _xms)
        _communicator.Gather(_xme_local, _xme)
        _communicator.Gather(_yms_local, _yms)
        _communicator.Gather(_yme_local, _yme)

        # Calculate the number of elements to send to each rank
        if (_myrank == 0):
            _nsend_elements = np.empty((_nranks), dtype=int)
            for n in range(_nranks):
                _nsend_elements[n] = (_xme[n] - _xms[n] + 1) * (_yme[n] - _yms[n] + 1)
        else:
            _nsend_elements = np.empty((1))

        # Calculate the send buffer offsets for each rank
        if (_myrank == 0):
            _send_offsets =  np.empty((_nranks), dtype=int)
            _send_offsets[0] = 0
            for n in range(1,_nranks):
                 _send_offsets[n] = _send_offsets[n-1] + _nsend_elements[n-1]
        else:
            _send_offsets = np.empty((1))

        # Allocate a send buffer for scattering u, v, and h
        if (_myrank == 0):
            _send_buffer = np.empty((_nsend_elements.sum()))
        else:
            _send_buffer = np.empty((1))

        # Fill the send buffer and scatter u, v, h
        if (_myrank == 0):
            for n in range(_nranks):
                _send_buffer[_send_offsets[n]:_send_offsets[n] + _nsend_elements[n]] = (u_full[_xms[n]-1:_xme[n], _yms[n]-1:_yme[n]]).flatten()
        _communicator.Scatterv([_send_buffer, _nsend_elements, _send_offsets, MPI.DOUBLE], self.u)
        if (_myrank == 0):
            for n in range(_nranks):
                _send_buffer[_send_offsets[n]:_send_offsets[n] + _nsend_elements[n]] = (v_full[_xms[n]-1:_xme[n], _yms[n]-1:_yme[n]]).flatten()
        _communicator.Scatterv([_send_buffer, _nsend_elements, _send_offsets, MPI.DOUBLE], self.v)
        if (_myrank == 0):
            for n in range(_nranks):
                _send_buffer[_send_offsets[n]:_send_offsets[n] + _nsend_elements[n]] = (h_full[_xms[n]-1:_xme[n], _yms[n]-1:_yme[n]]).flatten()
        _communicator.Scatterv([_send_buffer, _nsend_elements, _send_offsets, MPI.DOUBLE], self.h)


    def gather(self):

        # Get the MPI communicator from the geometry
        _communicator = self.geometry.communicator

        # Get the number of MPI ranks from the geometry
        _nranks = self.geometry.nranks

        # Get the MPI rank of this process from the geometry
        _myrank = self.geometry.rank

        # Get the local indices (excluding the halo) from the geometry
        _xps_local = self.geometry.xps
        _xpe_local = self.geometry.xpe
        _yps_local = self.geometry.yps
        _ype_local = self.geometry.ype

        # Get the local indices (including the halo) from the geometry
        _xms_local = self.geometry.xms
        _xme_local = self.geometry.xme
        _yms_local = self.geometry.yms
        _yme_local = self.geometry.yme

        # Allocate full domain arrays for the gather
        if (_myrank == 0):
            _u_full = np.zeros((self.geometry.nx, self.geometry.ny))
            _v_full = np.zeros((self.geometry.nx, self.geometry.ny))
            _h_full = np.zeros((self.geometry.nx, self.geometry.ny))
        else:
            _u_full = None
            _v_full = None
            _h_full = None

        # Calculate the local number of elements
        _nelements_local = (_xpe_local - _xps_local + 1) * (_ype_local - _yps_local + 1)

        # Allocate space for the indices of each rank
        if (_myrank == 0):
            _xps = np.empty((_nranks), dtype=int)
            _xpe = np.empty((_nranks), dtype=int)
            _yps = np.empty((_nranks), dtype=int)
            _ype = np.empty((_nranks), dtype=int)
        else:
            _xps = np.empty((1), dtype=int)
            _xpe = np.empty((1), dtype=int)
            _yps = np.empty((1), dtype=int)
            _ype = np.empty((1), dtype=int)

        # Gather the local indices for each rank
        _communicator.Gather(np.asarray(_xps_local), _xps)
        _communicator.Gather(np.asarray(_xpe_local), _xpe)
        _communicator.Gather(np.asarray(_yps_local), _yps)
        _communicator.Gather(np.asarray(_ype_local), _ype)

        # Calculate the number of elements that will be receieved from each rank
        if (_myrank == 0):
            _nrecv_elements = np.empty((_nranks), dtype=int)
            for n in range(_nranks):
                _nrecv_elements[n] = (_xpe[n] - _xps[n] + 1) * (_ype[n] - _yps[n] + 1)
        else:
            _nrecv_elements = np.empty((1))

        # Calculate the receive buffer offsets for each rank
        if (_myrank == 0):
            _recv_offsets =  np.empty((_nranks), dtype=int)
            _recv_offsets[0] = 0
            for n in range(1,_nranks):
                 _recv_offsets[n] = _recv_offsets[n-1] + _nrecv_elements[n-1]
        else:
            _recv_offsets = np.empty((1))

        # Allocate a receive buffer for gathering u, v, and h
        if (_myrank == 0):
            _recv_buffer = np.empty((_nrecv_elements.sum()))
        else:
            _recv_buffer = np.empty((1))

        # Gather u, v, and h from all ranks and unpack into full size arrays
        _communicator.Gatherv(np.ascontiguousarray(self.u[_xps_local-_xms_local:_xpe_local-_xms_local+1, _yps_local-_yms_local:_ype_local-_yms_local+1]), [_recv_buffer, _nrecv_elements, _recv_offsets, MPI.DOUBLE])
        if (_myrank == 0):
            for n in range(_nranks):
                _u_full[_xps[n]-1:_xpe[n], _yps[n]-1:_ype[n]] = np.reshape(_recv_buffer[_recv_offsets[n]:_recv_offsets[n]+_nrecv_elements[n]], (_xpe[n] - _xps[n] + 1, _ype[n] - _yps[n] + 1))
        _communicator.Gatherv(np.ascontiguousarray(self.v[_xps_local-_xms_local:_xpe_local-_xms_local+1, _yps_local-_yms_local:_ype_local-_yms_local+1]), [_recv_buffer, _nrecv_elements, _recv_offsets, MPI.DOUBLE])
        if (_myrank == 0):
            for n in range(_nranks):
                _v_full[_xps[n]-1:_xpe[n], _yps[n]-1:_ype[n]] = np.reshape(_recv_buffer[_recv_offsets[n]:_recv_offsets[n]+_nrecv_elements[n]], (_xpe[n] - _xps[n] + 1, _ype[n] - _yps[n] + 1))
        _communicator.Gatherv(np.ascontiguousarray(self.h[_xps_local-_xms_local:_xpe_local-_xms_local+1, _yps_local-_yms_local:_ype_local-_yms_local+1]), [_recv_buffer, _nrecv_elements, _recv_offsets, MPI.DOUBLE])
        if (_myrank == 0):
            for n in range(_nranks):
                _h_full[_xps[n]-1:_xpe[n], _yps[n]-1:_ype[n]] = np.reshape(_recv_buffer[_recv_offsets[n]:_recv_offsets[n]+_nrecv_elements[n]], (_xpe[n] - _xps[n] + 1, _ype[n] - _yps[n] + 1))

        # Return the full domain arrays (valid on rank 0 only)
        return _u_full, _v_full, _h_full


    def write(self, filename):

        from netCDF4 import Dataset
        import time

        # Get the MPI rank of this process from the geometry
        _myrank = self.geometry.rank

        # Get the full grid size from the geometry
        _nx = self.geometry.nx
        _ny = self.geometry.ny

        # Gather full u, v, and h
        _u_full, _v_full, _h_full = self.gather()

        # Write the full state
        if (_myrank == 0):

            # Open new file, overwriting previous contents
            _dataset = Dataset(filename, "w")

            # Write Global Attributes
            _dataset.creation_date = time.strftime("%Y/%m/%d %H:%M:%S",time.localtime())
            _dataset.model_name = "Shallow Water"
            _dataset.xmax = self.geometry.xmax
            _dataset.ymax = self.geometry.ymax
            _dataset.clock = self.clock

            # Define the x/y dimensions
            _nxDim = _dataset.createDimension("nx", _nx)
            _nyDim = _dataset.createDimension("ny", _ny)

            # Define the x/y fields
            _xVar = _dataset.createVariable("x", "f8", (_nxDim,))
            _xVar.long_name = "x"
            _xVar.units = "Nondimensional"
            _yVar = _dataset.createVariable("y", "f8", (_nyDim,))
            _yVar.long_name = "y"
            _yVar.units = "Nondimensional"

            # Define the u/v/h variables
            _uVar = _dataset.createVariable("U", "f8", (_nxDim, _nyDim))
            _uVar.long_name = "Zonal Velocity"
            _uVar.units = "m / s"
            _vVar = _dataset.createVariable("V", "f8", (_nxDim, _nyDim))
            _vVar.long_name = "Meridional Velocity"
            _vVar.units = "m / s"
            _hVar = _dataset.createVariable("H", "f8", (_nxDim, _nyDim))
            _hVar.long_name = "Pressure Surface Height"
            _hVar.units = "m"

            # Fill the x variable
            _dx = self.geometry.dx
            for i in range(_nx):
                _xVar[i] = i * _dx

            # Fill the y variable
            _dy = self.geometry.dy
            for i in range(_ny):
                _yVar[i] = i * _dy

            # Fill the u, v, h variables
            _uVar[:,:] = _u_full[:,:]
            _vVar[:,:] = _v_full[:,:]
            _hVar[:,:] = _h_full[:,:]

            # Close the NetCDF file
            _dataset.close()

    def read(self, filename):

        from netCDF4 import Dataset
        import time

        # Get the MPI rank of this process from the geometry
        _myrank = self.geometry.rank

        # Read the full state
        if (_myrank == 0):

            # Open new file for reading
            _dataset = Dataset(filename, "r")

            # Read global attributes
            _xmax = _dataset.xmax
            _ymax = _dataset.ymax
            _clock = np.full(1, _dataset.clock)

            # Read the model dimensions
            _nx = len(_dataset.dimensions["nx"])
            _ny = len(_dataset.dimensions["ny"])

            # Check to make sure state read in matches this state's geometry
            if (self.geometry.nx != _nx or self.geometry.ny != _ny or
                self.geometry.xmax != _xmax or self.geometry.ymax != _ymax):
                self.geometry.communicator.Abort()

            # Get the u, v, h
            _u_full = np.empty((_nx, _ny))
            _u_full[:,:] = _dataset.variables["U"][:,:]
            _v_full = np.empty((_nx, _ny))
            _v_full[:,:] = _dataset.variables["V"][:,:]
            _h_full = np.empty((_nx, _ny))
            _h_full[:,:] = _dataset.variables["H"][:,:]

        else:
            _u_full = np.empty(1)
            _v_full = np.empty(1)
            _h_full = np.empty(1)
            _clock = np.empty(1)

        # Scatter the full state
        self.scatter(_u_full, _v_full, _h_full)

        # Broadcast the clock
        self.geometry.communicator.Bcast(_clock)
        self.clock = _clock[0]

    def advance_clock(self, dt):

        self.clock = self.clock + dt
