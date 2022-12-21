import numpy as np
from mpi4py import MPI
import gt4py.gtscript as gtscript
import gt4py.storage as gt_storage

from shallow_water_gt4py_config import ShallowWaterGT4PyConfig
from shallow_water_geometry import ShallowWaterGeometry

# Check if CUDA is available
try:
    import cupy as cp
except ImportError:
    cp = None

class ShallowWaterState:

    def __init__(self, geometry: ShallowWaterGeometry, config: ShallowWaterGT4PyConfig, u=None, v=None, h=None, clock=0):

        # Physical constants
        _g  = 9.81

        # Set the geometry associated with this state
        self.geometry = geometry

        # Set the config for use in GT4Py storage creation
        self.backend = config.backend
        self.float_type = config.float_type
        self.field_type = gtscript.Field[gtscript.IJ, self.float_type]

        # Initialize u
        self.u = gt_storage.zeros(shape=(self.geometry.xme - self.geometry.xms + 1, self.geometry.yme - self.geometry.yms + 1),
                                  dtype=self.float_type, backend=self.backend, default_origin=(1,1))
        if (u) is not None:
            for i in range(geometry.xps, geometry.xpe + 1):
                for j in range(geometry.yps, geometry.ype + 1):
                    self.u[i - geometry.xms, j - geometry.yms] = u[i - geometry.xps, j - geometry.yps]

        # Initialize v
        self.v = gt_storage.zeros(shape=(self.geometry.xme - self.geometry.xms + 1, self.geometry.yme - self.geometry.yms + 1),
                                  dtype=self.float_type, backend=self.backend, default_origin=(1,1))
        if (v) is not None:
            for i in range(geometry.xps, geometry.xpe + 1):
                for j in range(geometry.yps, geometry.ype + 1):
                    self.v[i - geometry.xms, j - geometry.yms] = v[i - geometry.xps, j - geometry.yps]

        # Initialize h
        self.h = gt_storage.zeros(shape=(self.geometry.xme - self.geometry.xms + 1, self.geometry.yme - self.geometry.yms + 1),
                                  dtype=self.float_type, backend=self.backend, default_origin=(1,1))
        if (h) is not None:
            for i in range(geometry.xps, geometry.xpe + 1):
                for j in range(geometry.yps, geometry.ype + 1):
                    self.h[i - geometry.xms, j - geometry.yms] = h[i - geometry.xps, j - geometry.yps]

        # ! Calculate the maximum wave speed from h
        _max_h = np.zeros(1, self.float_type)
        _local_max = np.full(1, np.amax(self.h))
        geometry.communicator.Allreduce(_local_max, _max_h, op=MPI.MAX)
        self.max_wavespeed = (_g * _max_h)**0.5

        # Initialize clock
        if (clock):
            self.clock = clock
        else:
            self.clock = 0.0

        # Run a deviceSynchronize() to check that the GPU is present and ready to run
        if cp is not None and "gpu" in self.backend:
            try:
                cp.cuda.runtime.deviceSynchronize()
                self._GPU_AVAILABLE = True
            except cp.cuda.runtime.CUDARuntimeError:
                self._GPU_AVAILABLE = False
        else:
            self._GPU_AVAILABLE = False

        # Set up array method for use in MPI calls
        if (self._GPU_AVAILABLE):
            self._array = cp.array
            self._empty_like = cp.empty_like
            print("Using cp.array to create MPI buffers")
        else:
            self._array = np.array
            self._empty_like = np.empty_like
            print("Using np.array to create MPI buffers")

        # Allocate send/recv storage and buffers for halo pack/unpack and exchanges
        self._nsendstorage = gt_storage.empty(shape=(self.geometry.npx, 3), dtype=self.float_type, backend=self.backend, default_origin=(0, 0))
        self._nrecvbuff = self._empty_like(self._nsendstorage.data)
        print(f"Initial empty nrecvbuf = {self._nrecvbuff[70:80, 2]}")

        self._ssendstorage = gt_storage.empty(shape=(self.geometry.npx, 3), dtype=self.float_type, backend=self.backend, default_origin=(0, 0))
        self._srecvbuff = self._empty_like(self._ssendstorage.data)

        self._wsendstorage = gt_storage.empty(shape=(3, self.geometry.npy), dtype=self.float_type, backend=self.backend, default_origin=(0, 0))
        self._wrecvbuff = self._empty_like(self._wsendstorage.data)

        self._esendstorage = gt_storage.empty(shape=(3, self.geometry.npy), dtype=self.float_type, backend=self.backend, default_origin=(0, 0))
        self._erecvbuff = self._empty_like(self._esendstorage.data)


        # Define copy stencil for packing/unpacking halo exchange buffers
        def copy_vector(inField: self.field_type,
                        outField: self.field_type):
            #    # NOTE: FORWARD ordering is required here to disambiguate the missing k dimension
            #    #       for assignment into our 2D arrays.
            with computation(FORWARD), interval(...):
                outField = inField

        #self._copy_vector = gtscript.stencil(definition=copy_vector, backend=self.backend, device_sync=False)
        self._copy_vector = gtscript.stencil(definition=copy_vector, backend=self.backend)

    # Transparent device memory synchronization
    def device_synchronize(self):
        """Synchronize all memory communication"""
        if self._GPU_AVAILABLE:
            cp.cuda.runtime.deviceSynchronize()
            cp.cuda.get_current_stream().synchronize()
            #False

    # Send boundaries to neighboring halos for each process
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

        # Syncrhonize before MPI calls
        self.device_synchronize()
        
        # Post the non-blocking receive half of the exhange first to reduce overhead
        _nrequests = 0
        _irequests = []
        if (_north != -1):
            _irequests.append(_communicator.Irecv(self._nrecvbuff, _north, _stag))
            _nrequests = _nrequests + 1
        if (_south != -1):
            _irequests.append(_communicator.Irecv(self._srecvbuff, _south, _ntag))
            _nrequests = _nrequests + 1
        if (_west != -1):
            _irequests.append(_communicator.Irecv(self._wrecvbuff, _west, _etag))
            _nrequests = _nrequests + 1
        if (_east != -1):
            _irequests.append(_communicator.Irecv(self._erecvbuff, _east, _wtag))
            _nrequests = _nrequests + 1

        # Should not need this synchronize
        self.device_synchronize()

        # Pack the state variables into send storages
        if (_north != -1):
            self._copy_vector(self.u, self._nsendstorage, origin={"inField": (_xps-_xms,_ype-_yms), "outField": (0,0)}, domain=(_npx,1,1))
            self._copy_vector(self.v, self._nsendstorage, origin={"inField": (_xps-_xms,_ype-_yms), "outField": (0,1)}, domain=(_npx,1,1))
            self._copy_vector(self.h, self._nsendstorage, origin={"inField": (_xps-_xms,_ype-_yms), "outField": (0,2)}, domain=(_npx,1,1))
        if (_south != -1):
            self._copy_vector(self.u, self._ssendstorage, origin={"inField": (_xps-_xms,_yps-_yms), "outField": (0,0)}, domain=(_npx,1,1))
            self._copy_vector(self.v, self._ssendstorage, origin={"inField": (_xps-_xms,_yps-_yms), "outField": (0,1)}, domain=(_npx,1,1))
            self._copy_vector(self.h, self._ssendstorage, origin={"inField": (_xps-_xms,_yps-_yms), "outField": (0,2)}, domain=(_npx,1,1))
        if (_west != -1):
            self._copy_vector(self.u, self._wsendstorage, origin={"inField": (_xps-_xms,_yps-_yms), "outField": (0,0)}, domain=(1,_npy,1))
            self._copy_vector(self.v, self._wsendstorage, origin={"inField": (_xps-_xms,_yps-_yms), "outField": (1,0)}, domain=(1,_npy,1))
            self._copy_vector(self.h, self._wsendstorage, origin={"inField": (_xps-_xms,_yps-_yms), "outField": (2,0)}, domain=(1,_npy,1))
        if (_east != -1):
            self._copy_vector(self.u, self._esendstorage, origin={"inField": (_xpe-_xms,_yps-_yms), "outField": (0,0)}, domain=(1,_npy,1))
            self._copy_vector(self.v, self._esendstorage, origin={"inField": (_xpe-_xms,_yps-_yms), "outField": (1,0)}, domain=(1,_npy,1))
            self._copy_vector(self.h, self._esendstorage, origin={"inField": (_xpe-_xms,_yps-_yms), "outField": (2,0)}, domain=(1,_npy,1))

        # Should not need this synchronize
        self.device_synchronize()

        # Set up send buffers with proper layout
        if (_north != -1):
            self._nsendbuff = self._array(self._nsendstorage.data, dtype=self.float_type, ndmin=2)
        if (_south != -1):
            self._ssendbuff = self._array(self._ssendstorage.data, dtype=self.float_type, ndmin=2)
        if (_west != -1):
            self._wsendbuff = self._array(self._wsendstorage.data, dtype=self.float_type, ndmin=2)
        if (_east != -1):
            self._esendbuff = self._array(self._esendstorage.data, dtype=self.float_type, ndmin=2)

        # Syncrhonize before MPI communication
        self.device_synchronize()

        # Now post the non-blocking send half of the exchange
        if (_north != -1):
            _irequests.append(_communicator.Isend(self._nsendbuff, _north, _ntag))
            _nrequests = _nrequests + 1
        if (_south != -1):
            _irequests.append(_communicator.Isend(self._ssendbuff, _south, _stag))
            _nrequests = _nrequests + 1
        if (_west != -1):
            _irequests.append(_communicator.Isend(self._wsendbuff, _west, _wtag))
            _nrequests = _nrequests + 1
        if (_east != -1):
            _irequests.append(_communicator.Isend(self._esendbuff, _east, _etag))
            _nrequests = _nrequests + 1

        # Synchronize before MPI call
        self.device_synchronize()

        # Wait for the exchange to complete
        if (_nrequests > 0):
            MPI.Request.Waitall(_irequests)

        # Should not need this synchronization
        self.device_synchronize()

        # Create storages from the receive buffers
        if (_north != -1):
            print(f"nrecvbuff after Irecv is complete = {self._nrecvbuff[70:80, 2]}")
            self._nrecvstorage = gt_storage.from_array(self._nrecvbuff, dtype=self.float_type, backend=self.backend, default_origin=(0, 0))
            print(f"gt storage created from nrecvbuff using from_array() = {self._nrecvstorage[70:80, 2]}")
        if (_south != -1):
            self._srecvstorage = gt_storage.from_array(self._srecvbuff, dtype=self.float_type, backend=self.backend, default_origin=(0, 0))
        if (_west != -1):
            self._wrecvstorage = gt_storage.from_array(self._wrecvbuff, dtype=self.float_type, backend=self.backend, default_origin=(0, 0))
        if (_east != -1):
            self._erecvstorage = gt_storage.from_array(self._erecvbuff, dtype=self.float_type, backend=self.backend, default_origin=(0, 0))

        # Unpack the storages into the state variables
        if (_north != -1):
            self._copy_vector(self._nrecvstorage, self.u, origin={"inField": (0,0), "outField": (_xps-_xms,_yme-_yms)}, domain=(_npx,1,1))
            self._copy_vector(self._nrecvstorage, self.v, origin={"inField": (0,1), "outField": (_xps-_xms,_yme-_yms)}, domain=(_npx,1,1))
            print(f"state variable h before unpack stencil call = {self.h[70:80, _yme-_yms]}")
            self._copy_vector(self._nrecvstorage, self.h, origin={"inField": (0,2), "outField": (_xps-_xms,_yme-_yms)}, domain=(_npx,1,1))
            print(f"state variable h after unpack stencil call = {self.h[70:80, _yme-_yms]}")
        if (_south != -1):
            self._copy_vector(self._srecvstorage, self.u, origin={"inField": (0,0), "outField": (_xps-_xms,0)}, domain=(_npx,1,1))
            self._copy_vector(self._srecvstorage, self.v, origin={"inField": (0,1), "outField": (_xps-_xms,0)}, domain=(_npx,1,1))
            self._copy_vector(self._srecvstorage, self.h, origin={"inField": (0,2), "outField": (_xps-_xms,0)}, domain=(_npx,1,1))
        if (_west != -1):
            self._copy_vector(self._wrecvstorage, self.u, origin={"inField": (0,0), "outField": (0,_yps-_yms)}, domain=(1,_npy,1))
            self._copy_vector(self._wrecvstorage, self.v, origin={"inField": (1,0), "outField": (0,_yps-_yms)}, domain=(1,_npy,1))
            self._copy_vector(self._wrecvstorage, self.h, origin={"inField": (2,0), "outField": (0,_yps-_yms)}, domain=(1,_npy,1))
        if (_east != -1):
            self._copy_vector(self._erecvstorage, self.u, origin={"inField": (0,0), "outField": (_xme-_xms,_yps-_yms)}, domain=(1,_npy,1))
            self._copy_vector(self._erecvstorage, self.v, origin={"inField": (1,0), "outField": (_xme-_xms,_yps-_yms)}, domain=(1,_npy,1))
            self._copy_vector(self._erecvstorage, self.h, origin={"inField": (2,0), "outField": (_xme-_xms,_yps-_yms)}, domain=(1,_npy,1))

        # Should not need this synchronization
        self.device_synchronize()


    # Scatter full state 
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


    # Gather local state
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


    # Get state u 
    def get_u(self):
        return self.u[self.geometry.xps - self.geometry.xms:self.geometry.xpe - self.geometry.xms+1, self.geometry.yps - self.geometry.yms:self.geometry.ype - self.geometry.yms+1]
        
    # Get state v 
    def get_v(self):
        return self.v[self.geometry.xps - self.geometry.xms:self.geometry.xpe - self.geometry.xms+1, self.geometry.yps - self.geometry.yms:self.geometry.ype - self.geometry.yms+1]

    # Get state h
    def get_h(self):
        return self.h[self.geometry.xps - self.geometry.xms:self.geometry.xpe - self.geometry.xms+1, self.geometry.yps - self.geometry.yms:self.geometry.ype - self.geometry.yms+1]


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
            _uVar = _dataset.createVariable("U", "f8", (_nyDim, _nxDim))
            _uVar.long_name = "Zonal Velocity"
            _uVar.units = "m / s"
            _vVar = _dataset.createVariable("V", "f8", (_nyDim, _nxDim))
            _vVar.long_name = "Meridional Velocity"
            _vVar.units = "m / s"
            _hVar = _dataset.createVariable("H", "f8", (_nyDim, _nxDim))
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
            _uVar[:,:] = np.transpose(_u_full[:,:])
            _vVar[:,:] = np.transpose(_v_full[:,:])
            _hVar[:,:] = np.transpose(_h_full[:,:])

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
            _u_full[:,:] = np.transpose(_dataset.variables["U"][:,:])
            _v_full = np.empty((_nx, _ny))
            _v_full[:,:] = np.transpose(_dataset.variables["V"][:,:])
            _h_full = np.empty((_nx, _ny))
            _h_full[:,:] = np.transpose(_dataset.variables["H"][:,:])

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
