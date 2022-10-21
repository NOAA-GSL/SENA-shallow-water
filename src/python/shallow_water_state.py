import time
from mpi4py import MPI
import numpy as np
from netCDF4 import Dataset
import gt4py.storage as gt_storage

from shallow_water_geometry import ShallowWaterGeometry


class ShallowWaterState:

    def __init__(self, geometry: ShallowWaterGeometry, clock: np.float64, backend="numpy", u=None, v=None, h=None):
        """
        Initialized a shallow water state class 

            Arguments: 
                geometry   
                clock   

            Return: 
                An initialized ShallowWaterState class 
        """
        # Set the physical constant of gravity
        _g = np.float64(9.81)

        # Set the geometry associated with this state
        self.geometry = geometry

        # Set the GT4Py backend to the state
        self.backend = backend

        # Get the domain index range for this patch from the geometry
        _xps = geometry.xps
        _xpe = geometry.xpe
        _yps = geometry.yps
        _ype = geometry.ype

        # Get the memory allocation index range for this patch from the geometry
        _xms = geometry.xms
        _xme = geometry.xme
        _yms = geometry.yms
        _yme = geometry.yme

        # Allocate u, v, h 
        self.u = gt_storage.zeros(backend, default_origin=(1,1), shape=(_xme - _xms + 1, _yme - _yms + 1), dtype=np.float64)
        self.v = gt_storage.zeros(backend, default_origin=(1,1), shape=(_xme - _xms + 1, _yme - _yms + 1), dtype=np.float64)
        self.h = gt_storage.zeros(backend, default_origin=(1,1), shape=(_xme - _xms + 1, _yme - _yms + 1), dtype=np.float64)

        # Initialize u 
        if (u) is not None:
            for i in range(_xps, _xpe + 1):
                for j in range(_yps, _ype + 1):
                    self.u[i - _xms, j - _yms] = u[i - _xps, j - _yps]

        # Initialize v 
        if (v) is not None:
            for i in range(_xps, _xpe + 1):
                for j in range(_yps, _ype + 1):
                    self.v[i - _xms, j - _yms] = v[i - _xps, j - _yps]

        # Initialize h
        if (h) is not None:
            for i in range(_xps, _xpe + 1):
                for j in range(_yps, _ype + 1):
                    self.h[i - _xms, j - _yms] = h[i - _xps, j - _yps]

        # Calculate the maximum wave speed from h
        _local_max = np.full(1, np.amax(self.h))
        _max_h = np.zeros(1, np.float64)

        # Allreduce(sendbuf, recvbuf, op=SUM)
        geometry.mpi_comm.Allreduce(_local_max, _max_h, op=MPI.MAX)
        self.max_wavespeed = np.sqrt(_g * _max_h)

        # Initialize clock
        self.clock = clock if clock is not None else np.float64(0.0)


    # Send boundaires to neighboring halos for each process
    def exchange_halo(self):

        # Get the MPI communicator from the geometry
        communicator = self.geometry.mpi_comm

        # Get the index ranges for this patch
        xms = self.geometry.xms
        xme = self.geometry.xme
        yms = self.geometry.yms
        yme = self.geometry.yme
        xps = self.geometry.xps
        xpe = self.geometry.xpe
        yps = self.geometry.yps
        ype = self.geometry.ype

        # Get the extents of the domain
        npx = self.geometry.npx
        npy = self.geometry.npy

        # Get MPI ranks of the neighbors of this patch
        north = self.geometry.north
        south = self.geometry.south
        west =  self.geometry.west
        east =  self.geometry.east

        # Set the MPI tags 
        ntag, stag, wtag, etag = 1, 2, 3, 4

        # Set irequests to empty array (will append MPI requests)
        irequests = []

        # Initialize the send and receive buffer arrays
        nsendbuffer = np.zeros((npx, 3), np.float64)
        ssendbuffer = np.zeros((npx, 3), np.float64)
        wsendbuffer = np.zeros((npy, 3), np.float64)
        esendbuffer = np.zeros((npy, 3), np.float64)
        nrecvbuffer = np.zeros((npx, 3), np.float64)
        srecvbuffer = np.zeros((npx, 3), np.float64)
        wrecvbuffer = np.zeros((npy, 3), np.float64)
        erecvbuffer = np.zeros((npy, 3), np.float64)

        # Post the non-blocking receive half of the exhange first to reduce overhead
        # Irecv(buf, source=ANY_SOURCE, tag=ANY_TAG)
        nrequests = 0 
        if (north != -1):
            irequests.append(communicator.Irecv(nrecvbuffer, source=north, tag=stag))
            nrequests = nrequests + 1
        if (south != -1):
            irequests.append(communicator.Irecv(srecvbuffer, source=south, tag=ntag))
            nrequests = nrequests + 1
        if (west != -1):
            irequests.append(communicator.Irecv(wrecvbuffer, source=west, tag=etag))
            nrequests = nrequests + 1
        if (east != -1):
            irequests.append(communicator.Irecv(erecvbuffer, source=east, tag=wtag))
            nrequests = nrequests + 1

        # Pack the send buffers
        if (north != -1):
            for i in range(xps, xpe + 1):
                nsendbuffer[i - xps, 0] = self.u[i - xms, ype - yms]
                nsendbuffer[i - xps, 1] = self.v[i - xms, ype - yms]
                nsendbuffer[i - xps, 2] = self.h[i - xms, ype - yms]

        if (south != -1):
            for i in range(xps, xpe + 1):
                ssendbuffer[i - xps, 0] = self.u[i - xms, yps - yms]
                ssendbuffer[i - xps, 1] = self.v[i - xms, yps - yms]
                ssendbuffer[i - xps, 2] = self.h[i - xms, yps - yms]

        if (west != -1):
            for j in range(yps, ype + 1):
                wsendbuffer[j - yps, 0] = self.u[xps - xms, j - yms]
                wsendbuffer[j - yps, 1] = self.v[xps - xms, j - yms]
                wsendbuffer[j - yps, 2] = self.h[xps - xms, j - yms]

        if (east != -1):
            for j in range(yps, ype + 1):
                esendbuffer[j - yps, 0] = self.u[xpe - xms, j - yms]
                esendbuffer[j - yps, 1] = self.v[xpe - xms, j - yms]
                esendbuffer[j - yps, 2] = self.h[xpe - xms, j - yms]

        # Now post the non-blocking send half of the exchange
        # Isend(buf, dest, tag=0)
        if (north != -1):
            irequests.append(communicator.Isend(nsendbuffer, north, tag=ntag))
            nrequests = nrequests + 1

        if (south != -1):
            irequests.append(communicator.Isend(ssendbuffer, south, tag=stag))
            nrequests = nrequests + 1

        if (west != -1):
            irequests.append(communicator.Isend(wsendbuffer, west, tag=wtag))
            nrequests = nrequests + 1
            
        if (east != -1):
            irequests.append(communicator.Isend(esendbuffer, east, tag=etag))
            nrequests = nrequests + 1

        #  Wait for the exchange to complete
        # MPI - Waitall(requests, statuses=None)
        if (nrequests > 0):
            MPI.Request.Waitall(irequests)
        
        # Unpack the receive buffers
        if (north != -1):
            for i in range(xps, xpe + 1):
                self.u[i - xms, yme - yms] = nrecvbuffer[i - xps, 0]
                self.v[i - xms, yme - yms] = nrecvbuffer[i - xps, 1]
                self.h[i - xms, yme - yms] = nrecvbuffer[i - xps, 2]

        if (south != -1):
            for i in range(xps, xpe + 1):
                self.u[i - xms, yms - yms] = srecvbuffer[i - xps, 0]
                self.v[i - xms, yms - yms] = srecvbuffer[i - xps, 1]
                self.h[i - xms, yms - yms] = srecvbuffer[i - xps, 2]

        if (west != -1):
            for j in range(yps, ype + 1):
                self.u[xms - xms, j - yms] = wrecvbuffer[j - yps, 0]
                self.v[xms - xms, j - yms] = wrecvbuffer[j - yps, 1]
                self.h[xms - xms, j - yms] = wrecvbuffer[j - yps, 2]

        if (east != -1):
            for j in range(yps, ype + 1):
                self.u[xme - xms, j - yms] = erecvbuffer[j - yps, 0]
                self.v[xme - xms, j - yms] = erecvbuffer[j - yps, 1]
                self.h[xme - xms, j - yms] = erecvbuffer[j - yps, 2]


    # Scatter full state 
    def scatter(self, u_full, v_full, h_full):

        # Get the MPI communicator from the geometry
        communicator = self.geometry.mpi_comm

        # Get the number of MPI ranks from the geometry
        _nranks = self.geometry.nranks

        # Get the MPI rank of this process from the geometry
        _myrank = self.geometry.rank

        # Get the local indices (excluding the halo) from the geometry
        _xms_local = np.full(1, self.geometry.xms, np.int32)
        _xme_local = np.full(1, self.geometry.xme, np.int32)
        _yms_local = np.full(1, self.geometry.yms, np.int32)
        _yme_local = np.full(1, self.geometry.yme, np.int32)

        # Calculate the local number of elements
        # _nelements_local = (_xme_local - _xms_local + 1) * (_yme_local - _yms_local + 1)

        # Allocate space for the indices of each rank
        if (_myrank == 0):
            _xms = np.empty(_nranks, np.int32)
            _xme = np.empty(_nranks, np.int32)
            _yms = np.empty(_nranks, np.int32)
            _yme = np.empty(_nranks, np.int32)
        else:
          # Allocate on other ranks to avoid triggering debug traps during calls to MPI_Gather
            _xms = np.empty(1, np.int32)
            _xme = np.empty(1, np.int32)
            _yms = np.empty(1, np.int32)
            _yme = np.empty(1, np.int32)
    
        # Gather the local indices for each rank
        # Gather(sendbuf, recvbuf, root=0)
        communicator.Gather(_xms_local, _xms)
        communicator.Gather(_xme_local, _xme)
        communicator.Gather(_yms_local, _yms)
        communicator.Gather(_yme_local, _yme)

        # Calculate the number of elements to send to each rank
        if (_myrank == 0):
          _nsend_elements = np.empty(_nranks, np.int32)
          for n in range(_nranks):
            _nsend_elements[n] = (_xme[n] - _xms[n] + 1) * (_yme[n] - _yms[n] + 1)
        else:
            # Allocate on other ranks to avoid triggering debug traps during calls to MPI_Scatterv
            _nsend_elements = np.empty(1, np.int32)  
          
        # Calculate the send buffer offsets for each rank
        if (_myrank == 0):
            _send_offsets = np.zeros(_nranks, np.int32)
            for n in range(1, _nranks):
                _send_offsets[n] = _send_offsets[n-1] + _nsend_elements[n-1]
        else:
            # Allocate on other ranks to avoid debug traps during calls to MPI_Scatterv
            _send_offsets = np.empty(1, np.int32)

        # Allocate a send buffer for scattering u, v, and h
        if (_myrank == 0):
            _send_buffer = np.empty(_nsend_elements.sum(), np.float64)
        else:
            # Allocate on other ranks to avoid triggering debug traps during calls to MPI_Scatterv
            _send_buffer = np.empty(1, np.float64)  

        # Fill the send buffer and scatter u, v, h
        # Scatterv(sendbuf, recvbuf, root=0)
        if (_myrank == 0):
            for n in range(_nranks): 
                _send_buffer[_send_offsets[n]: _send_offsets[n] + _nsend_elements[n]] = (u_full[_xms[n]-1:_xme[n], _yms[n]-1:_yme[n]]).flatten()
        communicator.Scatterv([_send_buffer, _nsend_elements, _send_offsets, MPI.DOUBLE], self.u)
        if (_myrank == 0):
            for n in range(_nranks): 
                _send_buffer[_send_offsets[n]: _send_offsets[n] + _nsend_elements[n]] = (v_full[_xms[n]-1:_xme[n], _yms[n]-1:_yme[n]]).flatten()
        communicator.Scatterv([_send_buffer, _nsend_elements, _send_offsets, MPI.DOUBLE], self.v)
        if (_myrank == 0):
            for n in range(_nranks): 
                _send_buffer[_send_offsets[n]: _send_offsets[n] + _nsend_elements[n]] = (h_full[_xms[n]-1:_xme[n], _yms[n]-1:_yme[n]]).flatten()
        communicator.Scatterv([_send_buffer, _nsend_elements, _send_offsets, MPI.DOUBLE], self.h)


    # Gather local state
    def gather(self):

        # Get the MPI communicator from the geometry
        communicator = self.geometry.mpi_comm

        # Get the number of MPI ranks from the geometry
        _nranks = self.geometry.nranks

        # Get the MPI rank of this process from the geometry
        _myrank = self.geometry.rank

        # Calculate the local number of elements
        # _nelements_local = (_xpe_local - _xps_local + 1) * (_ype_local - _yps_local + 1)

        # Get the local indices (excluding the halo) from the geometry
        _xps_local = self.geometry.xps
        _xpe_local = self.geometry.xpe
        _yps_local = self.geometry.yps
        _ype_local = self.geometry.ype

        # Get the local indices (including the halo) from the geometry
        _xms_local = self.geometry.xms
        _yms_local = self.geometry.yms

        # Allocate full domain arrays for the gather
        if (_myrank == 0):
            _u_full = np.zeros((self.geometry.nx, self.geometry.ny))
            _v_full = np.zeros((self.geometry.nx, self.geometry.ny))
            _h_full = np.zeros((self.geometry.nx, self.geometry.ny))
        else:
            _u_full = None
            _v_full = None
            _h_full = None

        # Allocate space for full domain arrays for the gather
        if (_myrank == 0):
            _xps = np.zeros(_nranks, np.int32)
            _xpe = np.zeros(_nranks, np.int32)
            _yps = np.zeros(_nranks, np.int32)
            _ype = np.zeros(_nranks, np.int32)
        else:
            _xps = np.empty(1, np.int32)
            _xpe = np.empty(1, np.int32)
            _yps = np.empty(1, np.int32)
            _ype = np.empty(1, np.int32)
    
        # Gather the local indices for each rank
        # Gather(sendbuf, recvbuf, root=0)
        communicator.Gather(np.asarray(_xps_local), _xps)
        communicator.Gather(np.asarray(_xpe_local), _xpe)
        communicator.Gather(np.asarray(_yps_local), _yps)
        communicator.Gather(np.asarray(_ype_local), _ype)

        #  Calculate the number of elements that will be receieved from each rank
        if (_myrank == 0):
          _nrecv_elements = np.empty(_nranks, np.int32)
          for n in range(_nranks):
            _nrecv_elements[n] = (_xpe[n] - _xps[n] + 1) * (_ype[n] - _yps[n] + 1)
        else:
            _nrecv_elements = np.empty(1, np.int32)  
          
        # Calculate the receive buffer offsets for each rank
        if (_myrank == 0):
            _recv_offsets = np.zeros(_nranks, np.int32)
            for n in range(1, _nranks):
                _recv_offsets[n] = _recv_offsets[n-1] + _nrecv_elements[n-1]
        else:
            _recv_offsets = np.empty(1, np.int32)

        # Allocate a receive buffer for gathering u, v, and h
        if (_myrank == 0):
            _recv_buffer = np.empty(_nrecv_elements.sum(), np.float64)
        else:
            _recv_buffer = np.empty(1, np.float64)  

        # Gather u, v, and h from all ranks and unpack into full size arrays
        # Gatherv(sendbuf, recvbuf, root=0)
        communicator.Gatherv(np.ascontiguousarray(self.u[_xps_local-_xms_local:_xpe_local-_xms_local+1, _yps_local-_yms_local:_ype_local-_yms_local+1]), [_recv_buffer, _nrecv_elements, _recv_offsets, MPI.DOUBLE])
        
        if (_myrank ==0):
          for n in range(_nranks):
            _u_full[_xps[n]-1: _xpe[n], _yps[n]-1: _ype[n]] = np.reshape(_recv_buffer[_recv_offsets[n]: _recv_offsets[n] + _nrecv_elements[n]],
                                                                (_xpe[n] - _xps[n] + 1, _ype[n] - _yps[n] + 1))
        communicator.Gatherv(np.ascontiguousarray(self.v[_xps_local-_xms_local:_xpe_local-_xms_local+1, _yps_local-_yms_local:_ype_local-_yms_local+1]), [_recv_buffer, _nrecv_elements, _recv_offsets, MPI.DOUBLE])
       
        if (_myrank ==0):
          for n in range(_nranks):
            _v_full[_xps[n]-1: _xpe[n], _yps[n]-1: _ype[n]] = np.reshape(_recv_buffer[_recv_offsets[n]: _recv_offsets[n] + _nrecv_elements[n]],
                                                                (_xpe[n] - _xps[n] + 1, _ype[n] - _yps[n] + 1))
        
        communicator.Gatherv(np.ascontiguousarray(self.h[_xps_local-_xms_local:_xpe_local-_xms_local+1, _yps_local-_yms_local:_ype_local-_yms_local+1]), [_recv_buffer, _nrecv_elements, _recv_offsets, MPI.DOUBLE])
        
        if (_myrank ==0):
          for n in range(_nranks):
            _h_full[_xps[n]-1: _xpe[n], _yps[n]-1: _ype[n]] = np.reshape(_recv_buffer[_recv_offsets[n]: _recv_offsets[n] + _nrecv_elements[n]],
                                                                 (_xpe[n] - _xps[n] + 1, _ype[n] - _yps[n] + 1))

        # Return the full domain arrays (valid on rank 0 only)
        return _u_full, _v_full, _h_full

   
    # Get state u 
    def get_u(self):
        return self.u[self.geometry.xps : self.geometry.xpe, self.geometry.yps : self.geometry.ype]

    # Get state v 
    def get_v(self):
        return self.v[self.geometry.xps : self.geometry.xpe, self.geometry.yps : self.geometry.ype]

    # Get state h
    def get_h(self):
        return self.h[self.geometry.xps : self.geometry.xpe, self.geometry.yps : self.geometry.ype]


    # Advance clock by dt 
    def advance_clock(self, dt):
        self.clock = self.clock + dt

    # Read state from NetCDF file 
    def read_NetCDF(self, filename: str):
        
        # Get the MPI communicator from the geometry
        communicator = self.geometry.mpi_comm
        
        # Get the MPI rank of this process from the geometry
        myrank = self.geometry.rank

        # Read full state from file on rank 0
        if (myrank == 0):
            # Load Data into a Dataset, in-memory, and save the dataset to disk when closing
            _shallow_water_data = Dataset(filename, 'r', diskless=True, persist=True)
            
            print("shallow_water netcdf data :", _shallow_water_data)

            # Read/Create the model dimensions
            _nx = _shallow_water_data.dimensions['nx'].size
            _ny = _shallow_water_data.dimensions['ny'].size
                        
            # Read/Create global attributes
            _xmax  = _shallow_water_data.getncattr("xmax")
            _ymax  = _shallow_water_data.getncattr("ymax")
            _clock = np.full(1,_shallow_water_data.getncattr("clock"))
            
            # Check to make sure state read in matches this state's geometry
            if (self.geometry.nx != _nx or self.geometry.ny != _ny or self.geometry.xmax != _xmax or self.geometry.ymax != _ymax):
              communicator.Abort(-1)

            # Get the u variable
            _u_full = np.empty((_nx,_ny))
            _u_full[:,:] = _shallow_water_data.variables["U"][:,:]
    
            # Get the v variable
            _v_full = np.empty((_nx,_ny))
            _v_full[:,:] = _shallow_water_data.variables["V"][:,:]
    
            # Get the h variable
            _h_full = np.empty((_nx,_ny))
            _h_full[:,:] = _shallow_water_data.variables["H"][:,:]
    
        else:
            _u_full = np.empty(1)
            _v_full = np.empty(1)
            _h_full = np.empty(1)
            _clock = np.empty(1)

        # Flush buffers
        _shallow_water_data.sync()
        # Close the NetCDF file
        _shallow_water_data.close()

        # Scatter u, v, and h
        self.scatter(u_full=_u_full, v_full=_v_full, h_full=_h_full)

        # Broadcast the clock
        if (myrank == 0):
            self.clock = _clock[0]
        
        communicator.Bcast(_clock)

        return _u_full, _v_full, _h_full


    # Write state to NetCDF file
    def write_NetCDF(self, filename: str):
        
        # Get the MPI rank of this process from the geometry
        _myrank = self.geometry.rank

        _nx = self.geometry.nx
        _ny = self.geometry.ny
        _xmax = self.geometry.xmax
        _ymax = self.geometry.ymax
        _clock = self.clock

        _dx = self.geometry.dx
        _dy = self.geometry.dy

        # Gather full u, v, and h
        _u_full, _v_full, _h_full = self.gather()

        if (_myrank == 0):
            
            # Open new file, overwriting previous contents
            _shallow_water_data = Dataset(filename, "w")
            
            # Write Global Attributes
            timestr = time.strftime("%Y/%m/%d %H:%M:%S")
            _shallow_water_data.creation_date = timestr
            _shallow_water_data.model_name = "Shallow Water"
            _shallow_water_data.xmax = _xmax
            _shallow_water_data.ymax = _ymax
            _shallow_water_data.clock = _clock
        
            #Define the x/y dimensions
            _shallow_water_data.createDimension("nx", _nx)
            _shallow_water_data.createDimension("ny", _ny)

            # Create variables 
            x = _shallow_water_data.createVariable("x","f8", ("nx",))
            y = _shallow_water_data.createVariable("y","f8", ("ny",))
            U = _shallow_water_data.createVariable("U", "f8", ("ny", "nx"))
            V = _shallow_water_data.createVariable("V", "f8", ("ny", "nx"))
            H = _shallow_water_data.createVariable("H", "f8", ("ny", "nx"))

            # Define the x field attributes
            x.long_name = "x" 
            x.units = "Nondimensional" 

            # Define the y field attributes
            y.long_name = "x" 
            y.units = "Nondimensional"

            # Define the U variable attributes
            U.long_name = "Zonal Velocity"
            U.units = "m / s"

            # Define the V variable attributes
            V.long_name = "Meridional Velocity"
            V.untis = "m / s"

            # Define the H variable attributes
            H.long_name = "Pressure Surface Height"
            H.units = "m"

            # Fill the x variable
            _x = np.zeros((_nx), np.float64)
            for i in range(_nx):
                _x[i] = i * _dx
            
            # Fill the y variable
            _y = np.zeros((_ny), np.float64)
            for i in range(_ny):
                _y[i] = i * _dy
            
            # Fill the velocity variables, & pressure surface height variable
            U[:,:] = _u_full[:,:]
            V[:,:] = _v_full[:,:] 
            H[:,:] = _h_full[:,:]

            _shallow_water_data.close()
        
