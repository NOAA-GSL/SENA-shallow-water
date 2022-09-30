from time import time
from mpi4py import MPI
import numpy as np
from netCDF4 import Dataset

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
                An initialized shallow_water_state class 
        """
        # Set the physical constant of gravity
        _g = np.float64(9.81)
        
        # Set the geometry associated with this state
        self.geometry = geometry

        # # Valid time of the state
        self.clock = clock if clock is not None else np.float64(0.0)
        
        # Get the domain index range for this patch from the geometry
        _xps = self.geometry.get_xps()
        _xpe = self.geometry.get_xpe()
        _yps = self.geometry.get_yps()
        _ype = self.geometry.get_ype()

        # Get the memory allocation index range for this patch from the geometry
        _xms = self.geometry.get_xms()
        _xme = self.geometry.get_xme()
        _yms = self.geometry.get_yms()
        _yme = self.geometry.get_yme()

        # Allocate u, v, h 
        self.u = np.zeros((_xme - _xms + 1, _yme - _yms + 1), np.float64, order='F')
        self.v = np.zeros((_xme - _xms + 1, _yme - _yms + 1), np.float64, order='F')
        self.h = np.zeros((_xme - _xms + 1, _yme - _yms + 1), np.float64, order='F')
        # self.u = np.ones((_xme - _xms + 1, _yme - _yms + 1), np.float64, order='F')
        # self.v = np.ones((_xme - _xms + 1, _yme - _yms + 1), np.float64, order='F')
        # self.h = np.ones((_xme - _xms + 1, _yme - _yms + 1), np.float64, order='F')

        # Initialize u 
        if (u):
            for j in range(_yps, _ype + 1):
                for i in range(_xps, _xpe + 1):
                    self.u[i - _xms, j - _yms] = u[i - _xps, j - _yps]

        # Initialize v 
        if (v):
            for j in range(_yps, _ype + 1):
                for i in range(_xps, _xpe + 1):
                    self.v[i - _xms, j - _yms] = v[i - _xps, j - _yps]

        # Initialize h
        if (h):
            for j in range(_yps, _ype + 1):
                for i in range(_xps, _xpe + 1):
                    self.h[i - _xms, j - _yms] = h[i - _xps, j - _yps]

        # Calculate the maximum wave speed from h 
        _local_max = np.full(1, np.amax(self.h[_xps:_xpe + 1, _yps:_ype + 1]))
        _max_h = np.zeros(1, np.float64)
        # Allreduce(sendbuf, recvbuf, op=SUM)
        self.geometry.mpi_comm.Allreduce(_local_max, _max_h, op=MPI.MAX)
           
        self.max_wavespeed = np.sqrt(_g * _max_h)
    

    # Send boundaires to neighboring halos for each process
    def exchange_halo(self):

        # Get the MPI communicator from the geometry
        communicator = self.geometry.get_communicator()

        # Set the MPI tags 
        ntag, stag, wtag, etag = 1, 2, 3, 4

        # Get the index ranges for this patch
        xms = self.geometry.get_xms()
        xme = self.geometry.get_xme()
        yms = self.geometry.get_yms()
        yme = self.geometry.get_yme()
        xps = self.geometry.get_xps()
        xpe = self.geometry.get_xpe()
        yps = self.geometry.get_yps()
        ype = self.geometry.get_ype()

        # Initialize the send and receive buffer arrays
        nsendbuffer = np.zeros((xpe - xps + 1, 3), np.float64, order='F')
        ssendbuffer = np.zeros((xpe - xps + 1, 3), np.float64, order='F')
        wsendbuffer = np.zeros((ype - yps + 1, 3), np.float64, order='F')
        esendbuffer = np.zeros((ype - yps + 1, 3), np.float64, order='F')
        nrecvbuffer = np.zeros((xpe - xps + 1, 3), np.float64, order='F')
        srecvbuffer = np.zeros((xpe - xps + 1, 3), np.float64, order='F')
        wrecvbuffer = np.zeros((ype - yps + 1, 3), np.float64, order='F')
        erecvbuffer = np.zeros((ype - yps + 1, 3), np.float64, order='F')

        # Get the extents of the domain
        npx = self.geometry.get_npx()
        npy = self.geometry.get_npy()

        # Get MPI ranks of the neighbors of this patch
        north = self.geometry.get_north()
        south = self.geometry.get_south()
        west = self.geometry.get_west()
        east = self.geometry.get_east()

        # Post the non-blocking receive half of the exhange first to reduce overhead
        nrequests = 0 
        if (north != -1):
            nrequests = nrequests + 1 
            communicator.Irecv(nrecvbuffer)
        if (south != -1):
           nrequests = nrequests + 1
           communicator.Irecv(srecvbuffer) 
        if (west != -1):
           nrequests = nrequests + 1
           communicator.Irecv(wrecvbuffer) 
        if (east != -1):
           nrequests = nrequests + 1
           communicator.Irecv(erecvbuffer) 

        # Pack the send buffers
        if (north != -1):
          for i in range(xpe - xps + 1):
            nsendbuffer[i, 0] = self.u[i, ype]
            nsendbuffer[i, 1] = self.v[i, ype]
            nsendbuffer[i, 2] = self.h[i, ype]

        if (south != -1):
          for i in range(xpe - xps + 1):
            ssendbuffer[i, 0] = self.u[i, yps]
            ssendbuffer[i, 1] = self.v[i, yps]
            ssendbuffer[i, 2] = self.h[i, yps]

        if (west != -1):
          for j in range(ype - yps + 1):
            wsendbuffer[j, 0] = self.u[xps, j]
            wsendbuffer[j, 1] = self.v[xps, j]
            wsendbuffer[j, 2] = self.h[xps, j]

        if (east != -1):
          for j in range(ype - yps + 1):
            esendbuffer[j, 0] = self.u[xpe, j]
            esendbuffer[j, 1] = self.v[xpe, j]
            esendbuffer[j, 2] = self.h[xpe, j]

        # Now post the non-blocking send half of the exchange
        # Isend(buf, dest, tag=0)
        if (north != -1):
           nrequests = nrequests + 1
           communicator.Isend(nsendbuffer, north, tag=ntag)
        if (south != -1):
           nrequests = nrequests + 1
           communicator.Isend(ssendbuffer, south, tag=stag)
        if (west != -1):
           nrequests = nrequests + 1
           communicator.Isend(wsendbuffer, west, tag=wtag)
        if (east != -1):
           nrequests = nrequests + 1
           communicator.Isend(esendbuffer, east, tag=etag)

        #  Wait for the exchange to complete
        # if ( nrequests > 0):
        #     communicator.Waitall()

        # Unpack the receive buffers
        if (north != -1):
          for i in range(xpe- xps + 1):
            self.u[i, yme] = nrecvbuffer[i, 1]
            self.v[i, yme] = nrecvbuffer[i, 2]
            self.h[i, yme] = nrecvbuffer[i, 3]

        if (south != -1):
          for i in range(xpe - xps + 1):
            self.u[i, yms] = srecvbuffer[i, 1]
            self.v[i, yms] = srecvbuffer[i, 2]
            self.h[i, yms] = srecvbuffer[i, 3]

        if (west != -1):
          for j in range(ype - yps + 1):
            self.u[xms,j] = wrecvbuffer[j,1]
            self.v[xms,j] = wrecvbuffer[j,2]
            self.h[xms,j] = wrecvbuffer[j,3]

        if (east != -1):
          for j in range(ype - yps + 1):
            self.u[xme,j] = erecvbuffer[j,1]
            self.v[xme,j] = erecvbuffer[j,2]
            self.h[xme,j] = erecvbuffer[j,3]


    # Scatter full state 
    def scatter(self, u_full, v_full, h_full):

        # Get the MPI communicator from the geometry
        communicator = self.geometry.get_communicator()

        # Get the number of MPI ranks from the geometry
        _nranks = self.geometry.get_nranks()

        # Get the MPI rank of this process from the geometry
        _myrank = self.geometry.get_rank()

        # Get the local indices (excluding the halo) from the geometry
        _xms_local = np.zeros(self.geometry.get_xms(), np.int32)
        _xme_local = np.zeros(self.geometry.get_xme(), np.int32)
        _yms_local = np.zeros(self.geometry.get_yms(), np.int32)
        _yme_local = np.zeros(self.geometry.get_yme(), np.int32)

        # Calculate the local number of elements
        _nelements_local = (_xme_local - _xms_local + 1) * (_yme_local - _yms_local + 1)

        # Allocate space for the indices of each rank
        if (_myrank == 0):
            _xms = np.zeros(_nranks, np.int32)
            _xme = np.zeros(_nranks, np.int32)
            _yms = np.zeros(_nranks, np.int32)
            _yme = np.zeros(_nranks, np.int32)
        else:
          # Allocate on other ranks to avoid triggering debug traps during calls to MPI_Gather
            _xms = np.zeros(1, np.int32)
            _xme = np.zeros(1, np.int32)
            _yms = np.zeros(1, np.int32)
            _yme = np.zeros(1, np.int32)
    
        # Gather the local indices for each rank
        # Gather(sendbuf, recvbuf, root=0)
        communicator.Gather(_xms_local, _xms)
        communicator.Gather(_xme_local, _xme)
        communicator.Gather(_yms_local, _yms)
        communicator.Gather(_yme_local, _yme)

        # Calculate the number of elements to send to each rank
        if (_myrank == 0):
          _nsend_elements = np.zeros(_nranks, np.in32)
          for n in range(_nranks):
            _nsend_elements[n] = (_xme[n] - _xms[n] + 1) * (_yme[n] - _yms[n] + 1)
        else:
            # Allocate on other ranks to avoid triggering debug traps during calls to MPI_Scatterv
            _nsend_elements = np.zeros(1, np.int32)  
          
        # Calculate the send buffer offsets for each rank
        if (_myrank == 0):
            _send_offsets = np.zeros(_nranks, np.int32)
            for n in range(1, _nranks):
                _send_offsets[n] = _send_offsets[n-1] + _nsend_elements[n-1]
        else:
            # Allocate on other ranks to avoid debug traps during calls to MPI_Scatterv
            _send_offsets = np.zeros(1, np.int32)

        # Allocate a send buffer for scattering u, v, and h
        if (_myrank == 0):
            _send_buffer = np.zeros(sum(_nsend_elements), np.float64)
        else:
            # Allocate on other ranks to avoid triggering debug traps during calls to MPI_Scatterv
            _send_buffer = np.zeros(1, np.float64)  

        # Fill the send buffer and scatter u, v, h
        # Scatterv(sendbuf, recvbuf, root=0)
        if (_myrank == 0):
            for n in range(_nranks): 
                _send_buffer[_send_offsets[n] + 1 : _send_offsets[n] + _nsend_elements[n]] = np.reshape(u_full[_xms[n]:_xme[n], _yms[n]:_yme[n]], (1 , _nsend_elements))
        communicator.Scatterv(_send_buffer, self.u)
        if (_myrank == 0):
            for n in range(_nranks): 
                _send_buffer[_send_offsets[n] + 1 : _send_offsets[n] + _nsend_elements[n]] = np.reshape(v_full[_xms[n]:_xme[n], _yms[n]:_yme[n]], (1 , _nsend_elements))
        communicator.Scatterv(_send_buffer, self.v)
        if (_myrank == 0):
            for n in range(_nranks): 
                _send_buffer[_send_offsets[n] + 1 : _send_offsets[n] + _nsend_elements[n]] = np.reshape(h_full[_xms[n]:_xme[n], _yms[n]:_yme[n]], (1 , _nsend_elements))
        communicator.Scatterv(_send_buffer, self.h)


    # Gather local state
    def gather(self, u_full, v_full, h_full):

        # Get the MPI communicator from the geometry
        communicator = self.geometry.get_communicator()

        # Get the number of MPI ranks from the geometry
        _nranks = self.geometry.get_nranks()

        # Get the MPI rank of this process from the geometry
        _myrank = self.geometry.get_rank()

        # Get the local indices (excluding the halo) from the geometry
        _xps_local = np.zeros(self.geometry.get_xps(), np.int32)
        _xpe_local = np.zeros(self.geometry.get_xpe(), np.int32)
        _yps_local = np.zeros(self.geometry.get_yps(), np.int32)
        _ype_local = np.zeros(self.geometry.get_ype(), np.int32)

        # Calculate the local number of elements
        _nelements_local = (_xpe_local - _xps_local + 1) * (_ype_local - _yps_local + 1)

        # Allocate space for the indices of each rank
        if (_myrank == 0):
            _xps = np.zeros(_nranks, np.int32)
            _xpe = np.zeros(_nranks, np.int32)
            _yps = np.zeros(_nranks, np.int32)
            _ype = np.zeros(_nranks, np.int32)
        else:
            # Allocate on other ranks to avoid triggering debug traps during calls to MPI_Gather
            _xps = np.zeros(1, np.int32)
            _xpe = np.zeros(1, np.int32)
            _yps = np.zeros(1, np.int32)
            _ype = np.zeros(1, np.int32)
    
        # Gather the local indices for each rank
        # Gather(sendbuf, recvbuf, root=0)
        communicator.Gather(_xps_local, _xps)
        communicator.Gather(_xpe_local, _xpe)
        communicator.Gather(_yps_local, _yps)
        communicator.Gather(_ype_local, _ype)

        #  Calculate the number of elements that will be receieved from each rank
        if (_myrank == 0):
          _nrecv_elements = np.zeros(_nranks, np.in32)
          for n in range(_nranks):
            _nrecv_elements[n] = (_xpe[n] - _xps[n] + 1) * (_ype[n] - _yps[n] + 1)
        else:
            # Allocate on other ranks to avoid triggering debug traps during calls to MPI_Gatherv
            _nrecv_elements = np.zeros(1, np.int32)  
          
        # Calculate the receive buffer offsets for each rank
        if (_myrank == 0):
            _recv_offsets = np.zeros(_nranks, np.int32)
            for n in range(1, _nranks):
                _recv_offsets[n] = _recv_offsets[n-1] + _nrecv_elements[n-1]
        else:
            # Allocate on other ranks to avoid debug traps during calls to MPI_Gatherv
            _recv_offsets = np.zeros(1, np.int32)

        # Allocate a receive buffer for gathering u, v, and h
        if (_myrank == 0):
            _recv_buffer = np.zeros(sum(_nrecv_elements), np.float64)
        else:
            # Allocate on other ranks to avoid debug traps during calls to MPI_Gatherv
            _recv_buffer = np.zeros(1, np.float64)  

        # Gather u, v, and h from all ranks and unpack into full size arrays
        # Gatherv(sendbuf, recvbuf, root=0)
        communicator.Gatherv(self.u[_xps_local:_xpe_local, _yps_local:_ype_local],_recv_buffer)
        if (_myrank ==0):
          for n in range(_nranks):
            u_full[_xps[n] : _xpe[n], _yps[n] : _ype[n]] = np.reshape(_recv_buffer[_recv_offsets[n]+1 : _recv_offsets[n] + _nrecv_elements[n]], (_xpe[n] - _xps[n] + 1, _ype[n] - _yps[n] + 1))
        communicator.Gatherv(self.v[_xps_local:_xpe_local, _yps_local:_ype_local],_recv_buffer)
        if (_myrank ==0):
          for n in range(_nranks):
            v_full[_xps[n] : _xpe[n], _yps[n] : _ype[n]] = np.reshape(_recv_buffer[_recv_offsets[n]+1 : _recv_offsets[n] + _nrecv_elements[n]], (_xpe[n] - _xps[n] + 1, _ype[n] - _yps[n] + 1))
        communicator.Gatherv(self.h[_xps_local:_xpe_local, _yps_local:_ype_local],_recv_buffer)
        if (_myrank ==0):
          for n in range(_nranks):
            h_full[_xps[n] : _xpe[n], _yps[n] : _ype[n]] = np.reshape(_recv_buffer[_recv_offsets[n]+1 : _recv_offsets[n] + _nrecv_elements[n]], (_xpe[n] - _xps[n] + 1, _ype[n] - _yps[n] + 1))


    # Get state geometry
    def get_geometry(self):
        return self.geometry 
    
    # Get state u 
    def get_u(self):
        return self.u[self.geometry.get_xps() : self.geometry.get_xpe(), self.geometry.get_yps() : self.geometry.get_ype()]

    # Get state v 
    def get_v(self):
        return self.v[self.geometry.get_xps() : self.geometry.get_xpe(), self.geometry.get_yps() : self.geometry.get_ype()]

    # Get state h
    def get_h(self):
        return self.h[self.geometry.get_xps() : self.geometry.get_xpe(), self.geometry.get_yps() : self.geometry.get_ype()]


    #### Pointers are not used, but ported anyways
    # Get pointer to state u
    def get_u_ptr(self, u_ptr):
        u_ptr = self.u

    # Get pointer to state v
    def get_v_ptr(self, v_ptr):
        v_ptr = self.v
    
    # Get pointer to state h
    def get_h_ptr(self, h_ptr):
        h_ptr = self.h

    # Get state clock
    def get_clock(self):
        return self.clock

    # Advance clock by dt 
    def advance_clock(self, dt):
        self.clock = self.clock + dt

    # Get state max wavespeed 
    def get_max_wavespeed(self):
        return self.max_wavespeed

    # Read state from NetCDF file 
    def read_NetCDF(self, filename: str):
        
        # Get the MPI rank of this process from the geometry
        _myrank = self.geometry.get_rank()

        # Read full state from file on rank 0
        if (_myrank == 0):
            # Load Data into a Dataset 
            shallow_water_data = Dataset(filename, 'r')
            
            # Read/Create the model dimensions
            nx = shallow_water_data.dimensions['nx']
            ny = shallow_water_data.dimensions['ny']
            nx_size = shallow_water_data.dimensions['nx'].size
            ny_size = shallow_water_data.dimensions['ny'].size

            print("nx_size ", nx_size)
            print("ny_size ", ny_size)

            # Read/Create global attributes
            xmax = shallow_water_data.getncattr("xmax")
            ymax = shallow_water_data.getncattr("ymax")
            clock = shallow_water_data.getncattr("clock")
            
            print("get xmax attribute ", shallow_water_data.getncattr("xmax"))
            print("get ymax attribute ", shallow_water_data.getncattr("ymax"))
            print("get clock attribute ", shallow_water_data.getncattr("clock"))
            
            print(shallow_water_data)
                
            # Check to make sure state read in matches this state's geometry
            # if (this%geometry%get_nx() /= nx .OR. this%geometry%get_ny() /= ny .OR. this%geometry%get_xmax() /= xmax .OR. this%geometry%get_ymax() /= ymax) then
            #   call MPI_Abort(this%geometry%get_communicator(), -1, ierr)
            

            # Close the NetCDF file 
            shallow_water_data.close()
            
    
        return shallow_water_data





    # Write state to NetCDF file
    def write_NetCDF(self, filename: str):
        pass


comm = MPI.COMM_WORLD 

comm.Get_rank()

geom_config = shallow_water_geometry_config(yamlpath="../../parm/shallow_water_test.yml")

geom = shallow_water_geometry(geometry=geom_config, mpi_comm=comm)

# print(geom.__dict__)

test_shallow_water_state = shallow_water_state(geometry=geom, clock=0.0)

print(test_shallow_water_state.geometry.mpi_comm)

# test_shallow_water_state.exchange_halo()

dimension = test_shallow_water_state.read_NetCDF("../../test/test_input/test_shallow_water_reader.nc")

# print(dimension)