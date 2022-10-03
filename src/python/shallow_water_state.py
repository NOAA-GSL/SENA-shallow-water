#!/usr/bin/env python

from shallow_water_geometry_config import ShallowWaterGeometryConfig
from shallow_water_geometry import ShallowWaterGeometry
import numpy as np
from mpi4py import MPI

class ShallowWaterState:

    def __init__(self, geometry, u=None, v=None, h=None, clock=0):

        # Physical constants
        _g  = 9.81

        # Set the geometry associated with this state
        self.geometry = geometry

        # Initialize u
        if (u):
            self.u = u
        else:
            self.u = np.zeros((geometry.npx, geometry.npy))

        # Initialize v
        if (v):
            self.v = v
        else:
            self.v = np.zeros((geometry.npx, geometry.npy))

        # Initialize h
        if (h):
            self.h = h
        else:
            self.h = np.zeros((geometry.npx, geometry.npy))

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

#    ! MPI variables
#    integer            :: communicator
#    integer, parameter :: ntag=1, stag=2, wtag=3, etag=4
#    integer            :: irequest(8), istatus(MPI_STATUS_SIZE, 8), nrequests, ierr
#    real(r8kind)       :: nsendbuffer(self.geometry.get_xps():self.geometry.get_xpe(), 3)
#    real(r8kind)       :: ssendbuffer(self.geometry.get_xps():self.geometry.get_xpe(), 3)
#    real(r8kind)       :: wsendbuffer(self.geometry.get_yps():self.geometry.get_ype(), 3)
#    real(r8kind)       :: esendbuffer(self.geometry.get_yps():self.geometry.get_ype(), 3)
#    real(r8kind)       :: nrecvbuffer(self.geometry.get_xps():self.geometry.get_xpe(), 3)
#    real(r8kind)       :: srecvbuffer(self.geometry.get_xps():self.geometry.get_xpe(), 3)
#    real(r8kind)       :: wrecvbuffer(self.geometry.get_yps():self.geometry.get_ype(), 3)
#    real(r8kind)       :: erecvbuffer(self.geometry.get_yps():self.geometry.get_ype(), 3)
#
#    ! Indexing variables
#    integer :: i, j
#    integer :: xps, xpe, yps, ype
#    integer :: xms, xme, yms, yme
#    integer :: npx, npy
#    integer :: north, south, west, east
#
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
        _nrecvbuffer = np.zeros((_npx, 3))
        if (_north != -1):
            _irequests[_nrequests] = _communicator.Irecv(_nrecvbuffer, _north, _stag)
            _nrequests = _nrequests + 1
#           call MPI_IRecv(nrecvbuffer, 3* npx, MPI_DOUBLE_PRECISION, north, stag, communicator, irequest(nrequests), ierr)

        _srecvbuffer = np.zeros((_npx, 3))
        if (_south != -1):
            _irequests[_nrequests] = _communicator.Irecv(_srecvbuffer, _south, _ntag)
            _nrequests = _nrequests + 1
#           call MPI_IRecv(srecvbuffer, 3* npx, MPI_DOUBLE_PRECISION, south, ntag, communicator, irequest(nrequests), ierr)

        _wrecvbuffer = np.zeros((_npy, 3))
        if (_west != -1):
            _irequests[_nrequests] = _communicator.Irecv(_wrecvbuffer, _west, _etag)
            _nrequests = _nrequests + 1
#           call MPI_IRecv(wrecvbuffer, 3* npy, MPI_DOUBLE_PRECISION, west, etag, communicator, irequest(nrequests), ierr)

        _erecvbuffer = np.zeros((_npy, 3))
        if (_east != -1):
            _irequests[_nrequests] = _communicator.Irecv(_erecvbuffer, _east, _wtag)
            _nrequests = _nrequests + 1
#           call MPI_IRecv(erecvbuffer, 3* npy, MPI_DOUBLE_PRECISION, east, wtag, communicator, irequest(nrequests), ierr)

#
#        ! Pack the send buffers
#        if (north != -1):
#          do i = xps, xpe
#            nsendbuffer(i, 1) = self.u(i, ype)
#            nsendbuffer(i, 2) = self.v(i, ype)
#            nsendbuffer(i, 3) = self.h(i, ype)
#          end do
#        end if
#        if (south != -1):
#          do i = xps, xpe
#            ssendbuffer(i, 1) = self.u(i, yps)
#            ssendbuffer(i, 2) = self.v(i, yps)
#            ssendbuffer(i, 3) = self.h(i, yps)
#          end do
#        end if
#        if (west != -1):
#          do j = yps, ype
#            wsendbuffer(j, 1) = self.u(xps, j)
#            wsendbuffer(j, 2) = self.v(xps, j)
#            wsendbuffer(j, 3) = self.h(xps, j)
#          end do
#        end if
#        if (east != -1):
#          do j = yps, ype
#            esendbuffer(j, 1) = self.u(xpe, j)
#            esendbuffer(j, 2) = self.v(xpe, j)
#            esendbuffer(j, 3) = self.h(xpe, j)
#          end do
#        end if
#
#        ! Now post the non-blocking send half of the exchange
#        if (north != -1):
#           nrequests = nrequests + 1
#           call MPI_ISend(nsendbuffer, 3 * npx, MPI_DOUBLE_PRECISION, north, ntag, communicator, irequest(nrequests), ierr)
#        end if
#        if (south != -1):
#           nrequests = nrequests + 1
#           call MPI_ISend(ssendbuffer, 3 * npx, MPI_DOUBLE_PRECISION, south, stag, communicator, irequest(nrequests), ierr)
#        end if
#        if (west != -1):
#           nrequests = nrequests + 1
#           call MPI_ISend(wsendbuffer, 3 * npy, MPI_DOUBLE_PRECISION, west, wtag, communicator, irequest(nrequests), ierr)
#        end if
#        if (east != -1):
#           nrequests = nrequests + 1
#           call MPI_ISend(esendbuffer, 3 * npy, MPI_DOUBLE_PRECISION, east, etag, communicator, irequest(nrequests), ierr)
#        end if
#
#        ! Wait for the exchange to complete
#        if (nrequests > 0):
#          call MPI_Waitall(nrequests, irequest, istatus, ierr)
#        end if
#
#        ! Unpack the receive buffers
#        if (north != -1):
#          do i = xps, xpe
#            self.u(i, yme) = nrecvbuffer(i, 1)
#            self.v(i, yme) = nrecvbuffer(i, 2)
#            self.h(i, yme) = nrecvbuffer(i, 3)
#          end do
#        end if
#        if (south != -1):
#          do i = xps, xpe
#            self.u(i, yms) = srecvbuffer(i, 1)
#            self.v(i, yms) = srecvbuffer(i, 2)
#            self.h(i, yms) = srecvbuffer(i, 3)
#          end do
#        end if
#        if (west != -1):
#          do j = yps, ype
#            self.u(xms,j) = wrecvbuffer(j,1)
#            self.v(xms,j) = wrecvbuffer(j,2)
#            self.h(xms,j) = wrecvbuffer(j,3)
#          end do
#        end if
#        if (east != -1):
#          do j = yps, ype
#            self.u(xme,j) = erecvbuffer(j,1)
#            self.v(xme,j) = erecvbuffer(j,2)
#            self.h(xme,j) = erecvbuffer(j,3)
#          end do
#        end if

comm = MPI.COMM_WORLD
gc = ShallowWaterGeometryConfig.from_YAML_filename('foo.yml')
g = ShallowWaterGeometry(gc, comm)
s = ShallowWaterState(g)
s.exchange_halo()
