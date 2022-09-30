module Shallow_Water_State

  use Shallow_Water_Kind,     only: r8kind
  use NetCDF_Utils,           only: nc_check
  use Shallow_Water_Geometry, only: shallow_water_geometry_type
  use mpi

  implicit none

  private

  public :: shallow_water_state_type

  type :: shallow_water_state_type
    private

    ! Shallow water geometry
    type(shallow_water_geometry_type) :: geometry

    ! Shallow water state variables u, v, h
    ! These are public to avoid expensive copies that degrade performance
    real(r8kind), public, allocatable :: u(:,:)
    real(r8kind), public, allocatable :: v(:,:)
    real(r8kind), public, allocatable :: h(:,:)

    ! Maximum wave speed for this state
    real(r8kind) :: max_wavespeed

    ! Valid time of the state
    real(r8kind) :: clock

  contains
    final :: destructor
    procedure, public :: exchange_halo
    procedure, public :: scatter
    procedure, public :: gather
    procedure, public :: get_geometry
    procedure, public :: get_u
    procedure, public :: get_v
    procedure, public :: get_h
    procedure, public :: get_u_ptr
    procedure, public :: get_v_ptr
    procedure, public :: get_h_ptr
    procedure, public :: advance_clock
    procedure, public :: get_clock
    procedure, public :: get_max_wavespeed
    procedure, public :: read
    procedure, public :: write
  end type shallow_water_state_type

  interface shallow_water_state_type
    procedure constructor_state
  end interface

contains

  !------------------------------------------------------------------
  ! constructor_state
  !
  ! Returns an initialized shallow_water_state_type object
  !------------------------------------------------------------------
  function constructor_state(geometry, u, v, h, clock) result(this)

    class(shallow_water_geometry_type),  intent(in) :: geometry
    real(r8kind), optional,              intent(in) :: u(:,:)
    real(r8kind), optional,              intent(in) :: v(:,:)
    real(r8kind), optional,              intent(in) :: h(:,:)
    real(r8kind), optional,              intent(in) :: clock

    ! Physical constants
    real(r8kind), parameter :: g  = 9.81_r8kind

    ! Return a shallow water state object
    type(shallow_water_state_type) :: this

    ! Local variables
    integer      :: i,j, ierr
    integer      :: xps, xpe, yps, ype
    integer      :: xms, xme, yms, yme
    real(r8kind) :: max_h

    ! Set the geometry associated with this state
    this%geometry = geometry

    ! Get the domain index range for this patch from the geometry
    xps = this%geometry%get_xps()
    xpe = this%geometry%get_xpe()
    yps = this%geometry%get_yps()
    ype = this%geometry%get_ype()

    ! Get the memory allocation index range for this patch from the geometry
    xms = this%geometry%get_xms()
    xme = this%geometry%get_xme()
    yms = this%geometry%get_yms()
    yme = this%geometry%get_yme()

    ! Allocate u, v, h
    allocate(this%u(xms:xme, yms:yme))
    allocate(this%v(xms:xme, yms:yme))
    allocate(this%h(xms:xme, yms:yme))

    ! Initialize u
    if (present(u)) then
      do j = yps, ype
        do i = xps, xpe
          this%u(i,j) = u(i - xps + 1, j - yps + 1)
        end do
      end do
    else
      do j = yps, ype
        do i = xps, xpe
          this%u(i,j) = 0.0_r8kind
        end do
      end do
    end if

    ! Initialize v
    if (present(v)) then
      do j = yps, ype
        do i = xps, xpe
          this%v(i,j) = v(i - xps + 1, j - yps + 1)
        end do
      end do
    else
      do j = yps, ype
        do i = xps, xpe
          this%v(i,j) = 0.0_r8kind
        end do
      end do
    end if

    ! Initialize h
    if (present(h)) then
      do j = yps, ype
        do i = xps, xpe
          this%h(i,j) = h(i - xps + 1, j - yps + 1)
        end do
      end do
    else
      do j = yps, ype
        do i = xps, xpe
          this%h(i,j) = 0.0_r8kind
        end do
      end do
    end if

    ! Calculate the maximum wave speed from h
    ! MPI_Allreduce(sendbuf, recvbuf, count, datatype, op, comm, ierror)
    call MPI_Allreduce(maxval(this%h(xps:xpe, yps:ype)), max_h, 1, MPI_DOUBLE_PRECISION, MPI_MAX, this%geometry%get_communicator(), ierr)
    this%max_wavespeed = sqrt(g * max_h)

    ! Initialize clock
    if (present(clock)) then
      this%clock = clock
    else
      this%clock = 0.0_r8kind
    end if

  end function constructor_state


  !------------------------------------------------------------------
  ! destructor
  !
  ! Deallocates pointers used by a shallow_water_state_type object (none currently)
  !------------------------------------------------------------------
  elemental subroutine destructor(this)

    type(shallow_water_state_type), intent(inout) :: this

    ! No pointers in shallow_water_state_type object so we do nothing

  end subroutine destructor


  !------------------------------------------------------------------
  ! exchange_halo
  !
  ! send boundaries to neighboring halos for each process
  !------------------------------------------------------------------
  subroutine exchange_halo(this)

    class(shallow_water_state_type), intent(inout) :: this

    ! MPI variables
    integer            :: communicator
    integer, parameter :: ntag=1, stag=2, wtag=3, etag=4
    integer            :: irequest(8), istatus(MPI_STATUS_SIZE, 8), nrequests, ierr
    real(r8kind)       :: nsendbuffer(this%geometry%get_xps():this%geometry%get_xpe(), 3)
    real(r8kind)       :: ssendbuffer(this%geometry%get_xps():this%geometry%get_xpe(), 3)
    real(r8kind)       :: wsendbuffer(this%geometry%get_yps():this%geometry%get_ype(), 3)
    real(r8kind)       :: esendbuffer(this%geometry%get_yps():this%geometry%get_ype(), 3)
    real(r8kind)       :: nrecvbuffer(this%geometry%get_xps():this%geometry%get_xpe(), 3)
    real(r8kind)       :: srecvbuffer(this%geometry%get_xps():this%geometry%get_xpe(), 3)
    real(r8kind)       :: wrecvbuffer(this%geometry%get_yps():this%geometry%get_ype(), 3)
    real(r8kind)       :: erecvbuffer(this%geometry%get_yps():this%geometry%get_ype(), 3)

    ! Indexing variables
    integer :: i, j
    integer :: xps, xpe, yps, ype
    integer :: xms, xme, yms, yme
    integer :: npx, npy
    integer :: north, south, west, east

    ! Get the MPI communicator from the geometry
    communicator = this%geometry%get_communicator()

    ! Get the index ranges for this patch
    xms = this%geometry%get_xms()
    xme = this%geometry%get_xme()
    yms = this%geometry%get_yms()
    yme = this%geometry%get_yme()
    xps = this%geometry%get_xps()
    xpe = this%geometry%get_xpe()
    yps = this%geometry%get_yps()
    ype = this%geometry%get_ype()

    ! Get the extents of the domain
    npx = this%geometry%get_npx()
    npy = this%geometry%get_npy()

    ! Get MPI ranks of the neighbors of this patch
    north = this%geometry%get_north()
    south = this%geometry%get_south()
    west = this%geometry%get_west()
    east = this%geometry%get_east()

    ! Post the non-blocking receive half of the exhange first to reduce overhead
    ! MPI_IRECV(BUF, COUNT, DATATYPE, SOURCE, TAG, COMM, REQUEST, IERROR)
    nrequests = 0
    if (north /= -1) then
       nrequests = nrequests + 1
       call MPI_IRecv(nrecvbuffer, 3* npx, MPI_DOUBLE_PRECISION, north, stag, communicator, irequest(nrequests), ierr)
    end if
    if (south /= -1) then
       nrequests = nrequests + 1
       call MPI_IRecv(srecvbuffer, 3* npx, MPI_DOUBLE_PRECISION, south, ntag, communicator, irequest(nrequests), ierr)
    end if
    if (west /= -1) then
       nrequests = nrequests + 1
       call MPI_IRecv(wrecvbuffer, 3* npy, MPI_DOUBLE_PRECISION, west, etag, communicator, irequest(nrequests), ierr)
    end if
    if (east /= -1) then
       nrequests = nrequests + 1
       call MPI_IRecv(erecvbuffer, 3* npy, MPI_DOUBLE_PRECISION, east, wtag, communicator, irequest(nrequests), ierr)
    end if

    ! Pack the send buffers
    if (north /= -1) then
      do i = xps, xpe
        nsendbuffer(i, 1) = this%u(i, ype)
        nsendbuffer(i, 2) = this%v(i, ype)
        nsendbuffer(i, 3) = this%h(i, ype)
      end do
    end if
    if (south /= -1) then
      do i = xps, xpe
        ssendbuffer(i, 1) = this%u(i, yps)
        ssendbuffer(i, 2) = this%v(i, yps)
        ssendbuffer(i, 3) = this%h(i, yps)
      end do
    end if
    if (west /= -1) then
      do j = yps, ype
        wsendbuffer(j, 1) = this%u(xps, j)
        wsendbuffer(j, 2) = this%v(xps, j)
        wsendbuffer(j, 3) = this%h(xps, j)
      end do
    end if
    if (east /= -1) then
      do j = yps, ype
        esendbuffer(j, 1) = this%u(xpe, j)
        esendbuffer(j, 2) = this%v(xpe, j)
        esendbuffer(j, 3) = this%h(xpe, j)
      end do
    end if

    ! Now post the non-blocking send half of the exchange
    ! MPI_Isend(buf, count, datatype, dest, tag, comm, request, ierror)
    if (north /= -1) then
       nrequests = nrequests + 1
       call MPI_ISend(nsendbuffer, 3 * npx, MPI_DOUBLE_PRECISION, north, ntag, communicator, irequest(nrequests), ierr)
    end if
    if (south /= -1) then
       nrequests = nrequests + 1
       call MPI_ISend(ssendbuffer, 3 * npx, MPI_DOUBLE_PRECISION, south, stag, communicator, irequest(nrequests), ierr)
    end if
    if (west /= -1) then
       nrequests = nrequests + 1
       call MPI_ISend(wsendbuffer, 3 * npy, MPI_DOUBLE_PRECISION, west, wtag, communicator, irequest(nrequests), ierr)
    end if
    if (east /= -1) then
       nrequests = nrequests + 1
       call MPI_ISend(esendbuffer, 3 * npy, MPI_DOUBLE_PRECISION, east, etag, communicator, irequest(nrequests), ierr)
    end if

    ! Wait for the exchange to complete
    if (nrequests > 0) then
      call MPI_Waitall(nrequests, irequest, istatus, ierr)
    end if

    ! Unpack the receive buffers
    if (north /= -1) then
      do i = xps, xpe
        this%u(i, yme) = nrecvbuffer(i, 1)
        this%v(i, yme) = nrecvbuffer(i, 2)
        this%h(i, yme) = nrecvbuffer(i, 3)
      end do
    end if
    if (south /= -1) then
      do i = xps, xpe
        this%u(i, yms) = srecvbuffer(i, 1)
        this%v(i, yms) = srecvbuffer(i, 2)
        this%h(i, yms) = srecvbuffer(i, 3)
      end do
    end if
    if (west /= -1) then
      do j = yps, ype
        this%u(xms,j) = wrecvbuffer(j,1)
        this%v(xms,j) = wrecvbuffer(j,2)
        this%h(xms,j) = wrecvbuffer(j,3)
      end do
    end if
    if (east /= -1) then
      do j = yps, ype
        this%u(xme,j) = erecvbuffer(j,1)
        this%v(xme,j) = erecvbuffer(j,2)
        this%h(xme,j) = erecvbuffer(j,3)
      end do
    end if

  end subroutine exchange_halo


  !------------------------------------------------------------------
  ! scatter
  !
  ! scatter full state
  !------------------------------------------------------------------
  subroutine scatter(this, u_full, v_full, h_full)

    class(shallow_water_state_type), intent(inout) :: this
    real(r8kind),                    intent(   in) :: u_full(:,:) ! Only valid on MPI rank 0
    real(r8kind),                    intent(   in) :: v_full(:,:) ! Only valid on MPI rank 0
    real(r8kind),                    intent(   in) :: h_full(:,:) ! Only valid on MPI rank 0

    ! MPI variables
    integer                   :: n, ierr
    integer                   :: communicator, myrank, nranks
    integer                   :: xms_local, xme_local, yms_local, yme_local
    integer                   :: nelements_local
    integer,      allocatable :: xms(:), xme(:), yms(:), yme(:)
    integer,      allocatable :: nsend_elements(:)
    integer,      allocatable :: send_offsets(:)
    real(r8kind), allocatable :: send_buffer(:)

    ! Get the MPI communicator from the geometry
    communicator = this%geometry%get_communicator()

    ! Get the number of MPI ranks from the geometry
    nranks = this%geometry%get_nranks()

    ! Get the MPI rank of this process from the geometry
    myrank = this%geometry%get_rank()

    ! Get the local indices (excluding the halo) from the geometry
    xms_local = this%geometry%get_xms()
    xme_local = this%geometry%get_xme()
    yms_local = this%geometry%get_yms()
    yme_local = this%geometry%get_yme()

    ! Calculate the local number of elements
    nelements_local = (xme_local - xms_local + 1) * (yme_local - yms_local + 1)

    ! Allocate space for the indices of each rank
    if (myrank == 0) then
      allocate(xms(nranks))
      allocate(xme(nranks))
      allocate(yms(nranks))
      allocate(yme(nranks))
    else
      ! Allocate on other ranks to avoid triggering debug traps during calls to MPI_Gather
      allocate(xms(1))
      allocate(xme(1))
      allocate(yms(1))
      allocate(yme(1))
    end if

    ! Gather the local indices for each rank
    ! MPI_Gather(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm, ierror)
    call MPI_Gather(xms_local, 1, MPI_INT, xms, 1, MPI_INT, 0, communicator, ierr)
    call MPI_Gather(xme_local, 1, MPI_INT, xme, 1, MPI_INT, 0, communicator, ierr)
    call MPI_Gather(yms_local, 1, MPI_INT, yms, 1, MPI_INT, 0, communicator, ierr)
    call MPI_Gather(yme_local, 1, MPI_INT, yme, 1, MPI_INT, 0, communicator, ierr)

    ! Calculate the number of elements to send to each rank
    if (myrank == 0) then
      allocate(nsend_elements(nranks))
      do n = 1, nranks
        nsend_elements(n) = (xme(n) - xms(n) + 1) * (yme(n) - yms(n) + 1)
      end do
    else
      ! Allocate on other ranks to avoid triggering debug traps during calls to MPI_Scatterv
      allocate(nsend_elements(1))
    end if

    ! Calculate the send buffer offsets for each rank
    if (myrank == 0) then
      allocate(send_offsets(nranks))
      send_offsets(1) = 0
      do n = 2, nranks
        send_offsets(n) = send_offsets(n-1) + nsend_elements(n-1)
      end do
    else
      ! Allocate on other ranks to avoid debug traps during calls to MPI_Scatterv
      allocate(send_offsets(1))
    end if

    ! Allocate a send buffer for scattering u, v, and h
    if (myrank == 0) then
      allocate(send_buffer(sum(nsend_elements)))
    else
      ! Allocate on other ranks to avoid triggering debug traps during calls to MPI_Scatterv
      allocate(send_buffer(1))
    end if

    ! Fill the send buffer and scatter u, v, h
    ! Scatterv(sendbuf, sendcounts, displs, sendtype, recvbuf, recvcount, recvtype, root, comm, ierror)
    if (myrank == 0) then
      do n = 1, nranks
        send_buffer(send_offsets(n) + 1:send_offsets(n) + nsend_elements(n)) = reshape(u_full(xms(n):xme(n), yms(n):yme(n)), (/nsend_elements(n)/))
      end do
    end if
    call MPI_Scatterv(send_buffer, nsend_elements, send_offsets, MPI_DOUBLE_PRECISION, this%u, nelements_local, MPI_DOUBLE_PRECISION, 0, communicator, ierr)
    if (myrank == 0) then
      do n = 1, nranks
        send_buffer(send_offsets(n) + 1:send_offsets(n) + nsend_elements(n)) = reshape(v_full(xms(n):xme(n), yms(n):yme(n)), (/nsend_elements(n)/))
      end do
    end if
    call MPI_Scatterv(send_buffer, nsend_elements, send_offsets, MPI_DOUBLE_PRECISION, this%v, nelements_local, MPI_DOUBLE_PRECISION, 0, communicator, ierr)
    if (myrank == 0) then
      do n = 1, nranks
        send_buffer(send_offsets(n) + 1:send_offsets(n) + nsend_elements(n)) = reshape(h_full(xms(n):xme(n), yms(n):yme(n)), (/nsend_elements(n)/))
      end do
    end if
    call MPI_Scatterv(send_buffer, nsend_elements, send_offsets, MPI_DOUBLE_PRECISION, this%h, nelements_local, MPI_DOUBLE_PRECISION, 0, communicator, ierr)

  end subroutine scatter


  !------------------------------------------------------------------
  ! gather
  !
  ! gather local state
  !------------------------------------------------------------------
  subroutine gather(this, u_full, v_full, h_full)

    class(shallow_water_state_type), intent( in) :: this
    real(r8kind),                    intent(out) :: u_full(:,:) ! Only valid on MPI rank 0
    real(r8kind),                    intent(out) :: v_full(:,:) ! Only valid on MPI rank 0
    real(r8kind),                    intent(out) :: h_full(:,:) ! Only valid on MPI rank 0

    ! Local variables
    integer                   :: n, ierr
    integer                   :: communicator, myrank, nranks
    integer                   :: xps_local, xpe_local, yps_local, ype_local
    integer                   :: nelements_local
    integer,      allocatable :: xps(:), xpe(:), yps(:), ype(:)
    integer,      allocatable :: nrecv_elements(:)
    integer,      allocatable :: recv_offsets(:)
    real(r8kind), allocatable :: recv_buffer(:)

    ! Get the MPI communicator from the geometry
    communicator = this%geometry%get_communicator()

    ! Get the number of MPI ranks from the geometry
    nranks = this%geometry%get_nranks()

    ! Get the MPI rank of this process from the geometry
    myrank = this%geometry%get_rank()

    ! Get the local indices (excluding the halo) from the geometry
    xps_local = this%geometry%get_xps()
    xpe_local = this%geometry%get_xpe()
    yps_local = this%geometry%get_yps()
    ype_local = this%geometry%get_ype()

    ! Calculate the local number of elements
    nelements_local = (xpe_local - xps_local + 1) * (ype_local - yps_local + 1)

    ! Allocate space for the indices of each rank
    if (myrank == 0) then
      allocate(xps(nranks))
      allocate(xpe(nranks))
      allocate(yps(nranks))
      allocate(ype(nranks))
    else
      ! Allocate on other ranks to avoid triggering debug traps during calls to MPI_Gather
      allocate(xps(1))
      allocate(xpe(1))
      allocate(yps(1))
      allocate(ype(1))
    end if

    ! Gather the local indices for each rank
    call MPI_Gather(xps_local, 1, MPI_INT, xps, 1, MPI_INT, 0, communicator, ierr)
    call MPI_Gather(xpe_local, 1, MPI_INT, xpe, 1, MPI_INT, 0, communicator, ierr)
    call MPI_Gather(yps_local, 1, MPI_INT, yps, 1, MPI_INT, 0, communicator, ierr)
    call MPI_Gather(ype_local, 1, MPI_INT, ype, 1, MPI_INT, 0, communicator, ierr)

    ! Calculate the number of elements that will be receieved from each rank
    if (myrank == 0) then
      allocate(nrecv_elements(nranks))
      do n=1, nranks
        nrecv_elements(n) = (xpe(n) - xps(n) + 1) * (ype(n) - yps(n) + 1)
      end do
    else
      ! Allocate on other ranks to avoid triggering debug traps during calls to MPI_Gatherv
      allocate(nrecv_elements(1))
    end if

    ! Calculate the receive buffer offsets for each rank
    if (myrank == 0) then
      allocate(recv_offsets(nranks))
      recv_offsets(1) = 0
      do n=2,nranks
        recv_offsets(n) = recv_offsets(n-1) + nrecv_elements(n-1)
      end do
    else
      ! Allocate on other ranks to avoid debug traps during calls to MPI_Gatherv
      allocate(recv_offsets(1))
    end if

    ! Allocate a receive buffer for gathering u, v, and h
    if (myrank == 0) then
      allocate(recv_buffer(sum(nrecv_elements)))
    else
      ! Allocate on other ranks to avoid debug traps during calls to MPI_Gatherv
      allocate(recv_buffer(1))
    end if

    ! Gather u, v, and h from all ranks and unpack into full size arrays
    ! MPI_Gatherv(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, root, comm, ierror)
    call MPI_Gatherv(this%u(xps_local:xpe_local, yps_local:ype_local), nelements_local, MPI_DOUBLE_PRECISION, recv_buffer, nrecv_elements, recv_offsets, MPI_DOUBLE_PRECISION, 0, communicator, ierr)
    if (myrank ==0) then
      do n=1, nranks
        u_full(xps(n):xpe(n), yps(n):ype(n)) = reshape(recv_buffer(recv_offsets(n)+1:recv_offsets(n)+nrecv_elements(n)), (/xpe(n) - xps(n) + 1, ype(n) - yps(n) + 1/))
      end do
    end if
    call MPI_Gatherv(this%v(xps_local:xpe_local, yps_local:ype_local), nelements_local, MPI_DOUBLE_PRECISION, recv_buffer, nrecv_elements, recv_offsets, MPI_DOUBLE_PRECISION, 0, communicator, ierr)
    if (myrank ==0) then
      do n=1, nranks
        v_full(xps(n):xpe(n), yps(n):ype(n)) = reshape(recv_buffer(recv_offsets(n)+1:recv_offsets(n)+nrecv_elements(n)), (/xpe(n) - xps(n) + 1, ype(n) - yps(n) + 1/))
      end do
    end if
    call MPI_Gatherv(this%h(xps_local:xpe_local, yps_local:ype_local), nelements_local, MPI_DOUBLE_PRECISION, recv_buffer, nrecv_elements, recv_offsets, MPI_DOUBLE_PRECISION, 0, communicator, ierr)
    if (myrank ==0) then
      do n=1, nranks
        h_full(xps(n):xpe(n), yps(n):ype(n)) = reshape(recv_buffer(recv_offsets(n)+1:recv_offsets(n)+nrecv_elements(n)), (/xpe(n) - xps(n) + 1, ype(n) - yps(n) + 1/))
      end do
    end if

  end subroutine gather


  !------------------------------------------------------------------
  ! get_geometry
  !
  ! Get state geometry
  !------------------------------------------------------------------
  pure function get_geometry(this) result(geometry)

    class(shallow_water_state_type), intent(in) :: this
    type(shallow_water_geometry_type)           :: geometry

    geometry = this%geometry

  end function get_geometry


  !------------------------------------------------------------------
  ! get_u
  !
  ! Get state u
  !------------------------------------------------------------------
  pure function get_u(this) result(u)

    class(shallow_water_state_type), intent(in) :: this
    real(r8kind) :: u(this%geometry%get_xps():this%geometry%get_xpe(), this%geometry%get_yps():this%geometry%get_ype())

    u = this%u(this%geometry%get_xps():this%geometry%get_xpe(), this%geometry%get_yps():this%geometry%get_ype())

  end function get_u


  !------------------------------------------------------------------
  ! get_u_ptr
  !
  ! Get pointer to state u
  !------------------------------------------------------------------
  subroutine get_u_ptr(this, u_ptr)

    class(shallow_water_state_type), target, intent(   in) :: this
    real(r8kind), pointer,                   intent(inout) :: u_ptr(:,:)

    u_ptr => this%u

  end subroutine get_u_ptr


  !------------------------------------------------------------------
  ! get_v
  !
  ! Get state v
  !------------------------------------------------------------------
  pure function get_v(this) result(v)

    class(shallow_water_state_type), intent(in) :: this
    real(r8kind) :: v(this%geometry%get_xps():this%geometry%get_xpe(), this%geometry%get_yps():this%geometry%get_ype())

    v = this%v(this%geometry%get_xps():this%geometry%get_xpe(), this%geometry%get_yps():this%geometry%get_ype())

  end function get_v


  !------------------------------------------------------------------
  ! get_v_ptr
  !
  ! Get pointer to state v
  !------------------------------------------------------------------
  subroutine get_v_ptr(this, v_ptr)

    class(shallow_water_state_type), target, intent(   in) :: this
    real(r8kind), pointer,                   intent(inout) :: v_ptr(:,:)

    v_ptr => this%v

  end subroutine get_v_ptr


  !------------------------------------------------------------------
  ! get_h
  !
  ! Get state h
  !------------------------------------------------------------------
  pure function get_h(this) result(h)

    class(shallow_water_state_type), intent(in) :: this
    real(r8kind) :: h(this%geometry%get_xps():this%geometry%get_xpe(), this%geometry%get_yps():this%geometry%get_ype())

    h = this%h(this%geometry%get_xps():this%geometry%get_xpe(), this%geometry%get_yps():this%geometry%get_ype())

  end function get_h


  !------------------------------------------------------------------
  ! get_h_ptr
  !
  ! Get pointer to state h
  !------------------------------------------------------------------
  subroutine get_h_ptr(this, h_ptr)

    class(shallow_water_state_type), target, intent(   in) :: this
    real(r8kind), pointer,                   intent(inout) :: h_ptr(:,:)

    h_ptr => this%h

  end subroutine get_h_ptr


  !------------------------------------------------------------------
  ! get_clock
  !
  ! Get state clock
  !------------------------------------------------------------------
  pure function get_clock(this) result(clock)

    class(shallow_water_state_type), intent(in) :: this
    real(r8kind) :: clock

    clock = this%clock

  end function get_clock


  !------------------------------------------------------------------
  ! advance_clock
  !
  ! Advance clock by dt
  !------------------------------------------------------------------
  subroutine advance_clock(this, dt)

    class(shallow_water_state_type), intent(inout) :: this
    real(r8kind) :: dt

    this%clock = this%clock + dt

  end subroutine advance_clock


  !------------------------------------------------------------------
  ! get_max_wavespeed
  !
  ! Get state max wavespeed
  !------------------------------------------------------------------
  pure function get_max_wavespeed(this) result(max_wavespeed)

    class(shallow_water_state_type), intent(in) :: this
    real(r8kind) :: max_wavespeed

    max_wavespeed = this%max_wavespeed

  end function get_max_wavespeed


  !------------------------------------------------------------------
  ! read
  !
  ! Read state from NetCDF file
  !------------------------------------------------------------------
  subroutine read(this, filename)

    use netcdf

    class(shallow_water_state_type), intent(inout) :: this
    character(len=*),                intent(   in) :: filename

    integer      :: myrank, ierr
    integer      :: nx, ny
    real(r8kind) :: xmax, ymax
    real(r8kind) :: clock

    ! Input grid fields
    real(r8kind), allocatable :: u_full(:,:) ! Grid values of zonal velocity in m/s
    real(r8kind), allocatable :: v_full(:,:) ! Grid values of meridional velocity in m/s
    real(r8kind), allocatable :: h_full(:,:) ! Grid values of pressure surface height in m

    ! General NetCDF variables
    integer :: ncFileID
    integer :: nDimensions, nVariables, nAttributes, unlimitedDimID
    integer :: nxDimID, nyDimID
    integer :: uVarID, vVarID, hVarID

    ! Get the MPI rank of this process from the geometry
    myrank = this%geometry%get_rank()

    ! Read full state from file on rank 0
    if (myrank == 0) then

      ! Open file for read only
      call nc_check(nf90_open(trim(filename), NF90_NOWRITE, ncFileID))
      call nc_check(nf90_Inquire(ncFileID, nDimensions, nVariables, nAttributes, unlimitedDimID))

      ! Read global attributes
      call nc_check(nf90_get_att(ncFileID, NF90_GLOBAL, "xmax", xmax ))
      call nc_check(nf90_get_att(ncFileID, NF90_GLOBAL, "ymax", ymax ))
      call nc_check(nf90_get_att(ncFileID, NF90_GLOBAL, "clock", clock ))

      ! Read the model dimensions
      call nc_check(nf90_inq_dimid(ncFileID, "nx", nxDimID))
      call nc_check(nf90_inquire_dimension(ncFileID, nxDimID, len=nx))
      call nc_check(nf90_inq_dimid(ncFileID, "ny", nyDimID))
      call nc_check(nf90_inquire_dimension(ncFileID, nyDimID, len=ny))

      ! Check to make sure state read in matches this state's geometry
      if (this%geometry%get_nx() /= nx .OR. this%geometry%get_ny() /= ny .OR. this%geometry%get_xmax() /= xmax .OR. this%geometry%get_ymax() /= ymax) then
        call MPI_Abort(this%geometry%get_communicator(), -1, ierr)
      end if

      ! Allocate space for the model variables on the full model grid
      allocate(u_full(nx, ny))
      allocate(v_full(nx, ny))
      allocate(h_full(nx, ny))

      ! Get the u variable
      call nc_check(nf90_inq_varid(ncFileID, "U", uVarID))
      call nc_check(nf90_get_var(ncFileID, uVarID, u_full))

      ! Get the v variable
      call nc_check(nf90_inq_varid(ncFileID, "V", vVarID))
      call nc_check(nf90_get_var(ncFileID, vVarID, v_full))

      ! Get the h variable
      call nc_check(nf90_inq_varid(ncFileID, "H", hVarID))
      call nc_check(nf90_get_var(ncFileID, hVarID, h_full))

      ! Flush buffers
      call nc_check(nf90_sync(ncFileID))

      ! Close the NetCDF file
      call nc_check(nf90_close(ncFileID))

    end if

    ! Scatter u, v, and h
    call this%scatter(u_full, v_full, h_full)

    ! Now broadcast the clock
    if (myrank == 0) then
      this%clock = clock
    end if
    call MPI_Bcast(this%clock, 1, MPI_DOUBLE_PRECISION, 0, this%geometry%get_communicator(), ierr)

  end subroutine read


  !------------------------------------------------------------------
  ! write
  !
  ! Write state to NetCDF file
  !------------------------------------------------------------------
  subroutine write(this, filename)

    use netcdf

    class(shallow_water_state_type), intent(in) :: this
    character(len=*),                intent(in) :: filename

    ! Output grid fields
    real(r8kind), allocatable :: x(:)        ! Grid x dimension
    real(r8kind), allocatable :: y(:)        ! Grid y dimension
    real(r8kind), allocatable :: u_full(:,:) ! Grid values of zonal velocity in m/s
    real(r8kind), allocatable :: v_full(:,:) ! Grid values of meridional velocity in m/s
    real(r8kind), allocatable :: h_full(:,:) ! Grid values of pressure surface height in m

    ! General netCDF variables
    integer :: ncFileID
    integer :: nDimensions, nVariables, nAttributes, unlimitedDimID
    integer :: nxDimID, nyDimID
    integer :: xVarID, yVarID, uVarID, vVarID, hVarID

    ! Local variables
    integer               :: i       ! Loop index variable
    integer               :: myrank  ! MPI rank
    integer               :: nx, ny  ! Dimensions of full grid
    real(r8kind)          :: dx, dy  ! Grid spacing
    character(len=8)      :: crdate  ! Needed by F90 DATE_AND_TIME intrinsic
    character(len=10)     :: crtime  ! Needed by F90 DATE_AND_TIME intrinsic
    character(len=5)      :: crzone  ! Needed by F90 DATE_AND_TIME intrinsic
    integer, dimension(8) :: values  ! Needed by F90 DATE_AND_TIME intrinsic
    character(len=19)     :: timestr ! String representation of clock

    ! Get the MPI rank of this process from the geometry
    myrank = this%geometry%get_rank()

    ! Get the full grid size from the geometry
    nx = this%geometry%get_nx()
    ny = this%geometry%get_ny()

    ! Allocate space for the full state
    if (myrank == 0) then
      allocate(u_full(nx, ny))
      allocate(v_full(nx, ny))
      allocate(h_full(nx, ny))
    end if

    ! Gather full u, v, and h
    call this%gather(u_full, v_full, h_full)

    ! Write the full state
    if (myrank == 0) then

      ! Open new file, overwriting previous contents
      call nc_check(nf90_create(trim(filename), NF90_CLOBBER, ncFileID))
      call nc_check(nf90_Inquire(ncFileID, nDimensions, nVariables, nAttributes, unlimitedDimID))

      ! Write Global Attributes
      call DATE_AND_TIME(crdate,crtime,crzone,values)
      write(timestr,'(i4,2(a,i2.2),1x,i2.2,2(a,i2.2))') &
            values(1), '/', values(2), '/', values(3), values(5), ':', values(6), ':', values(7)
      call nc_check(nf90_put_att(ncFileID, NF90_GLOBAL, "creation_date",timestr))
      call nc_check(nf90_put_att(ncFileID, NF90_GLOBAL, "model_name", "Shallow Water"))
      call nc_check(nf90_put_att(ncFileID, NF90_GLOBAL, "xmax", this%geometry%get_xmax()))
      call nc_check(nf90_put_att(ncFileID, NF90_GLOBAL, "ymax", this%geometry%get_ymax()))
      call nc_check(nf90_put_att(ncFileID, NF90_GLOBAL, "clock", this%clock))

      ! Define the x/y dimensions
      call nc_check(nf90_def_dim(ncid=ncFileID, name="nx", len=nx, dimid = nxDimID))
      call nc_check(nf90_def_dim(ncid=ncFileID, name="ny", len=ny, dimid = nyDimID))

      ! Define the x field
      call nc_check(nf90_def_var(ncid=ncFileID,name="x", xtype=nf90_double, &
                    dimids=(/nxDimID/), varid=xVarID))
      call nc_check(nf90_put_att(ncFileID, xVarID, "long_name", "x"))
      call nc_check(nf90_put_att(ncFileID, xVarID, "units",     "Nondimensional"))

      ! Define the y field
      call nc_check(nf90_def_var(ncid=ncFileID,name="y", xtype=nf90_double, &
                    dimids=(/nyDimID/), varid=yVarID))
      call nc_check(nf90_put_att(ncFileID, yVarID, "long_name", "y"))
      call nc_check(nf90_put_att(ncFileID, yVarID, "units",     "Nondimensional"))

      ! Define the u variable
      call nc_check(nf90_def_var(ncid=ncFileID, name="U", xtype=nf90_double, &
                    dimids=(/nxDimID, nyDimID/), varid=uVarID))
      call nc_check(nf90_put_att(ncFileID, uVarID, "long_name", "Zonal Velocity"))
      call nc_check(nf90_put_att(ncFileID, uVarID, "units", "m / s"))

      ! Define the v variable
      call nc_check(nf90_def_var(ncid=ncFileID, name="V", xtype=nf90_double, &
                    dimids=(/nxDimID, nyDimID/), varid=vVarID))
      call nc_check(nf90_put_att(ncFileID, vVarID, "long_name", "Meridional Velocity"))
      call nc_check(nf90_put_att(ncFileID, vVarID, "units", "m / s"))

      ! Define the h variable
      call nc_check(nf90_def_var(ncid=ncFileID, name="H", xtype=nf90_double, &
                    dimids=(/nxDimID, nyDimID/), varid=hVarID))
      call nc_check(nf90_put_att(ncFileID, hVarID, "long_name", "Pressure Surface Height"))
      call nc_check(nf90_put_att(ncFileID, hVarID, "units", "m"))

      ! Leave define mode so we can fill
      call nc_check(nf90_enddef(ncfileID))

      ! Fill the x variable
      allocate(x(nx))
      dx = this%geometry%get_dx()
      do i=1, nx
        x(i) = (i - 1) * dx
      end do
      call nc_check(nf90_put_var(ncFileID, xVarID, x))

      ! Fill the y variable
      allocate(y(ny))
      dy = this%geometry%get_dy()
      do i=1, ny
        y(i) = (i - 1) * dy
      end do
      call nc_check(nf90_put_var(ncFileID, yVarID, y))

      ! Fill the velocity variables
      call nc_check(nf90_put_var(ncFileID, uVarID, u_full))
      call nc_check(nf90_put_var(ncFileID, vVarID, v_full))

      ! Fill the pressure surface height variable
      call nc_check(nf90_put_var(ncFileID, hVarID, h_full))

      ! Flush buffers
      call nc_check(nf90_sync(ncFileID))

      ! Close the NetCDF file
      call nc_check(nf90_close(ncFileID))

    end if

  end subroutine write


end module Shallow_Water_State
