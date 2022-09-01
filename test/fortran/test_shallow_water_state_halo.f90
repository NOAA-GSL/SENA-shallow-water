program Test_Shallow_Water_State_Halo

  use Shallow_Water_Kind,            only : r8kind
  use Shallow_Water_Geometry_Config, only : shallow_water_geometry_config_type
  use Shallow_Water_Geometry,        only : shallow_water_geometry_type
  use Shallow_Water_State,           only : shallow_water_state_type
  use Test_Utilities,                only : check_real_scalar, check_integer_scalar, check_min_max_real1d, check_min_max_real2d
  use mpi

  implicit none

  ! Shallow water config object
  type(shallow_water_geometry_config_type) :: geometry_config
  type(shallow_water_geometry_type)        :: geometry
  type(shallow_water_state_type)           :: state

  ! Config parameters
  integer, parameter :: step = 1000
  integer, parameter :: nx = 101
  integer, parameter :: ny = 201
  real(r8kind), parameter :: xmax = 100000.0
  real(r8kind), parameter :: ymax = 100000.0

  ! Model variables
  integer                   :: i, j
  integer                   :: xms, xme, yms, yme
  integer                   :: xps, xpe, yps, ype
  integer                   :: north, south, east, west
  real(r8kind), allocatable :: u(:,:), v(:,:), h(:,:)
  real(r8kind)              :: umax, umin
  real(r8kind)              :: vmax, vmin
  real(r8kind)              :: hmax, hmin

  ! Test variables
  integer :: errors

  ! MPI variables
  integer :: ierr
  integer :: myrank

  ! Star up MPI
  call MPI_Init(ierr)

  ! Get our MPI rank
  call MPI_Comm_rank(MPI_COMM_WORLD, myrank, ierr)

  ! Initialize error count to 0
  errors = 0

  ! Create a shallow water model configuration from namelist file
  geometry_config = shallow_water_geometry_config_type(nx, ny, xmax, ymax)

  ! Create a shallow water geometry from the configuration
  geometry = shallow_water_geometry_type(geometry_config, MPI_COMM_WORLD)

  ! Get memory index ranges from geometry
  xms = geometry%get_xms()
  xme = geometry%get_xme()
  yms = geometry%get_yms()
  yme = geometry%get_yme()

  ! Get patch index ranges from geometry
  xps = geometry%get_xps()
  xpe = geometry%get_xpe()
  yps = geometry%get_yps()
  ype = geometry%get_ype()

  ! Get ranks of our neighbors
  north = geometry%get_north()
  south = geometry%get_south()
  west = geometry%get_west()
  east = geometry%get_east()

  ! Allocate u, v, h
  allocate(u(xps:xpe, yps:ype))
  allocate(v(xps:xpe, yps:ype))
  allocate(h(xps:xpe, yps:ype))

  ! Initialize u, v, h to a function of our rank
  do j = yps, ype
    do i = xps, xpe
      u(i, j) = 10.0_r8kind * myrank
      v(i, j) = 20.0_r8kind * myrank
      h(i, j) = 30.0_r8kind * myrank
    end do
  end do

  umax=maxval(u)
  umin=minval(u)
  vmax=maxval(v)
  vmin=minval(v)
  hmax=maxval(h)
  hmin=minval(h)

  ! Create a shallow water state
  state = shallow_water_state_type(geometry, u=u, v=v, h=h)

  ! Do a halo exchange
  call state%exchange_halo()

  ! Check interior of the state to make sure it hasn't changed
  call check_min_max_real2d(xps, xpe, yps, ype, state%get_u(), "u", umin, umax, errors)
  call check_min_max_real2d(xps, xpe, yps, ype, state%get_v(), "v", vmin, vmax, errors)
  call check_min_max_real2d(xps, xpe, yps, ype, state%get_h(), "h", hmin, hmax, errors)

  ! Now check our halo to make sure it contains our neighbor's values
  if (north /= -1) then
    call check_min_max_real1d(xps, xpe, state%u(xps:xpe, yme), "u", 10.0_r8kind * north, 10.0_r8kind * north, errors)
    call check_min_max_real1d(xps, xpe, state%v(xps:xpe, yme), "v", 20.0_r8kind * north, 20.0_r8kind * north, errors)
    call check_min_max_real1d(xps, xpe, state%h(xps:xpe, yme), "h", 30.0_r8kind * north, 30.0_r8kind * north, errors)    
  end if
  if (south /= -1) then
    call check_min_max_real1d(xps, xpe, state%u(xps:xpe, yms), "u", 10.0_r8kind * south, 10.0_r8kind * south, errors)
    call check_min_max_real1d(xps, xpe, state%v(xps:xpe, yms), "v", 20.0_r8kind * south, 20.0_r8kind * south, errors)
    call check_min_max_real1d(xps, xpe, state%h(xps:xpe, yms), "h", 30.0_r8kind * south, 30.0_r8kind * south, errors)    
  end if
  if (west /= -1) then
    call check_min_max_real1d(xps, xpe, state%u(xms, yps:ype), "u", 10.0_r8kind * west, 10.0_r8kind * west, errors)
    call check_min_max_real1d(xps, xpe, state%v(xms, yps:ype), "v", 20.0_r8kind * west, 20.0_r8kind * west, errors)
    call check_min_max_real1d(xps, xpe, state%h(xms, yps:ype), "h", 30.0_r8kind * west, 30.0_r8kind * west, errors)    
  end if
  if (east /= -1) then
    call check_min_max_real1d(xps, xpe, state%u(xme, yps:ype), "u", 10.0_r8kind * east, 10.0_r8kind * east, errors)
    call check_min_max_real1d(xps, xpe, state%v(xme, yps:ype), "v", 20.0_r8kind * east, 20.0_r8kind * east, errors)
    call check_min_max_real1d(xps, xpe, state%h(xme, yps:ype), "h", 30.0_r8kind * east, 30.0_r8kind * east, errors)    
  end if

  if (errors > 0) then
    call MPI_Abort(MPI_COMM_WORLD, errors, ierr)
  end if

  ! Shut down MPI
  call MPI_Finalize(ierr)

end program Test_Shallow_Water_State_Halo
