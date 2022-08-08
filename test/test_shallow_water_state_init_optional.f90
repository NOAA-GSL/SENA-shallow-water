program Test_Shallow_Water_State_Init_Optional

  use Shallow_Water_Kind,            only : r8kind
  use Shallow_Water_Geometry_Config, only : shallow_water_geometry_config_type
  use Shallow_Water_Geometry,        only : shallow_water_geometry_type
  use Shallow_Water_State,           only : shallow_water_state_type
  use Test_Utilities,                only : check_real_scalar, check_integer_scalar, check_min_max_real2d
  use mpi

  implicit none

  ! Shallow water config object
  type(shallow_water_geometry_config_type) :: config
  type(shallow_water_geometry_type)        :: geometry
  type(shallow_water_state_type)           :: state

  ! Config parameters
  integer, parameter :: nx = 101
  integer, parameter :: ny = 201
  real(r8kind), parameter :: xmax = 100000.0
  real(r8kind), parameter :: ymax = 100000.0

  ! Model variables
  integer                   :: i, j
  integer                   :: xps, xpe, yps, ype
  real(r8kind), allocatable :: u(:,:), v(:,:), h(:,:)
  real(r8kind)              :: umax, umin
  real(r8kind)              :: vmax, vmin
  real(r8kind)              :: hmax, hmin

  ! Test variables
  integer :: errors

  ! MPI variables
  integer :: ierr

  ! Star up MPI
  call MPI_Init(ierr)

  ! Initialize error count to 0
  errors = 0

  ! Create a shallow water geometry configuration from namelist file
  config = shallow_water_geometry_config_type(nx, ny, xmax, ymax)

  ! Create a shallow water geometry from the configuration
  geometry = shallow_water_geometry_type(config, MPI_COMM_WORLD)

  ! Get index ranges from geometry
  xps = geometry%get_xps()
  xpe = geometry%get_xpe()
  yps = geometry%get_yps()
  ype = geometry%get_ype()

  ! Allocate u, v, h
  allocate(u(xps:xpe, yps:ype))
  allocate(v(xps:xpe, yps:ype))
  allocate(h(xps:xpe, yps:ype))

  ! Initialize u, v, h
  do j = yps, ype
    do i = xps, xpe
      u(i, j) = 1.0
      v(i, j) = 2.0
      h(i, j) = 3.0
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

  ! Check u
  call check_min_max_real2d(xps, xpe, yps, ype, state%get_u(), "u", umin, umax, errors)

  ! Check v
  call check_min_max_real2d(xps, xpe, yps, ype, state%get_v(), "v", vmin, vmax, errors)

  ! Check h
  call check_min_max_real2d(xps, xpe, yps, ype, state%get_h(), "h", hmin, hmax, errors)

  if (errors > 0) then
    call MPI_Abort(MPI_COMM_WORLD, errors, ierr)
  end if

  ! Shut down MPI
  call MPI_Finalize(ierr)

end program Test_Shallow_Water_State_Init_Optional
