program Test_Shallow_Water_State_Init_Default

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

  ! Model parameters
  real(r8kind), parameter :: umax=0.0, umin=0.0
  real(r8kind), parameter :: vmax=0.0, vmin=0.0
  real(r8kind), parameter :: hmax=0.0, hmin=0.0

  ! Test variables
  integer :: errors

  ! MPI variables
  integer :: ierr

  ! Start MPI
  call MPI_Init(ierr)

  ! Initialize error count to 0
  errors = 0

  ! Create a shallow water geometry configuration from namelist file
  config = shallow_water_geometry_config_type(nx, ny, xmax, ymax)

  ! Create a shallow water geometry from the configuration
  geometry = shallow_water_geometry_type(config, MPI_COMM_WORLD)

  ! Initialize shallow water state
  state = shallow_water_state_type(geometry)

  ! Check u
  call check_min_max_real2d(geometry%get_xps(), geometry%get_xpe(), geometry%get_yps(), geometry%get_ype(), state%get_u(), "u", umin, umax, errors)

  ! Check v
  call check_min_max_real2d(geometry%get_xps(), geometry%get_xpe(), geometry%get_yps(), geometry%get_ype(), state%get_v(), "v", vmin, vmax, errors)

  ! Check h
  call check_min_max_real2d(geometry%get_xps(), geometry%get_xpe(), geometry%get_yps(), geometry%get_ype(), state%get_h(), "h", hmin, hmax, errors)

  if (errors > 0) then
    call MPI_Abort(MPI_COMM_WORLD, errors, ierr)
  end if

  ! Finalize MPI
  call MPI_Finalize(ierr)

end program Test_Shallow_Water_State_Init_Default
