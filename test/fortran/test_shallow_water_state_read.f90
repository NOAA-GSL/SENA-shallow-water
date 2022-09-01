program Test_Shallow_Water_State_Read

  use Shallow_Water_Kind,            only : r8kind
  use Shallow_Water_Geometry_Config, only : shallow_water_geometry_config_type
  use Shallow_Water_Geometry,        only : shallow_water_geometry_type
  use Shallow_Water_State,           only : shallow_water_state_type
  use Test_Utilities,                only : check_integer_scalar, check_real_scalar, check_min_max_real2d
  use mpi

  implicit none

  ! Shallow water config object
  type(shallow_water_geometry_config_type) :: geometry_config
  type(shallow_water_geometry_type)        :: geometry
  type(shallow_water_state_type)           :: state

  ! Config parameters
  integer :: nx
  integer :: ny
  real(r8kind) :: xmax
  real(r8kind) :: ymax

  ! Model parameters
  real(r8kind), allocatable :: u(:,:), v(:,:), h(:,:)
  real(r8kind) :: clock
  real(r8kind) :: umax, umin
  real(r8kind) :: vmax, vmin
  real(r8kind) :: hmax, hmin

  ! Test variables
  integer :: errors

  ! MPI variables
  integer :: ierr

  ! Start up MPI
  call MPI_Init(ierr)

  ! Set expected geometry
  nx = 10
  ny = 10
  xmax = 10000.0
  ymax = 10000.0

  ! Set expected state
  allocate(u(nx,ny))
  allocate(v(nx,ny))
  allocate(h(nx,ny))
  u = 5.0
  v = 6.0
  h = 7.0
  umax=maxval(u)
  umin=minval(u)
  vmax=maxval(v)
  vmin=minval(v)
  hmax=maxval(h)
  hmin=minval(h)
  clock = 182850.615187004_r8kind

  ! Initialize error count to 0
  errors = 0

  ! Initialize the geometry config
  geometry_config = shallow_water_geometry_config_type(nx, ny, xmax, ymax)

  ! Initialize the geometry
  geometry = shallow_water_geometry_type(geometry_config, MPI_COMM_WORLD)

  ! Initialize a default state
  state = shallow_water_state_type(geometry)

  ! Read in state from the input file
  call state%read("test_shallow_water_reader.nc")

  ! Check clock
  call check_real_scalar(state%get_clock(), "clock", clock, 10E-9_r8kind, errors)

  ! Check u
  call check_min_max_real2d(geometry%get_xps(), geometry%get_xpe(), geometry%get_yps(), geometry%get_ype(), state%get_u(), "u", umin, umax, errors)

  ! Check v
  call check_min_max_real2d(geometry%get_xps(), geometry%get_xpe(), geometry%get_yps(), geometry%get_ype(), state%get_v(), "v", vmin, vmax, errors)

  ! Check h
  call check_min_max_real2d(geometry%get_xps(), geometry%get_xpe(), geometry%get_yps(), geometry%get_ype(), state%get_h(), "h", hmin, hmax, errors)

  if (errors > 0) then
    call MPI_Abort(MPI_COMM_WORLD, errors, ierr)
  end if

  ! Shut down MPI
  call MPI_Finalize(ierr)

end program Test_Shallow_Water_State_Read
