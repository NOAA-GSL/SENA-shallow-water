program Test_Shallow_Water_State_Write

  use Shallow_Water_Kind,            only : r8kind
  use Shallow_Water_Geometry_Config, only : shallow_water_geometry_config_type
  use Shallow_Water_Geometry,        only : shallow_water_geometry_type
  use Shallow_Water_State,           only : shallow_water_state_type
  use Test_Utilities,                only : check_real_scalar, check_integer_scalar, check_min_max_real2d
  use mpi

  implicit none

  ! Shallow water config object
  type(shallow_water_geometry_config_type) :: geometry_config
  type(shallow_water_geometry_type)        :: geometry
  type(shallow_water_state_type)           :: state, state_read

  ! Geometry parameters
  integer, parameter :: nx = 10
  integer, parameter :: ny = 10
  real(r8kind), parameter :: xmax = 10000.0
  real(r8kind), parameter :: ymax = 10000.0

  ! State variables
  integer                   :: xps, xpe, yps, ype
  real(r8kind), allocatable :: u(:,:), v(:,:), h(:,:)
  real(r8kind), parameter   :: clock = 1000.0

  ! Test variables
  integer :: errors=0

  ! MPI variables
  integer :: ierr

  ! Start up MPI
  call MPI_Init(ierr) 

  ! Create a shallow water geometry configuration
  geometry_config = shallow_water_geometry_config_type(nx, ny, xmax, ymax)

  ! Create a shallow water geometry
  geometry = shallow_water_geometry_type(geometry_config, MPI_COMM_WORLD)

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
  u = 5.0
  v = 6.0
  h = 7.0

  ! Create a shallow water state
  state = shallow_water_state_type(geometry, u=u, v=v, h=h, clock=clock)

  ! Write the state to the output file
  call state%write("test_shallow_water_writer.nc")

  ! Now create a new empty state and read the state back in
  state_read = shallow_water_state_type(geometry)
  call state_read%read("test_shallow_water_writer.nc")

  ! Check u
  call check_min_max_real2d(xps, xpe, yps, ype, state_read%get_u(), "u", minval(u), maxval(u), errors)

  ! Check v
  call check_min_max_real2d(xps, xpe, yps, ype, state_read%get_v(), "v", minval(v), maxval(v), errors)

  ! Check h
  call check_min_max_real2d(xps, xpe, yps, ype, state_read%get_h(), "h", minval(h), maxval(h), errors)

  if (errors > 0) then
    call MPI_Abort(MPI_COMM_WORLD, errors, ierr)
  end if

  ! Shut down MPI
  call MPI_Finalize(ierr) 

end program Test_Shallow_Water_State_Write
