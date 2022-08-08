program Test_Shallow_Water_Model_Init

  use Shallow_Water_Kind,            only : r8kind
  use Shallow_Water_Geometry_Config, only : shallow_water_geometry_config_type
  use Shallow_Water_Model_Config,    only : shallow_water_model_config_type
  use Shallow_Water_Geometry,        only : shallow_water_geometry_type
  use Shallow_Water_Model,           only : shallow_water_model_type
  use Test_Utilities,                only : check_real_scalar
  use mpi

  implicit none

  ! Shallow water config object
  type(shallow_water_geometry_config_type) :: geometry_config
  type(shallow_water_model_config_type)    :: model_config
  type(shallow_water_geometry_type)        :: geometry
  type(shallow_water_model_type)           :: model

  ! Physical constants
  real(r8kind), parameter :: g  = 9.81_r8kind

  ! Config parameters
  integer, parameter :: nx = 101
  integer, parameter :: ny = 201
  real(r8kind), parameter :: xmax = 100000.0
  real(r8kind), parameter :: ymax = 100000.0
  real(r8kind), parameter :: u0 = 0.0
  real(r8kind), parameter :: v0 = 0.0
  real(r8kind), parameter :: b0 = 0.0
  real(r8kind), parameter :: h0 = 5030.0

  ! Model parameters
  real(r8kind), parameter :: dx = 1000.0
  real(r8kind), parameter :: dy = 500.0
  real(r8kind), parameter :: dt = 0.68_r8kind * dx / (u0 + sqrt(g * (h0 - b0)))

  ! Test variables
  integer :: errors

  ! MPI variables
  integer :: ierr

  ! Start MPI
  call MPI_Init(ierr)

  ! Initialize error count to 0
  errors = 0

  ! Create a shallow water geometry configuration from namelist file
  geometry_config = shallow_water_geometry_config_type(nx, ny, xmax, ymax)

  ! Create a shallow water geometry from the configuration
  geometry = shallow_water_geometry_type(geometry_config, MPI_COMM_WORLD)

  ! Create a shallow water model configuration from namelist file
  model_config = shallow_water_model_config_type(dt, u0, v0, b0, h0)

  ! Initialize shallow water model
  model = shallow_water_model_type(model_config, geometry)

  ! Get the model config
  model_config = model%get_config()

  ! Get the model geometry
  geometry = model%get_geometry()

  ! Check dx
  call check_real_scalar(geometry%get_dx(), "dx", dx, 10E-12_r8kind, errors)

  ! Check dy
  call check_real_scalar(geometry%get_dy(), "dy", dy, 10E-12_r8kind, errors)

  ! Check dt
  call check_real_scalar(model%get_dt(), "dt", dt, 10E-12_r8kind, errors)

  if (errors > 0) then
    call MPI_Abort(MPI_COMM_WORLD, errors, ierr)
  end if

  ! Finalize MPI
  call MPI_Finalize(ierr)

end program Test_Shallow_Water_Model_Init
