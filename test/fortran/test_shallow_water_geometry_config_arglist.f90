program Test_Shallow_Water_Geometry_Config_Arglist

  use Shallow_Water_Kind,            only : r8kind
  use Shallow_Water_Geometry_Config, only : shallow_water_geometry_config_type
  use Test_Utilities,                only : check_integer_scalar, check_real_scalar

  implicit none

  ! Shallow water config object
  type(shallow_water_geometry_config_type) :: config

  ! Config parameters
  integer, parameter :: nx = 100
  integer, parameter :: ny = 200
  real(r8kind), parameter :: xmax = 500.0
  real(r8kind), parameter :: ymax = 600.0

  ! Test variables
  integer :: errors

  ! Initialize error count to 0
  errors = 0

  ! Create a shallow water model configuration from input parameters
  config = shallow_water_geometry_config_type(nx, ny, xmax, ymax)

  ! Check nx
  call check_integer_scalar(config%get_nx(), "nx", nx, errors)

  ! Check ny
  call check_integer_scalar(config%get_ny(), "ny", ny, errors)

  ! Check xmax
  call check_real_scalar(config%get_xmax(), "xmax", xmax, 0.0_r8kind, errors)

  ! Check ymax
  call check_real_scalar(config%get_ymax(), "ymax", ymax, 0.0_r8kind, errors)

  if (errors > 0) then
    stop 1
  end if

end program Test_Shallow_Water_Geometry_Config_Arglist
