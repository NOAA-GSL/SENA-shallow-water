program Test_Shallow_Water_Geometry_Config_NLFile

  use Shallow_Water_Kind,            only : r8kind
  use Shallow_Water_Geometry_Config, only : shallow_water_geometry_config_type
  use Test_Utilities,                only : check_integer_scalar, check_real_scalar

  implicit none

  ! Shallow water config object
  type(shallow_water_geometry_config_type) :: config

  ! Config parameters
  integer :: nx
  integer :: ny
  real(r8kind) :: xmax
  real(r8kind) :: ymax
  namelist /geometry_parm/ nx, ny, xmax, ymax

  ! Namelist file descriptor
  integer :: nl_unit

  ! Test variables
  integer :: errors

  ! Initialize error count to 0
  errors = 0

  ! Read namelist to get true values
  open(newunit=nl_unit, file='test_shallow_water_config.nl', form='formatted', status='old')
  read(nl_unit, nml=geometry_parm)
  close(nl_unit)

  ! Create a shallow water model configuration from namelist file
  config = shallow_water_geometry_config_type("test_shallow_water_config.nl")

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

end program Test_Shallow_Water_Geometry_Config_NLFile
