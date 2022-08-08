program Test_Shallow_Water_Model_Config_NLUnit

  use Shallow_Water_Kind,         only : r8kind
  use Shallow_Water_Model_Config, only : shallow_water_model_config_type
  use Test_Utilities,             only : check_integer_scalar, check_real_scalar

  implicit none

  ! Shallow water config object
  type(shallow_water_model_config_type) :: config

  ! Config parameters
  real(r8kind) :: u0
  real(r8kind) :: v0
  real(r8kind) :: b0
  real(r8kind) :: h0
  namelist /model_parm/ u0, v0, b0, h0

  ! Namelist file descriptor
  integer :: nl_unit

  ! Test variables
  integer :: errors

  ! Initialize error count to 0
  errors = 0

  ! Read namelist to get true values
  open(newunit=nl_unit, file='test_shallow_water_config.nl', form='formatted', status='old')
  read(nl_unit, nml=model_parm)
  rewind(nl_unit)

  ! Create a shallow water model configuration from unit number of open namelist file
  config = shallow_water_model_config_type(nl_unit)

  ! Check u0
  call check_real_scalar(config%get_u0(), "u0", u0, 0.0_r8kind, errors)

  ! Check v0
  call check_real_scalar(config%get_v0(), "v0", v0, 0.0_r8kind, errors)

  ! Check b0
  call check_real_scalar(config%get_b0(), "b0", b0, 0.0_r8kind, errors)

  ! Check h0
  call check_real_scalar(config%get_h0(), "h0", h0, 0.0_r8kind, errors)

  if (errors > 0) then
    stop 1
  end if

end program Test_Shallow_Water_Model_Config_NLUnit
