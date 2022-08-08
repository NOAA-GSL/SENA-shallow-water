program Test_Shallow_Water_Model_Config_Arglist

  use Shallow_Water_Kind,         only : r8kind
  use Shallow_Water_Model_Config, only : shallow_water_model_config_type
  use Test_Utilities,             only : check_integer_scalar, check_real_scalar

  implicit none

  ! Shallow water config object
  type(shallow_water_model_config_type) :: config

  ! Config parameters
  real(r8kind), parameter :: dt = 1.0
  real(r8kind), parameter :: u0 = 2.0
  real(r8kind), parameter :: v0 = 3.0
  real(r8kind), parameter :: b0 = 4.0
  real(r8kind), parameter :: h0 = 5.0

  ! Test variables
  integer :: errors

  ! Initialize error count to 0
  errors = 0

  ! Create a shallow water model configuration from input parameters
  config = shallow_water_model_config_type(dt, u0, v0, b0, h0)

  ! Check dt
  call check_real_scalar(config%get_dt(), "dt", dt, 0.0_r8kind, errors)

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

end program Test_Shallow_Water_Model_Config_Arglist
