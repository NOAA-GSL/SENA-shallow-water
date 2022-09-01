module Test_Utilities

use Shallow_Water_Kind, only : r8kind

implicit none

contains

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! check_real_scalar
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
subroutine check_real_scalar(data, name, expected, tolerance, errors)

  real(r8kind), intent(   in) :: data
  character(*), intent(   in) :: name
  real(r8kind), intent(   in) :: expected
  real(r8kind), intent(   in) :: tolerance
  integer,      intent(inout) :: errors

  if (abs(data - expected) > tolerance) then
    write(*,'(3A,E16.4,3A,E16.4)') "ERROR: Expected ", name, " = ", expected, " but got ", name, " = ", data
    errors = errors + 1
  end if

end subroutine check_real_scalar


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! check_integer_scalar
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
subroutine check_integer_scalar(data, name, expected, errors)

  integer,         intent(   in) :: data
  character(*),    intent(   in) :: name
  integer        , intent(   in) :: expected
  integer,         intent(inout) :: errors

  if (data /= expected) then
    write(*,'(3A,I12,3A,I12)') "ERROR: Expected ", name, " = ", expected, " but got ", name, " = ", data
    errors = errors + 1
  end if

end subroutine check_integer_scalar


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! check_min_max_real1d
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
subroutine check_min_max_real1d(xps, xpe, data, name, expected_min, expected_max, errors)

  integer,      intent(   in) :: xps, xpe
  real(r8kind), intent(   in) :: data(xps:xpe)
  character(*), intent(   in) :: name
  real(r8kind), intent(   in) :: expected_min
  real(r8kind), intent(   in) :: expected_max
  integer,      intent(inout) :: errors

  ! Test variables
  real(r8kind) :: float_value

  float_value = minval(data)
  if (float_value /= expected_min) then
    write(*,'(3A,E16.4,3A,E16.4)') "ERROR: Expected minval(", name, ") = ", expected_min, " but got minval(", name, ") = ", float_value
    errors = errors + 1
  end if
  float_value = maxval(data)
  if (float_value /= expected_max) then
    write(*,'(3A,E16.4,3A,E16.4)') "ERROR: Expected maxval(", name, ") = ", expected_max, " but got maxval(", name, ") = ", float_value
    errors = errors + 1
  end if

end subroutine check_min_max_real1d


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! check_min_max_real2d
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
subroutine check_min_max_real2d(xps, xpe, yps, ype, data, name, expected_min, expected_max, errors)

  integer,      intent(   in) :: xps, xpe, yps, ype
  real(r8kind), intent(   in) :: data(xps:xpe, yps:ype)
  character(*), intent(   in) :: name
  real(r8kind), intent(   in) :: expected_min
  real(r8kind), intent(   in) :: expected_max
  integer,      intent(inout) :: errors

  ! Test variables
  real(r8kind) :: float_value

  float_value = minval(data)
  if (float_value /= expected_min) then
    write(*,'(3A,E16.4,3A,E16.4)') "ERROR: Expected minval(", name, ") = ", expected_min, " but got minval(", name, ") = ", float_value
    errors = errors + 1
  end if
  float_value = maxval(data)
  if (float_value /= expected_max) then
    write(*,'(3A,E16.4,3A,E16.4)') "ERROR: Expected maxval(", name, ") = ", expected_max, " but got maxval(", name, ") = ", float_value
    errors = errors + 1
  end if

end subroutine check_min_max_real2d


end module Test_Utilities
