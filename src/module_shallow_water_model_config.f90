module Shallow_Water_Model_Config

  use Shallow_Water_kind, only : r8kind

  implicit none

  private

  public :: shallow_water_model_config_type

  type :: shallow_water_model_config_type
    private
    ! Input model configuration parameters
    real(r8kind) :: dt
    real(r8kind) :: u0
    real(r8kind) :: v0
    real(r8kind) :: b0
    real(r8kind) :: h0
  contains
    final :: destructor
    procedure :: get_dt
    procedure :: get_u0
    procedure :: get_v0
    procedure :: get_b0
    procedure :: get_h0
    procedure :: print
  end type shallow_water_model_config_type

  interface shallow_water_model_config_type
    procedure constructor_arglist
    procedure constructor_namelist_file
    procedure constructor_namelist_unit
  end interface

contains

  !------------------------------------------------------------------
  ! constructor_arglist
  !
  ! Returns an initialized shallow_water_model_config_type object
  !------------------------------------------------------------------
  pure function constructor_arglist(dt, u0, v0, b0, h0) result(config)

    ! Input model configuration parameters
    real(r8kind),  intent(in) :: dt
    real(r8kind),  intent(in) :: u0
    real(r8kind),  intent(in) :: v0
    real(r8kind),  intent(in) :: b0
    real(r8kind),  intent(in) :: h0

    ! Return config
    type(shallow_water_model_config_type) :: config

    ! Initialize model parameters
    config%dt = dt
    config%u0 = u0
    config%v0 = v0
    config%b0 = b0
    config%h0 = h0

  end function constructor_arglist


  !------------------------------------------------------------------
  ! constructor_namelist_file
  !
  ! Returns an initialized shallow_water_model_config_type object
  !------------------------------------------------------------------
  function constructor_namelist_file(nl_filename) result(config)

    ! Namelist filename
    character(len=*) :: nl_filename

    ! Return config
    type(shallow_water_model_config_type) :: config

    ! Namelist file descriptor
    integer :: nl_unit

    ! Define namelists and default values
    real(r8kind) :: dt = 0.0_r8kind
    real(r8kind) :: u0 = 0.0_r8kind
    real(r8kind) :: v0 = 0.0_r8kind
    real(r8kind) :: b0 = 0.0_r8kind
    real(r8kind) :: h0 = 5030.0_r8kind

    ! Namelist for shallow water configuration
    namelist /model_parm/ dt, u0, v0, b0, h0

    ! Open the namelist file
    open(newunit=nl_unit, file=trim(nl_filename), form='formatted', status='old')

    ! Read the configuration
    read(nl_unit, nml=model_parm)

    ! Construct the configuration
    config = shallow_water_model_config_type(dt, u0, v0, b0, h0)

    ! Close the namelist
    close(nl_unit)

  end function constructor_namelist_file


  !------------------------------------------------------------------
  ! constructor_namelist_unit
  !
  ! Returns an initialized shallow_water_model_config_type object
  !------------------------------------------------------------------
  function constructor_namelist_unit(nl_unit) result(config)

    ! Namelist unit number (must be an open file)
    integer :: nl_unit

    ! Return config
    type(shallow_water_model_config_type) :: config

    ! Define namelists and default values
    real(r8kind) :: dt = 0.0_r8kind
    real(r8kind) :: u0 = 0.0_r8kind
    real(r8kind) :: v0 = 0.0_r8kind
    real(r8kind) :: b0 = 0.0_r8kind
    real(r8kind) :: h0 = 5030.0_r8kind

    ! Namelist for shallow water configuration
    namelist /model_parm/ dt, u0, v0, b0, h0

    ! Read the configuration
    read(nl_unit, nml=model_parm)

    ! Construct the configuration
    config = shallow_water_model_config_type(dt, u0, v0, b0, h0)

    ! Rewind the namelist
    rewind(nl_unit)

  end function constructor_namelist_unit


  !------------------------------------------------------------------
  ! destructor
  !
  ! Deallocates pointers used by a shallow_water_model_config_type object (none currently)
  !------------------------------------------------------------------
  elemental subroutine destructor(this)

    type(shallow_water_model_config_type), intent(inout) :: this

    ! No pointers in shallow_water_model_config_type object so we do nothing

  end subroutine destructor


  !------------------------------------------------------------------
  ! get_dt
  !------------------------------------------------------------------
  pure function get_dt(this) result(dt)

    class(Shallow_water_model_config_type), intent(in) :: this
    real(r8kind) :: dt

    dt = this%dt

  end function get_dt

  !------------------------------------------------------------------
  ! get_u0
  !------------------------------------------------------------------
  pure function get_u0(this) result(u0)

    class(Shallow_water_model_config_type), intent(in) :: this
    real(r8kind) :: u0

    u0 = this%u0

  end function get_u0

  !------------------------------------------------------------------
  ! get_v0
  !------------------------------------------------------------------
  pure function get_v0(this) result(v0)

    class(Shallow_water_model_config_type), intent(in) :: this
    real(r8kind) :: v0

    v0 = this%v0

  end function get_v0

  !------------------------------------------------------------------
  ! get_b0
  !------------------------------------------------------------------
  pure function get_b0(this) result(b0)

    class(Shallow_water_model_config_type), intent(in) :: this
    real(r8kind) :: b0

    b0 = this%b0

  end function get_b0

  !------------------------------------------------------------------
  ! get_h0
  !------------------------------------------------------------------
  pure function get_h0(this) result(h0)

    class(Shallow_water_model_config_type), intent(in) :: this
    real(r8kind) :: h0

    h0 = this%h0

  end function get_h0


  !------------------------------------------------------------------
  ! print
  !
  ! Prints contents of config object
  !------------------------------------------------------------------
  subroutine print(this)

    class(Shallow_water_model_config_type), intent(in) :: this

    character(len=32) :: numstr

    write(numstr,'(F16.4)') this%dt
    write(*,'(A20, A)') "dt = ", adjustl(numstr)
    write(numstr,'(F16.4)') this%u0
    write(*,'(A20, A)') "u0 = ", adjustl(numstr)
    write(numstr,'(F16.4)') this%v0
    write(*,'(A20, A)') "v0 = ", adjustl(numstr)
    write(numstr,'(F16.4)') this%b0
    write(*,'(A20, A)') "b0 = ", adjustl(numstr)
    write(numstr,'(F16.4)') this%h0
    write(*,'(A20, A)') "h0 = ", adjustl(numstr)

  end subroutine print


end module Shallow_Water_Model_Config
