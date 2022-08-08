module Shallow_Water_Geometry_Config

  use Shallow_Water_kind, only : r8kind

  implicit none

  private

  public :: shallow_water_geometry_config_type

  type :: shallow_water_geometry_config_type
    private
    ! Input model configuration parameters
    integer      :: nx
    integer      :: ny
    real(r8kind) :: xmax
    real(r8kind) :: ymax
  contains
    final :: destructor
    procedure :: get_nx
    procedure :: get_ny
    procedure :: get_xmax
    procedure :: get_ymax
    procedure :: print
  end type shallow_water_geometry_config_type

  interface shallow_water_geometry_config_type
    procedure constructor_arglist
    procedure constructor_namelist_file
    procedure constructor_namelist_unit
  end interface

contains

  !------------------------------------------------------------------
  ! constructor_arglist
  !
  ! Returns an initialized shallow_water_geometry_config_type object
  !------------------------------------------------------------------
  pure function constructor_arglist(nx, ny, xmax, ymax) result(config)

    ! Input model configuration parameters
    integer,       intent(in) :: nx
    integer,       intent(in) :: ny
    real(r8kind),  intent(in) :: xmax
    real(r8kind),  intent(in) :: ymax

    ! Return config
    type(shallow_water_geometry_config_type) :: config

    ! Initialize model parameters
    config%nx   = nx
    config%ny   = ny
    config%xmax = xmax
    config%ymax = ymax

  end function constructor_arglist


  !------------------------------------------------------------------
  ! constructor_namelist_file
  !
  ! Returns an initialized shallow_water_geometry_config_type object
  !------------------------------------------------------------------
  function constructor_namelist_file(nl_filename) result(config)

    ! Namelist filename
    character(len=*) :: nl_filename

    ! Return config
    type(shallow_water_geometry_config_type) :: config

    ! Namelist file descriptor
    integer :: nl_unit

    ! Define namelists and default values
    integer      :: nx = 151
    integer      :: ny = 151
    real(r8kind) :: xmax = 100000.0_r8kind
    real(r8kind) :: ymax = 100000.0_r8kind

    ! Namelist for shallow water geometry configuration
    namelist /geometry_parm/ nx, ny, xmax, ymax

    ! Open the namelist file
    open(newunit=nl_unit, file=trim(nl_filename), form='formatted', status='old')

    ! Read the configuration
    read(nl_unit, nml=geometry_parm)

    ! Construct the configuration
    config = shallow_water_geometry_config_type(nx, ny, xmax, ymax)

    ! Close the namelist
    close(nl_unit)

  end function constructor_namelist_file


  !------------------------------------------------------------------
  ! constructor_namelist_unit
  !
  ! Returns an initialized shallow_water_geometry_config_type object
  !------------------------------------------------------------------
  function constructor_namelist_unit(nl_unit) result(config)

    ! Namelist unit number (must be an open file)
    integer :: nl_unit

    ! Return config
    type(shallow_water_geometry_config_type) :: config

    ! Define namelists and default values
    integer      :: nx = 151
    integer      :: ny = 151
    real(r8kind) :: xmax = 100000.0_r8kind
    real(r8kind) :: ymax = 100000.0_r8kind

    ! Namelist for shallow water configuration
    namelist /geometry_parm/ nx, ny, xmax, ymax

    ! Read the configuration
    read(nl_unit, nml=geometry_parm)

    ! Construct the configuration
    config = shallow_water_geometry_config_type(nx, ny, xmax, ymax)

    ! Rewind the namelist
    rewind(nl_unit)

  end function constructor_namelist_unit


  !------------------------------------------------------------------
  ! destructor
  !
  ! Deallocates pointers used by a shallow_water_geometry_config_type object (none currently)
  !------------------------------------------------------------------
  elemental subroutine destructor(this)

    type(shallow_water_geometry_config_type), intent(inout) :: this

    ! No pointers in shallow_water_geometry_config_type object so we do nothing

  end subroutine destructor


  !------------------------------------------------------------------
  ! get_nx
  !
  ! Returns the length of the x dimension
  !------------------------------------------------------------------
  pure function get_nx(this) result(nx)

    class(Shallow_water_geometry_config_type), intent(in) :: this
    integer :: nx

    nx = this%nx

  end function get_nx


  !------------------------------------------------------------------
  ! get_ny
  !
  ! Returns the length of the y dimension
  !------------------------------------------------------------------
  pure function get_ny(this) result(ny)

    class(Shallow_water_geometry_config_type), intent(in) :: this
    integer :: ny

    ny = this%ny

  end function get_ny


  !------------------------------------------------------------------
  ! get_xmax
  !
  ! Returns the extent of the x dimension
  !------------------------------------------------------------------
  pure function get_xmax(this) result(xmax)

    class(Shallow_water_geometry_config_type), intent(in) :: this
    real(r8kind) :: xmax

    xmax = this%xmax

  end function get_xmax


  !------------------------------------------------------------------
  ! get_ymax
  !
  ! Returns the extent of the y dimension
  !------------------------------------------------------------------
  pure function get_ymax(this) result(ymax)

    class(Shallow_water_geometry_config_type), intent(in) :: this
    real(r8kind) :: ymax

    ymax = this%ymax

  end function get_ymax


  !------------------------------------------------------------------
  ! print
  !
  ! Prints contents of config object
  !------------------------------------------------------------------
  subroutine print(this)

    class(Shallow_water_geometry_config_type), intent(in) :: this

    character(len=32) :: numstr

    write(numstr,'(I12)') this%nx
    write(*,'(A20, A)') "nx = ", adjustl(numstr)
    write(numstr,'(I12)') this%ny
    write(*,'(A20, A)') "ny = ", adjustl(numstr)
    write(numstr,'(F16.4)') this%xmax
    write(*,'(A20, A)') "xmax = ", adjustl(numstr)
    write(numstr,'(F16.4)') this%ymax
    write(*,'(A20, A)') "ymax = ", adjustl(numstr)

  end subroutine print


end module Shallow_Water_Geometry_Config
