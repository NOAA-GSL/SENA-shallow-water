module Shallow_Water_Geometry

  use Shallow_Water_Kind,            only : r8kind
  use Shallow_Water_Geometry_Config, only : shallow_water_geometry_config_type
  use mpi

  implicit none

  private

  ! Physical constants
  real(r8kind), parameter :: g  = 9.81_r8kind

  !------------------------------------------------------------------
  ! shallow_water_geometry_type
  !------------------------------------------------------------------
  public :: shallow_water_geometry_type

!  type, extends(abstract_geometry_type) :: shallow_water_geometry_type
  type :: shallow_water_geometry_type

    private
    integer      :: nx    ! Number of grid points in the x direction
    integer      :: ny    ! Number of grid points in the y direction
    real(r8kind) :: xmax  ! Maximum extent of the domain in the x direction
    real(r8kind) :: ymax  ! Maximum extent of the domain in the y direction
    real(r8kind) :: dx    ! Grid spacing in the x direction
    real(r8kind) :: dy    ! Grid spacing in the y direction

    integer      :: mpi_comm  ! MPI communicator
    integer      :: nranks    ! Total number of MPI ranks
    integer      :: rank      ! MPI rank of this task
    integer      :: nxprocs   ! Size of the processor grid in the x direction
    integer      :: nyprocs   ! Size of the processor grid in the y direction
    integer      :: xproc     ! Processor grid coordinate of this MPI task in the x direction
    integer      :: yproc     ! Processor grid coordinate of this MPI task in the y direction
    integer      :: north     ! MPI rank of northern neighbor
    integer      :: south     ! MPI rank of southern neighbor
    integer      :: west      ! MPI rank of western neighbor
    integer      :: east      ! MPI rank of eastern neighbor
    integer      :: npx, npy  ! Extent of the domain for this patch in x/y directions
    integer      :: xps, xpe  ! Start/end indices of this grid patch in the x direction
    integer      :: yps, ype  ! Start/end indices of this grid patch in the y direction
    integer      :: xts, xte  ! Start/end indices of interior points for this grid patch in the x direction
    integer      :: yts, yte  ! Start/end indices of interior points for this grid patch in the y direction
    integer      :: xms, xme  ! Start/end indices of the memory allocated for this grid patch in the x direction
    integer      :: yms, yme  ! Start/end indices of the memory allocated for this grid patch in the y direction

  contains
    final :: destructor_geometry
    procedure, public :: get_communicator
    procedure, public :: get_rank
    procedure, public :: get_nranks
    procedure, public :: get_nx
    procedure, public :: get_ny
    procedure, public :: get_xmax
    procedure, public :: get_ymax
    procedure, public :: get_dx
    procedure, public :: get_dy
    procedure, public :: get_north
    procedure, public :: get_south
    procedure, public :: get_west
    procedure, public :: get_east
    procedure, public :: get_npx
    procedure, public :: get_npy
    procedure, public :: get_xps
    procedure, public :: get_xpe
    procedure, public :: get_yps
    procedure, public :: get_ype
    procedure, public :: get_xts
    procedure, public :: get_xte
    procedure, public :: get_yts
    procedure, public :: get_yte
    procedure, public :: get_xms
    procedure, public :: get_xme
    procedure, public :: get_yms
    procedure, public :: get_yme
  end type shallow_water_geometry_type

  interface shallow_water_geometry_type
    procedure constructor_geometry
  end interface

contains

  !------------------------------------------------------------------
  ! constructor
  !
  ! Returns an initialized shallow_water_geometry_type object
  !------------------------------------------------------------------
  function constructor_geometry(config, mpi_comm) result(this)

    type(shallow_water_geometry_config_type), intent(in) :: config
    integer,                                  intent(in) :: mpi_comm

    ! Return a shallow water geometry object
    type(shallow_water_geometry_type) :: this

    ! MPI variables
    integer :: ierr
    integer :: factor

    ! Get geometry grid dimensions from config
    this%nx = config%get_nx()
    this%ny = config%get_ny()

    ! Get geometry domain dimensions from config
    this%xmax = config%get_xmax()
    this%ymax = config%get_ymax()

    ! Define the geometry grid spacing
    this%dx = this%xmax / (dble(this%nx) - 1.0_r8kind)
    this%dy = this%ymax / (dble(this%ny) - 1.0_r8kind)

    ! Initialize the MPI communicator
    this%mpi_comm = mpi_comm

    ! Get the number of MPI ranks
    call MPI_Comm_size(this%mpi_comm, this%nranks, ierr)

    ! Get the rank of this MPI process
    call MPI_Comm_rank(this%mpi_comm, this%rank, ierr)

    ! Compute the size of the processor grid
    factor = this%nranks**0.5 + 0.5
    do while (mod(this%nranks, factor) /= 0 )
      factor = factor - 1
    end do
    if (this%nx >= this%ny) then
      this%nxprocs = factor
    else
      this%nxprocs = this%nranks / factor
    end if
    this%nyprocs = this%nranks / this%nxprocs

    ! Compute the processor coordinate for this rank
    this%xproc = mod(this%rank, this%nxprocs)
    this%yproc = this%rank / this%nxprocs

    ! Compute the ranks of our neighbors
    if (this%yproc == this%nyprocs - 1) then
      this%north = -1
    else
      this%north = (this%yproc + 1) * this%nxprocs + this%xproc
    end if
    if (this%yproc == 0) then
      this%south = -1
    else
      this%south = (this%yproc - 1) * this%nxprocs + this%xproc
    end if
    if (this%xproc == 0) then
      this%west = -1
    else
      this%west  = this%yproc * this%nxprocs + this%xproc - 1
    end if
    if (this%xproc == this%nxprocs - 1) then
      this%east = -1
    else
      this%east = this%yproc * this%nxprocs + this%xproc + 1
    end if

    ! Compute the size of the x and y extents for this patch of the domain
    this%npx = this%nx / this%nxprocs
    if (this%xproc >= (this%nxprocs - mod(this%nx, this%nxprocs))) then
      this%npx = this%npx + 1
    end if
    this%npy = this%ny / this%nyprocs
    if (this%yproc >= (this%nyprocs - mod(this%ny, this%nyprocs))) then
      this%npy = this%npy + 1
    end if

    ! Compute the start/end indices for this patch of the domain
    this%xps = this%nx / this%nxprocs * this%xproc + 1 + max(0, this%xproc - (this%nxprocs - mod(this%nx, this%nxprocs)))
    this%xpe = this%xps + this%npx - 1
    this%yps = this%ny / this%nyprocs * this%yproc + 1 + max(0, this%yproc - (this%nyprocs - mod(this%ny, this%nyprocs)))
    this%ype = this%yps + this%npy - 1

    ! Compute the start/end indices for the interior points and memory allocated for this patch
    if (this%north == -1) then
      this%yte = this%ype - 1
      this%yme = this%ype
    else
      this%yte = this%ype
      this%yme = this%ype + 1
    end if
    if (this%south == -1) then
      this%yts = this%yps + 1
      this%yms = this%yps
    else
      this%yts = this%yps
      this%yms = this%yps - 1
    end if
    if (this%west == -1) then
      this%xts = this%xps + 1
      this%xms = this%xps
    else
      this%xts = this%xps
      this%xms = this%xps - 1
    end if
    if (this%east == -1) then
      this%xte = this%xpe - 1
      this%xme = this%xpe
    else
      this%xte = this%xpe
      this%xme = this%xpe + 1
    end if

  end function


  !------------------------------------------------------------------
  ! destructor_geometry
  !
  ! Deallocates pointers used by a shallow_water_geometry_type object (none currently)
  !------------------------------------------------------------------
  elemental subroutine destructor_geometry(this)

    type(shallow_water_geometry_type), intent(inout) :: this

    ! No pointers in shallow_water_geometry_type object so we do nothing

  end subroutine
  

  !------------------------------------------------------------------
  ! get_communicator
  !
  ! Returns MPI communicator for this geometry
  !------------------------------------------------------------------
  pure function get_communicator(this) result(communicator)

    class(shallow_water_geometry_type), intent(in) :: this
    integer :: communicator

    communicator = this%mpi_comm

  end function get_communicator


  !------------------------------------------------------------------
  ! get_rank
  !
  ! Returns MPI rank of this process
  !------------------------------------------------------------------
  pure function get_rank(this) result(rank)

    class(shallow_water_geometry_type), intent(in) :: this
    integer :: rank

    rank = this%rank

  end function get_rank


  !------------------------------------------------------------------
  ! get_nranks
  !
  ! Returns the number of MPI ranks
  !------------------------------------------------------------------
  pure function get_nranks(this) result(nranks)

    class(shallow_water_geometry_type), intent(in) :: this
    integer :: nranks

    nranks = this%nranks

  end function get_nranks


  !------------------------------------------------------------------
  ! get_nx
  !
  ! Returns the length of the x dimension
  !------------------------------------------------------------------
  pure function get_nx(this) result(nx)

    class(shallow_water_geometry_type), intent(in) :: this
    integer :: nx

    nx = this%nx

  end function get_nx


  !------------------------------------------------------------------
  ! get_ny
  !
  ! Returns the length of the y dimension
  !------------------------------------------------------------------
  pure function get_ny(this) result(ny)

    class(shallow_water_geometry_type), intent(in) :: this
    integer :: ny

    ny = this%ny

  end function get_ny


  !------------------------------------------------------------------
  ! get_xmax
  !
  ! Returns the extent of the x dimension
  !------------------------------------------------------------------
  pure function get_xmax(this) result(xmax)

    class(shallow_water_geometry_type), intent(in) :: this
    real(r8kind) :: xmax

    xmax = this%xmax

  end function get_xmax


  !------------------------------------------------------------------
  ! get_ymax
  !
  ! Returns the extent of the y dimension
  !------------------------------------------------------------------
  pure function get_ymax(this) result(ymax)

    class(shallow_water_geometry_type), intent(in) :: this
    real(r8kind) :: ymax

    ymax = this%ymax

  end function get_ymax


  !------------------------------------------------------------------
  ! get_dx
  !
  ! Get geometry dx
  !------------------------------------------------------------------
  pure function get_dx(this) result(dx)

    class(shallow_water_geometry_type), intent(in) :: this
    real(r8kind) :: dx

    dx = this%dx

  end function get_dx


  !------------------------------------------------------------------
  ! get_dy
  !
  ! Get geometry dy
  !------------------------------------------------------------------
  pure function get_dy(this) result(dy)

    class(shallow_water_geometry_type), intent(in) :: this
    real(r8kind) :: dy

    dy = this%dy

  end function get_dy


  !------------------------------------------------------------------
  ! get_north
  !
  ! Returns the MPI rank of the northern neighbor (-1 if no neighbor exists)
  !------------------------------------------------------------------
  pure function get_north(this) result(north)

    class(shallow_water_geometry_type), intent(in) :: this
    integer :: north

    north = this%north

  end function get_north


  !------------------------------------------------------------------
  ! get_south
  !
  ! Returns the MPI rank of the southern neighbor (-1 if no neighbor exists)
  !------------------------------------------------------------------
  pure function get_south(this) result(south)

    class(shallow_water_geometry_type), intent(in) :: this
    integer :: south

    south = this%south

  end function get_south


  !------------------------------------------------------------------
  ! get_west
  !
  ! Returns the MPI rank of the western neighbor (-1 if no neighbor exists)
  !------------------------------------------------------------------
  pure function get_west(this) result(west)

    class(shallow_water_geometry_type), intent(in) :: this
    integer :: west

    west = this%west

  end function get_west


  !------------------------------------------------------------------
  ! get_east
  !
  ! Returns the MPI rank of the eastern neighbor (-1 if no neighbor exists)
  !------------------------------------------------------------------
  pure function get_east(this) result(east)

    class(shallow_water_geometry_type), intent(in) :: this
    integer :: east

    east = this%east

  end function get_east


  !------------------------------------------------------------------
  ! get_npx
  !
  ! Returns the size of this patch in the x direction
  !------------------------------------------------------------------
  pure function get_npx(this) result(npx)

    class(shallow_water_geometry_type), intent(in) :: this
    integer :: npx

    npx = this%npx

  end function get_npx


  !------------------------------------------------------------------
  ! get_npy
  !
  ! Returns the size of this patch in the y direction
  !------------------------------------------------------------------
  pure function get_npy(this) result(npy)

    class(shallow_water_geometry_type), intent(in) :: this
    integer :: npy

    npy = this%npy

  end function get_npy


  !------------------------------------------------------------------
  ! get_xps
  !
  ! Returns the starting index of for this patch in the x direction
  !------------------------------------------------------------------
  pure function get_xps(this) result(xps)

    class(shallow_water_geometry_type), intent(in) :: this
    integer :: xps

    xps = this%xps

  end function get_xps


  !------------------------------------------------------------------
  ! get_xpe
  !
  ! Returns the ending index of for this patch in the x direction
  !------------------------------------------------------------------
  pure function get_xpe(this) result(xpe)

    class(shallow_water_geometry_type), intent(in) :: this
    integer :: xpe

    xpe = this%xpe

  end function get_xpe


  !------------------------------------------------------------------
  ! get_yps
  !
  ! Returns the starting index of for this patch in the y direction
  !------------------------------------------------------------------
  pure function get_yps(this) result(yps)

    class(shallow_water_geometry_type), intent(in) :: this
    integer :: yps

    yps = this%yps

  end function get_yps


  !------------------------------------------------------------------
  ! get_ype
  !
  ! Returns the ending index of for this patch in the y direction
  !------------------------------------------------------------------
  pure function get_ype(this) result(ype)

    class(shallow_water_geometry_type), intent(in) :: this
    integer :: ype

    ype = this%ype

  end function get_ype


  !------------------------------------------------------------------
  ! get_xts
  !
  ! Returns the starting index of interior points for this grid patch in the x direction
  !------------------------------------------------------------------
  pure function get_xts(this) result(xts)

    class(shallow_water_geometry_type), intent(in) :: this
    integer :: xts

    xts = this%xts

  end function get_xts


  !------------------------------------------------------------------
  ! get_xte
  !
  ! Returns the ending index of interior points for this grid patch in the x direction
  !------------------------------------------------------------------
  pure function get_xte(this) result(xte)

    class(shallow_water_geometry_type), intent(in) :: this
    integer :: xte

    xte = this%xte

  end function get_xte


  !------------------------------------------------------------------
  ! get_yts
  !
  ! Returns the starting index of interior points for this grid patch in the y direction
  !------------------------------------------------------------------
  pure function get_yts(this) result(yts)

    class(shallow_water_geometry_type), intent(in) :: this
    integer :: yts

    yts = this%yts

  end function get_yts


  !------------------------------------------------------------------
  ! get_yte
  !
  ! Returns the ending index of interior points for this grid patch in the y direction
  !------------------------------------------------------------------
  pure function get_yte(this) result(yte)

    class(shallow_water_geometry_type), intent(in) :: this
    integer :: yte

    yte = this%yte

  end function get_yte


  !------------------------------------------------------------------
  ! get_xms
  !
  ! Returns the starting index of the memory allocated for this grid patch in the x direction
  !------------------------------------------------------------------
  pure function get_xms(this) result(xms)

    class(shallow_water_geometry_type), intent(in) :: this
    integer :: xms

    xms = this%xms

  end function get_xms


  !------------------------------------------------------------------
  ! get_xme
  !
  ! Returns the ending index of the memory allocated for this grid patch in the x direction
  !------------------------------------------------------------------
  pure function get_xme(this) result(xme)

    class(shallow_water_geometry_type), intent(in) :: this
    integer :: xme

    xme = this%xme

  end function get_xme


  !------------------------------------------------------------------
  ! get_yps
  !
  ! Returns the starting index of the memory allocated for this grid patch in the y direction
  !------------------------------------------------------------------
  pure function get_yms(this) result(yms)

    class(shallow_water_geometry_type), intent(in) :: this
    integer :: yms

    yms = this%yms

  end function get_yms


  !------------------------------------------------------------------
  ! get_yme
  !
  ! Returns the ending index of the memory allocated for this grid patch in the y direction
  !------------------------------------------------------------------
  pure function get_yme(this) result(yme)

    class(shallow_water_geometry_type), intent(in) :: this
    integer :: yme

    yme = this%yme

  end function get_yme


end module Shallow_Water_Geometry

