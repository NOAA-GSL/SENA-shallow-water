module Shallow_Water_Model

  !Steven McHale
  !Tsunami Model
  !Shallow-Water Wave Equation
  !Crank-Nicholson Discretization

  use Shallow_Water_Kind,          only : r8kind
  use Shallow_Water_Model_Config,  only : shallow_water_model_config_type
  use Shallow_Water_Geometry,      only : shallow_water_geometry_type
  use Shallow_Water_State,         only : shallow_water_state_type
  use mpi

  implicit none

  private

  ! Physical constants
  real(r8kind), parameter :: g  = 9.81_r8kind

  !------------------------------------------------------------------
  ! shallow_water_model_type
  !------------------------------------------------------------------
  public :: shallow_water_model_type

  type :: shallow_water_model_type

    private

    type(shallow_water_model_config_type) :: config   ! Configuration
    type(shallow_water_geometry_type)     :: geometry ! Grid geometry and decomposition
    real(r8kind)                          :: dt       ! Time step
    real(r8kind), allocatable             :: b(:,:)   ! Slope (currently unused)

  contains
      final :: destructor_model
      procedure, private         :: adv_nsteps_model
      generic,   public          :: adv_nsteps => adv_nsteps_model
      procedure, nopass, private :: update_interior_model
      procedure, nopass, private :: update_boundaries_model
      procedure, public          :: get_config
      procedure, public          :: get_geometry
      procedure, public          :: get_dt
  end type shallow_water_model_type

  interface shallow_water_model_type
    procedure constructor_model
  end interface

  !------------------------------------------------------------------
  ! shallow_water_tl_type
  !------------------------------------------------------------------
  public :: shallow_water_tl_type

  type, extends(shallow_water_model_type) :: shallow_water_tl_type

    private

  contains
      final :: destructor_tl
      procedure, private         :: adv_nsteps_tl
      generic,   public          :: adv_nsteps => adv_nsteps_tl
      procedure, nopass, private :: update_interior_tl
      procedure, nopass, private :: update_boundaries_tl
  end type shallow_water_tl_type

  interface shallow_water_tl_type
    procedure constructor_tl
  end interface


  !------------------------------------------------------------------
  ! shallow_water_adj_type
  !------------------------------------------------------------------
  public :: shallow_water_adj_type

  type, extends(shallow_water_model_type) :: shallow_water_adj_type

    private

  contains
      final :: destructor_adj
      procedure, private         :: adv_nsteps_adj
      generic,   public          :: adv_nsteps => adv_nsteps_adj
      procedure, nopass, private :: update_interior_adj
      procedure, nopass, private :: update_boundaries_adj
  end type shallow_water_adj_type

  interface shallow_water_adj_type
    procedure constructor_adj
  end interface


contains


  !------------------------------------------------------------------
  ! constructor
  !
  ! Returns an initialized shallow_water_model_type object
  !------------------------------------------------------------------
  function constructor_model(config, geometry) result(this)

    type(shallow_water_model_config_type), intent(in) :: config
    type(shallow_water_geometry_type),     intent(in) :: geometry

    ! Return a shallow water model object
    type(shallow_water_model_type) :: this

    ! Local variables
    integer      :: i, j
    integer      :: xms, xme, yms, yme

    ! Set the configuration
    this%config = config

    ! Create the geometry
    this%geometry = geometry

    ! Set the time step
    this%dt = this%config%get_dt()

    ! Get the memory allocation index range for this patch from the geometry
    xms = this%geometry%get_xms()
    xme = this%geometry%get_xme()
    yms = this%geometry%get_yms()
    yme = this%geometry%get_yme()

    ! Initialize b (currently unused)
    allocate(this%b(xms:xme, yms:yme))
    do j = yms, yme
      do i = xms, xme
        this%b(i,j) = 0.0_r8kind
      end do
    end do

  end function


  !------------------------------------------------------------------
  ! destructor_model
  !
  ! Deallocates pointers used by a shallow_water_model_type object (none currently)
  !------------------------------------------------------------------
  elemental subroutine destructor_model(this)

    type(shallow_water_model_type), intent(inout) :: this

    ! No pointers in shallow_water_model_type object so we do nothing

  end subroutine

  
  !------------------------------------------------------------------
  ! adv_nsteps_model
  !
  ! Advance model state n steps
  !------------------------------------------------------------------
  subroutine adv_nsteps_model(this, state, nsteps)

    class(shallow_water_model_type), intent(   in) :: this
    type(shallow_water_state_type),  intent(inout) :: state
    integer,                         intent(   in) :: nsteps

    integer                   :: n, i, j
    integer                   :: xps, xpe, yps, ype
    integer                   :: xts, xte, yts, yte
    integer                   :: xms, xme, yms, yme
    integer                   :: north, south, west, east
    real(r8kind)              :: dx, dy, maxdt, local_dt
    real(r8kind), allocatable :: u_new(:,:)
    real(r8kind), allocatable :: v_new(:,:)
    real(r8kind), allocatable :: h_new(:,:)
    real(r8kind), allocatable :: local_b(:,:)
    real(r8kind), allocatable :: local_u(:,:)
    real(r8kind), allocatable :: local_v(:,:)
    real(r8kind), allocatable :: local_h(:,:)

    ! Get grid spacing
    dx = this%geometry%get_dx()
    dy = this%geometry%get_dy()

    ! Sanity check for time step
    if (state%get_max_wavespeed() > 0.0) then
      maxdt = 0.68_r8kind * min(dx, dy) / state%get_max_wavespeed()
      if (this%dt > maxdt) then
        write(*,'(A,F7.2)') "WARNING: time step is too large, should be <= ", maxdt
      end if
    end if

    ! Get local bounds including halo
    xms = this%geometry%get_xms()
    xme = this%geometry%get_xme()
    yms = this%geometry%get_yms()
    yme = this%geometry%get_yme()

    ! Get local bounds exluding the halo
    xps = this%geometry%get_xps()
    xpe = this%geometry%get_xpe()
    yps = this%geometry%get_yps()
    ype = this%geometry%get_ype()

    ! Get local bounds of the interior
    xts = this%geometry%get_xts()
    xte = this%geometry%get_xte()
    yts = this%geometry%get_yts()
    yte = this%geometry%get_yte()

    ! Get MPI ranks of our neighbors
    north = this%geometry%get_north()
    south = this%geometry%get_south()
    west = this%geometry%get_west()
    east = this%geometry%get_east()


    ! Allocate space for new state
    allocate(u_new(xps:xpe, yps:ype))
    allocate(v_new(xps:xpe, yps:ype))
    allocate(h_new(xps:xpe, yps:ype))

    ! Make local copies of u,v,h dt & b for Serialization Tests
    local_u = state%u
    local_v = state%v
    local_h = state%h
    local_dt = this%dt 
    allocate(local_b, SOURCE=this%b)


    ! Move the model state n steps into the future
    do n=1,nsteps

      ! Exchange halos
      ! call state%exchange_halo()

      !*** Serialbox calls for initialization***
      !$ser init directory='./serialbox_data' prefix='update_model' unique_id=.true.
      !$ser mode write
      !$ser on

      !*** Serialbox calls to create a savepoint to save initial starting data***
      !$ser verbatim if (n == 1) then
      !$ser savepoint update_boundaries-IN
      !$ser data xps=xps xpe=xpe yps=yps ype=ype xms=xms xme=xme yms=yms yme=yme
      !$ser data north=north south=south west=west east=east 
      !$ser data local_u=local_u local_v=local_v local_h=local_h
      !$ser data u_new=u_new v_new=v_new h_new=h_new nhalo=xts-xps
      !$ser verbatim endif

      ! Update the domain boundaries
      call this%update_boundaries_model(                          &
                                        xps, xpe, yps, ype,       &
                                        xms, xme, yms, yme,       &
                                        north, south, west, east, &
                                        state%u,                  &
                                        state%v,                  &
                                        state%h,                  &
                                        u_new, v_new, h_new       &
                                       )

      ! $ser verbatim if (n == 11) then
      !$ser savepoint update_boundaries-OUT
      !$ser data u_new_out_boundaries=u_new v_new_out_boundaries=v_new h_new_out_boundaries=h_new
      ! $ser verbatim endif
     
      !$ser verbatim if (n == 1) then
      !$ser savepoint update_interior-IN
      !$ser data xps=xps xpe=xpe yps=yps ype=ype xts=xts xte=xte yts=yts yte=yte 
      !$ser data xms=xms xme=xme yms=yms yme=yme
      !$ser data local_u=local_u local_v=local_v local_h=local_h
      !$ser data local_b=local_b dx=dx dy=dy local_dt=local_dt
      !$ser data u_new=u_new v_new=v_new h_new=h_new 

      ! Serialize indexing variables for ease of creating GT4Py storages
      ! nhalo can be calculated by the starting index of the interior points of this grid patch (xts) minus starting index of for this grid patch (xps)
      !$ser data nhalo=xts-xps  dtdx=local_dt/dx dtdy=local_dt/dy 
      !$ser verbatim endif

      ! Update the domain interior
      call this%update_interior_model(                     &
                                      xps, xpe, yps, ype,  &
                                      xts, xte, yts, yte,  &
                                      xms, xme, yms, yme,  &
                                      local_u,             &
                                      local_v,             &
                                      local_h,             &
                                      local_b,             &
                                      u_new, v_new, h_new, &
                                      dx, dy, local_dt     &
                                     )

      ! $ser verbatim if (n == 11) then
      !$ser savepoint update_interior-OUT
      !$ser data u_new_out_interior=u_new v_new_out_interior=v_new h_new_out_interior=h_new
      ! $ser verbatim endif
      !$ser cleanup 

      ! Update state with new state
      do j = yps, ype
        do i = xps, xpe
          state%u(i,j) = u_new(i,j)
          state%v(i,j) = v_new(i,j)
          state%h(i,j) = h_new(i,j)
        end do
      end do

      ! Update the model clock and step counter
      call state%advance_clock(this%dt)

    end do

  end subroutine adv_nsteps_model


  !------------------------------------------------------------------
  ! update_interior_model
  !
  ! Get model state one step in the future for the domain interior
  !------------------------------------------------------------------
  subroutine update_interior_model(xps, xpe, yps, ype, xts, xte, yts, yte, xms, xme, yms, yme, u, v, h, b, u_new, v_new, h_new, dx, dy, dt)

    integer,      intent(   in) :: xps, xpe, yps, ype
    integer,      intent(   in) :: xts, xte, yts, yte
    integer,      intent(   in) :: xms, xme, yms, yme
    real(r8kind), intent(   in) :: u(xms:xme, yms:yme)
    real(r8kind), intent(   in) :: v(xms:xme, yms:yme)
    real(r8kind), intent(   in) :: h(xms:xme, yms:yme)
    real(r8kind), intent(   in) :: b(xms:xme, yms:yme)
    real(r8kind), intent(inout) :: u_new(xps:xpe, yps:ype)
    real(r8kind), intent(inout) :: v_new(xps:xpe, yps:ype)
    real(r8kind), intent(inout) :: h_new(xps:xpe, yps:ype)
    real(r8kind), intent(   in) :: dx, dy, dt

    integer      :: i, j
    real(r8kind) :: dtdx, dtdy

    ! Pre compute dtdx and dtdy
    dtdx = dt / dx
    dtdy = dt / dy

    ! Employ Lax to parts of this patch that lie in the interior of the domain
    do j=yts, yte
      do i=xts, xte
        u_new(i,j) = ((u(i+1,j) + u(i-1,j) + u(i,j+1) + u(i,j-1)) / 4.0_r8kind)                       &
                     - 0.5_r8kind * dtdx * ((u(i+1,j)**2) / 2.0_r8kind - (u(i-1,j)**2) / 2.0_r8kind)  &
                     - 0.5_r8kind * dtdy * (v(i,j)) * (u(i,j+1) - u(i,j-1))                           &
                     - 0.5_r8kind * g * dtdx * (h(i+1,j) - h(i-1,j))

        v_new(i,j) = ((v(i+1,j) + v(i-1,j) + v(i,j+1) + v(i,j-1)) / 4.0_r8kind)                       &
                     - 0.5_r8kind * dtdy * ((v(i,j+1)**2) / 2.0_r8kind - (v(i,j+1)**2) / 2.0_r8kind)  &
                     - 0.5_r8kind * dtdx * (u(i,j)) * (v(i+1,j) - v(i-1,j))                           &
                     - 0.5_r8kind * g * dtdy * (h(i,j+1) - h(i,j-1))

        h_new(i,j) = ((h(i+1,j) + h(i-1,j) + h(i,j+1) + h(i,j-1)) / 4.0_r8kind)                       &
                     - 0.5_r8kind * dtdx * (u(i,j)) * ((h(i+1,j) - b(i+1,j)) - (h(i-1,j) - b(i-1,j))) &
                     - 0.5_r8kind * dtdy * (v(i,j)) * ((h(i,j+1) - b(i,j+1)) - (h(i,j-1) - b(i,j-1))) &
                     - 0.5_r8kind * dtdx * (h(i,j) - b(i,j)) * (u(i+1,j) - u(i-1,j))                  &
                     - 0.5_r8kind * dtdy * (h(i,j) - b(i,j)) * (v(i,j+1) - v(i,j-1))
      end do
    end do

  end subroutine update_interior_model


  !------------------------------------------------------------------
  ! Update boundaries
  !
  ! Get model state one step in the future for the domain boundaries
  !------------------------------------------------------------------
  subroutine update_boundaries_model(xps, xpe, yps, ype, xms, xme, yms, yme, north, south, west, east, u, v, h, u_new, v_new, h_new)

    integer,      intent(   in) :: xps, xpe, yps, ype
    integer,      intent(   in) :: xms, xme, yms, yme
    integer,      intent(   in) :: north, south, west, east
    real(r8kind), intent(   in) :: u(xms:xme, yms:yme)
    real(r8kind), intent(   in) :: v(xms:xme, yms:yme)
    real(r8kind), intent(   in) :: h(xms:xme, yms:yme)
    real(r8kind), intent(inout) :: u_new(xps:xpe, yps:ype)
    real(r8kind), intent(inout) :: v_new(xps:xpe, yps:ype)
    real(r8kind), intent(inout) :: h_new(xps:xpe, yps:ype)

    integer :: i, j

    ! Update southern boundary if there is one
    if (south == -1) then
      do i = xps, xpe
        h_new(i, yps) =  h(i, yps + 1);
        u_new(i, yps) =  u(i, yps + 1);
        v_new(i, yps) = -v(i, yps + 1);
        ! h_new(i, yps) = 1.;
        ! u_new(i, yps) = 1.;
        ! v_new(i, yps) = 1.;
      end do
    end if

    ! Update northern boundary if there is one
    if (north == -1) then
      do i = xps, xpe
        h_new(i, ype)   =  h(i, ype - 1);
        u_new(i, ype)   =  u(i, ype - 1);
        v_new(i, ype)   = -v(i, ype - 1);
        ! h_new(i, ype)   = 1.;
        ! u_new(i, ype)   = 1.;
        ! v_new(i, ype)   = 1.;
      end do
    end if

    ! Update western boundary if there is one
    if (west == -1) then
      do j = yps, ype
        h_new(xps, j)   =  h(xps + 1, j);
        u_new(xps, j)   = -u(xps + 1, j);
        v_new(xps, j)   =  v(xps + 1, j);
        ! h_new(xps, j)   = 1.;
        ! u_new(xps, j)   = 1.;
        ! v_new(xps, j)   = 1.;
      end do
    end if

    ! Update eastern boundary if there is one
    if (east == -1) then
      do j = yps, ype
        h_new(xpe, j) =  h(xpe - 1, j);
        u_new(xpe, j) = -u(xpe - 1, j);
        v_new(xpe, j) =  v(xpe - 1, j);
        ! h_new(xpe, j) = 1.;
        ! u_new(xpe, j) = 1.;
        ! v_new(xpe, j) = 1.;
      end do
    end if

  end subroutine update_boundaries_model


  !------------------------------------------------------------------
  ! constructor
  !
  ! Returns an initialized shallow_water_tl_type object
  !------------------------------------------------------------------
  function constructor_tl(config, geometry) result(this)

    type(shallow_water_model_config_type), intent(in) :: config
    type(shallow_water_geometry_type),     intent(in) :: geometry

    ! Return a shallow water model object
    type(shallow_water_tl_type) :: this

    ! Initialize the state
    this%shallow_water_model_type = shallow_water_model_type(config, geometry)

  end function


  !------------------------------------------------------------------
  ! destructor_tl
  !
  ! Deallocates pointers used by a shallow_water_tl_type object (none currently)
  !------------------------------------------------------------------
  elemental subroutine destructor_tl(this)

    type(shallow_water_tl_type), intent(inout) :: this

    ! No pointers in shallow_water_tl_type object so we do nothing

  end subroutine


  !------------------------------------------------------------------
  ! adv_nsteps_tl
  !
  ! Advance tl state n steps
  !------------------------------------------------------------------
  subroutine adv_nsteps_tl(this, state, trajectory ,nsteps)

    class(shallow_water_tl_type),   intent(in   ) :: this
    type(shallow_water_state_type), intent(inout) :: state
    type(shallow_water_state_type), intent(inout) :: trajectory
    integer,                        intent(   in) :: nsteps

    integer :: n, i, j
    integer :: xps, xpe, yps, ype
    integer :: xts, xte, yts, yte
    integer :: xms, xme, yms, yme
    integer :: north, south, west, east
    real(r8kind) :: dx, dy, maxdt
    real(r8kind), allocatable :: u_new(:,:)
    real(r8kind), allocatable :: v_new(:,:)
    real(r8kind), allocatable :: h_new(:,:)

    dx = this%geometry%get_dx()
    dy = this%geometry%get_dy()

    ! Sanity check for time step
    if (state%get_max_wavespeed() > 0.0) then
      maxdt = 0.68_r8kind * min(dx, dy) / state%get_max_wavespeed()
      if (this%dt > maxdt) then
        write(*,'(A,F7.2)') "WARNING: time step is too large, should be <= ", maxdt
      end if
    end if

    xps = this%geometry%get_xps()
    xpe = this%geometry%get_xpe()
    yps = this%geometry%get_yps()
    ype = this%geometry%get_ype()

    xts = this%geometry%get_xts()
    xte = this%geometry%get_xte()
    yts = this%geometry%get_yts()
    yte = this%geometry%get_yte()

    xms = this%geometry%get_xms()
    xme = this%geometry%get_xme()
    yms = this%geometry%get_yms()
    yme = this%geometry%get_yme()

    north = this%geometry%get_north()
    south = this%geometry%get_south()
    west = this%geometry%get_west()
    east = this%geometry%get_east()

    allocate(u_new(xps:xpe, yps:ype))
    allocate(v_new(xps:xpe, yps:ype))
    allocate(h_new(xps:xpe, yps:ype))

    do n=1,nsteps

      ! Exchange halos
      call state%exchange_halo()
      call trajectory%exchange_halo()

      ! Update the domain boundaries
      call this%update_boundaries_tl(                         &
                                    xps, xpe, yps, ype,       &
                                    xms, xme, yms, yme,       &
                                    north, south, west, east, &
                                    trajectory%u,             &
                                    trajectory%v,             &
                                    trajectory%h,             &
                                    state%u,                  &
                                    state%v,                  &
                                    state%h,                  &
                                    u_new, v_new, h_new       &
                                   )

      ! Update the domain interior
      call this%update_interior_tl(                           &
                                  xps, xpe, yps, ype,         &
                                  xts, xte, yts, yte,         &
                                  xms, xme, yms, yme,         &
                                  trajectory%u,               &
                                  trajectory%v,               &
                                  trajectory%h,               &
                                  state%u,                    &
                                  state%v,                    &
                                  state%h,                    &
                                  this%b,                     &
                                  u_new, v_new, h_new,        &
                                  dx, dy, this%dt             &
                                 )

      ! Update state with new state
      do j = yps, ype
        do i = xps, xpe
          state%u(i,j) = u_new(i,j)
          state%v(i,j) = v_new(i,j)
          state%h(i,j) = h_new(i,j)
        end do
      end do

    end do

  end subroutine adv_nsteps_tl


  !------------------------------------------------------------------
  ! update_interior_tl
  !
  ! Get tl state one step in the future for the domain interior
  !------------------------------------------------------------------
  subroutine update_interior_tl(xps, xpe, yps, ype, xts, xte, yts, yte, xms, xme, yms, yme, traj_u, traj_v, traj_h, u, v, h, b, u_new, v_new, h_new, dx, dy, dt)

    integer,      intent(   in) :: xps, xpe, yps, ype
    integer,      intent(   in) :: xts, xte, yts, yte
    integer,      intent(   in) :: xms, xme, yms, yme
    real(r8kind), intent(   in) :: traj_u(xms:xme,yms:yme)
    real(r8kind), intent(   in) :: traj_v(xms:xme,yms:yme)
    real(r8kind), intent(   in) :: traj_h(xms:xme,yms:yme)
    real(r8kind), intent(   in) :: u(xms:xme,yms:yme)
    real(r8kind), intent(   in) :: v(xms:xme,yms:yme)
    real(r8kind), intent(   in) :: h(xms:xme,yms:yme)
    real(r8kind), intent(   in) :: b(xms:xme, yms:yme)
    real(r8kind), intent(inout) :: u_new(xps:xpe,yps:ype)
    real(r8kind), intent(inout) :: v_new(xps:xpe,yps:ype)
    real(r8kind), intent(inout) :: h_new(xps:xpe,yps:ype)
    real(r8kind), intent(   in) :: dx, dy, dt

    real(r8kind) :: dtdx, dtdy
    integer      :: i, j

    dtdx = dt/dx
    dtdy = dt/dy

    ! Employ Lax
    do j = yts, yte
      do i = xts, xte
        u_new(i,j) = (u(i+1,j) + u(i-1,j) + u(i,j+1) + u(i,j-1)) / 4.0_r8kind                                              &
                    - 0.5_r8kind * dtdx * (2 * traj_u(i+1,j) * u(i+1,j) / 2.0_r8kind                                       &
                    - 2.0_r8kind * traj_u(i-1,j) * u(i-1, j) / 2.0_r8kind)                                                 &
                    - 0.5_r8kind * dtdy * (v(i,j) * (traj_u(i,j+1) - traj_u(i,j-1)) + traj_v(i,j) * (u(i,j+1) - u(i,j-1))) &
                    - 0.5_r8kind * g * dtdx * (h(i+1,j) - h(i-1,j))

        v_new(i,j) = (v(i+1,j) + v(i-1,j) + v(i,j+1) + v(i,j-1)) / 4.0_r8kind                                              &
                    - 0.5_r8kind * dtdx * (u(i,j) * (traj_v(i+1,j) - traj_v(i-1,j)) + traj_u(i,j) * (v(i+1,j) - v(i-1,j))) &
                    - 0.5_r8kind * g * dtdy * (h(i,j+1) - h(i,j-1))

        h_new(i,j) = (h(i+1,j) + h(i-1,j) + h(i,j+1) + h(i,j-1)) / 4.0_r8kind                                              &
                    - 0.5_r8kind * dtdx * (u(i,j) * (traj_h(i+1,j) - b(i+1,j) - (traj_h(i-1,j)                             &
                    - b(i-1,j))) + traj_u(i,j) * (h(i+1,j) - h(i-1,j)))                                                    &
                    - 0.5_r8kind * dtdy * (v(i,j) * (traj_h(i,j+1) - b(i,j+1) - (traj_h(i,j-1)                             &
                    - b(i,j-1))) + traj_v(i,j) * (h(i,j+1) - h(i,j-1)))                                                    &
                    - 0.5_r8kind * dtdx * (h(i,j) * (traj_u(i+1,j) - traj_u(i-1,j)) + (traj_h(i,j)                         &
                    - b(i,j)) * (u(i+1,j) - u(i-1,j)))                                                                     &
                    - 0.5_r8kind * dtdy * (h(i,j) * (traj_v(i,j+1) - traj_v(i,j-1)) + (traj_h(i,j)                         &
                    - b(i,j)) * (v(i,j+1) - v(i,j-1)))

      end do
    end do

  end subroutine update_interior_tl


  !------------------------------------------------------------------
  ! update_boundaries_tl
  !
  ! Get tl state one step in the future for the domain boundaries
  !------------------------------------------------------------------
  subroutine update_boundaries_tl(xps, xpe, yps, ype, xms, xme, yms, yme, north, south, west, east, traj_u, traj_v, traj_h, u, v, h, u_new, v_new, h_new)

    integer,      intent(   in) :: xps, xpe, yps, ype
    integer,      intent(   in) :: xms, xme, yms, yme
    integer,      intent(   in) :: north, south, west, east
    real(r8kind), intent(   in) :: traj_u(xms:xme, yms:yme)
    real(r8kind), intent(   in) :: traj_v(xms:xme, yms:yme)
    real(r8kind), intent(   in) :: traj_h(xms:xme, yms:yme)
    real(r8kind), intent(   in) :: u(xms:xme,yms:yme)
    real(r8kind), intent(   in) :: v(xms:xme,yms:yme)
    real(r8kind), intent(   in) :: h(xms:xme,yms:yme)
    real(r8kind), intent(inout) :: u_new(xps:xpe,yps:ype)
    real(r8kind), intent(inout) :: v_new(xps:xpe,yps:ype)
    real(r8kind), intent(inout) :: h_new(xps:xpe,yps:ype)

    integer :: i, j

    ! Update southern boundary if there is one
    if (south == -1) then
      do i = xps, xpe
        h_new(i, yps) =  h(i, yps + 1);
        u_new(i, yps) =  u(i, yps + 1);
        v_new(i, yps) = -v(i, yps + 1);
      end do
    end if

    ! Update northern boundary if there is one
    if (north == -1) then
      do i = xps, xpe
        h_new(i, ype)   =  h(i, ype - 1);
        u_new(i, ype)   =  u(i, ype - 1);
        v_new(i, ype)   = -v(i, ype - 1);
      end do
    end if

    ! Update western boundary if there is one
    if (west == -1) then
      do j = yps, ype
        h_new(xps, j)   =  h(xps + 1, j);
        u_new(xps, j)   = -u(xps + 1, j);
        v_new(xps, j)   =  v(xps + 1, j);
      end do
    end if

    ! Update eastern boundary if there is one
    if (east == -1) then
      do j = yps, ype
        h_new(xpe, j) =  h(xpe - 1, j);
        u_new(xpe, j) = -u(xpe - 1, j);
        v_new(xpe, j) =  v(xpe - 1, j);
      end do
    end if

  end subroutine update_boundaries_tl

  !------------------------------------------------------------------
  ! constructor
  !
  ! Returns an initialized shallow_water_adj_type object
  !------------------------------------------------------------------
  function constructor_adj(config, geometry) result(this)

    type(shallow_water_model_config_type), intent(in) :: config
    type(shallow_water_geometry_type),     intent(in) :: geometry

    ! Return a shallow water model object
    type(shallow_water_adj_type) :: this

    ! Initialize the trajectory
    this%shallow_water_model_type = shallow_water_model_type(config, geometry)

  end function


  !------------------------------------------------------------------
  ! adv_nsteps_adj
  !
  ! Advance adj state n steps
  !------------------------------------------------------------------
  subroutine adv_nsteps_adj(this, state, trajectory, nsteps)

    class(shallow_water_adj_type),  intent(   in) :: this
    type(shallow_water_state_type), intent(inout) :: state
    type(shallow_water_state_type), intent(inout) :: trajectory
    integer,                        intent(   in) :: nsteps

    integer                   :: n, i, j
    integer                   :: xps, xpe, yps, ype
    integer                   :: xts, xte, yts, yte
    integer                   :: xms, xme, yms, yme
    integer                   :: nx, ny
    integer                   :: north, south, west, east
    real(r8kind)              :: dx, dy, maxdt
    real(r8kind), allocatable :: u_new(:,:)
    real(r8kind), allocatable :: v_new(:,:)
    real(r8kind), allocatable :: h_new(:,:)

    dx = this%geometry%get_dx()
    dy = this%geometry%get_dy()

    ! Sanity check for time step
    if (state%get_max_wavespeed() > 0.0) then
      maxdt = 0.68_r8kind * min(dx, dy) / state%get_max_wavespeed()
      if (this%dt > maxdt) then
        write(*,'(A,F7.2)') "WARNING: time step is too large, should be <= ", maxdt
      end if
    end if

    xps = this%geometry%get_xps()
    xpe = this%geometry%get_xpe()
    yps = this%geometry%get_yps()
    ype = this%geometry%get_ype()

    xts = this%geometry%get_xts()
    xte = this%geometry%get_xte()
    yts = this%geometry%get_yts()
    yte = this%geometry%get_yte()

    xms = this%geometry%get_xms()
    xme = this%geometry%get_xme()
    yms = this%geometry%get_yms()
    yme = this%geometry%get_yme()

    nx = this%geometry%get_nx()
    ny = this%geometry%get_ny()

    north = this%geometry%get_north()
    south = this%geometry%get_south()
    west = this%geometry%get_west()
    east = this%geometry%get_east()

    allocate(u_new(xms:xme, yms:yme))
    allocate(v_new(xms:xme, yms:yme))
    allocate(h_new(xms:xme, yms:yme))

    do n=1,nsteps

      ! Exchange halos
      call state%exchange_halo()
      call trajectory%exchange_halo()

      ! Adjoint of update state with new state
      h_new(:,:) = state%h(:,:)
      v_new(:,:) = state%v(:,:)
      u_new(:,:) = state%u(:,:)
      state%h(:,:) = 0.0_r8kind
      state%v(:,:) = 0.0_r8kind
      state%u(:,:) = 0.0_r8kind

      ! Update the domain interior
      call this%update_interior_adj(                    &
                                   xps, xpe, yps, ype,  &
                                   xts, xte, yts, yte,  &
                                   xms, xme, yms, yme,  &
                                   nx, ny,              &
                                   trajectory%u,        &
                                   trajectory%v,        &
                                   trajectory%h,        &
                                   state%u,             &
                                   state%v,             &
                                   state%h,             &
                                   this%b,              &
                                   u_new, v_new, h_new, &
                                   dx, dy, this%dt      &
                                  )

      ! Update the domain boundaries
      call this%update_boundaries_adj(                         &
                                     xps, xpe, yps, ype,       &
                                     xms, xme, yms, yme,       &
                                     north, south, west, east, &
                                     trajectory%u,             &
                                     trajectory%v,             &
                                     trajectory%h,             &
                                     state%u,                  &
                                     state%v,                  &
                                     state%h,                  &
                                     u_new, v_new, h_new       &
                                    )

      ! Update the model clock and step counter
      call state%advance_clock(-this%dt)

    end do

  end subroutine adv_nsteps_adj


  !------------------------------------------------------------------
  ! update_interior_adj
  !
  ! Get adj state one step in the future for the domain interior
  !------------------------------------------------------------------
  subroutine update_interior_adj(xps, xpe, yps, ype, xts, xte, yts, yte, xms, xme, yms, yme, nx, ny, traj_u, traj_v, traj_h, u, v, h, b, u_new, v_new, h_new, dx, dy, dt)

    integer,      intent(   in) :: xps, xpe, yps, ype
    integer,      intent(   in) :: xts, xte, yts, yte
    integer,      intent(   in) :: xms, xme, yms, yme
    integer,      intent(   in) :: nx, ny
    real(r8kind), intent(   in) :: traj_u(xms:xme,yms:yme)
    real(r8kind), intent(   in) :: traj_v(xms:xme,yms:yme)
    real(r8kind), intent(   in) :: traj_h(xms:xme,yms:yme)
    real(r8kind), intent(inout) :: u(xms:xme,yms:yme)
    real(r8kind), intent(inout) :: v(xms:xme,yms:yme)
    real(r8kind), intent(inout) :: h(xms:xme,yms:yme)
    real(r8kind), intent(   in) :: b(xms:xme,yms:yme)
    real(r8kind), intent(inout) :: u_new(xms:xme,yms:yme)
    real(r8kind), intent(inout) :: v_new(xms:xme,yms:yme)
    real(r8kind), intent(inout) :: h_new(xms:xme,yms:yme)
    real(r8kind), intent(   in) :: dx, dy, dt

    real(r8kind) :: dtdx, dtdy
    integer      :: i, j

    real(r8kind) :: tempb
    real(r8kind) :: tempb0
    real(r8kind) :: tempb1
    real(r8kind) :: tempb2
    real(r8kind) :: tempb3
    real(r8kind) :: tempb4
    real(r8kind) :: tempb5
    real(r8kind) :: tempb6
    real(r8kind) :: tempb7
    real(r8kind) :: tempb8
    real(r8kind) :: tempb9
    real(r8kind) :: tempb10
    real(r8kind) :: tempb11
    real(r8kind) :: tempb12
    real(r8kind) :: tempb13
    real(r8kind) :: tempb14
    real(r8kind) :: tempb15
    real(r8kind) :: tempb16

    dtdx = dt/dx
    dtdy = dt/dy

    ! Take care of our northern neighbor's southernmost j-1
    if (ype /= ny) then
      j = yte+1
      do i = xte, xts, -1
        tempb = h_new(i, j) / 4.0_r8kind
        tempb2 = -(dtdy * 0.5_r8kind * h_new(i, j))
        tempb3 = traj_v(i, j) * tempb2
        tempb6 = -(dtdy * 0.5_r8kind * h_new(i, j))
        tempb7 = (traj_h(i, j) - b(i, j)) * tempb6
        tempb8 = v_new(i, j) / 4.0_r8kind
        tempb11 = -(g * 0.5_r8kind * dtdy * v_new(i, j))
        tempb12 = u_new(i, j) / 4.0_r8kind
        tempb14 = -(dtdy * 0.5_r8kind * u_new(i, j))
        tempb15 = traj_v(i, j) * tempb14

        h_new(i, j) = 0.0_r8kind
        v_new(i, j) = 0.0_r8kind
        u_new(i, j) = 0.0_r8kind

        u(i, j-1) = u(i, j-1) + tempb12 - tempb15
        v(i, j-1) = v(i, j-1) - tempb7 + tempb8
        h(i, j-1) = h(i, j-1) + tempb - tempb3 - tempb11
      end do
    end if

    ! Take care of our interior j
    do j = yte, yts, -1

      ! Take care of our eastern neighbor's westernmost i-1
      if (xpe /= nx) then
        i = xte+1
        tempb = h_new(i, j) / 4.0_r8kind
        tempb0 = -(dtdx * 0.5_r8kind * h_new(i, j))
        tempb1 = traj_u(i, j) * tempb0
        tempb4 = -(dtdx * 0.5_r8kind * h_new(i, j))
        tempb5 = (traj_h(i, j) - b(i, j)) * tempb4
        tempb8 = v_new(i, j) / 4.0_r8kind
        tempb9 = -(dtdx * 0.5_r8kind * v_new(i, j))
        tempb10 = traj_u(i, j) * tempb9
        tempb12 = u_new(i, j) / 4.0_r8kind
        tempb13 = -(dtdx * 0.5_r8kind * u_new(i, j))
        tempb16 = -(g * 0.5_r8kind * dtdx * u_new(i, j))

        h_new(i, j) = 0.0_r8kind
        v_new(i, j) = 0.0_r8kind
        u_new(i, j) = 0.0_r8kind

        u(i-1, j) = u(i-1, j) - tempb5 + tempb12 - 2.0_r8kind * traj_u(i-1, j) * tempb13 / 2.0_r8kind
        v(i-1, j) = v(i-1, j) + tempb8 - tempb10
        h(i-1, j) = h(i-1, j) + tempb - tempb1 - tempb16
      end if

      ! Take care of our interior i
      do i = xte, xts, -1
        tempb = h_new(i, j) / 4.0_r8kind
        tempb0 = -(dtdx * 0.5_r8kind * h_new(i, j))
        tempb1 = traj_u(i, j) * tempb0
        tempb2 = -(dtdy * 0.5_r8kind * h_new(i, j))
        tempb3 = traj_v(i, j) * tempb2
        tempb4 = -(dtdx * 0.5_r8kind * h_new(i, j))
        tempb5 = (traj_h(i, j) - b(i, j)) * tempb4
        tempb6 = -(dtdy * 0.5_r8kind * h_new(i, j))
        tempb7 = (traj_h(i, j) - b(i, j)) * tempb6
        tempb8 = v_new(i, j) / 4.0_r8kind
        tempb9 = -(dtdx * 0.5_r8kind * v_new(i, j))
        tempb10 = traj_u(i, j) * tempb9
        tempb11 = -(g * 0.5_r8kind * dtdy * v_new(i, j))
        tempb12 = u_new(i, j) / 4.0_r8kind
        tempb13 = -(dtdx * 0.5_r8kind * u_new(i, j))
        tempb14 = -(dtdy * 0.5_r8kind * u_new(i, j))
        tempb15 = traj_v(i, j) * tempb14
        tempb16 = -(g * 0.5_r8kind * dtdx * u_new(i, j))

        h_new(i, j) = 0.0_r8kind
        v_new(i, j) = 0.0_r8kind
        u_new(i, j) = 0.0_r8kind

        u(i-1, j) = u(i-1, j) - tempb5 + tempb12 - 2.0_r8kind * traj_u(i-1, j) * tempb13 / 2.0_r8kind
        v(i-1, j) = v(i-1, j) + tempb8 - tempb10
        h(i-1, j) = h(i-1, j) + tempb - tempb1 - tempb16

        u(i, j) = u(i, j) + (b(i-1, j) - b(i+1, j) + traj_h(i+1, j) - traj_h(i-1, j)) * tempb0 + (traj_v(i+1, j) - traj_v(i-1, j)) * tempb9
        v(i, j) = v(i, j) + (b(i, j-1) - b(i, j+1) + traj_h(i, j+1) - traj_h(i, j-1)) * tempb2 + (traj_u(i, j+1) - traj_u(i, j-1)) * tempb14
        h(i, j) = h(i, j) + (traj_v(i, j+1) - traj_v(i, j-1)) * tempb6 + (traj_u(i+1, j) - traj_u(i-1, j)) * tempb4

        u(i+1, j) = u(i+1, j) + tempb5 + 2.0_r8kind * traj_u(i+1, j) * tempb13 / 2.0_r8kind + tempb12
        v(i+1, j) = v(i+1, j) + tempb10 + tempb8
        h(i+1, j) = h(i+1, j) + tempb1 + tempb + tempb16

        u(i, j-1) = u(i, j-1) + tempb12 - tempb15
        v(i, j-1) = v(i, j-1) - tempb7 + tempb8
        h(i, j-1) = h(i, j-1) + tempb - tempb3 - tempb11

        u(i, j+1) = u(i, j+1) + tempb15 + tempb12
        v(i, j+1) = v(i, j+1) + tempb7 + tempb8
        h(i, j+1) = h(i, j+1) + tempb3 + tempb + tempb11
      end do  ! Our interior i

      ! Take care of our western neighbor's easternmost i+1
      if (xps /= 1) then
        i = xts - 1
        tempb = h_new(i, j) / 4.0_r8kind
        tempb0 = -(dtdx * 0.5_r8kind * h_new(i, j))
        tempb1 = traj_u(i, j) * tempb0
        tempb4 = -(dtdx * 0.5_r8kind * h_new(i, j))
        tempb5 = (traj_h(i, j) - b(i, j)) * tempb4
        tempb8 = v_new(i, j) / 4.0_r8kind
        tempb9 = -(dtdx * 0.5_r8kind * v_new(i, j))
        tempb10 = traj_u(i, j) * tempb9
        tempb12 = u_new(i, j) / 4.0_r8kind
        tempb13 = -(dtdx * 0.5_r8kind * u_new(i, j))
        tempb16 = -(g * 0.5_r8kind * dtdx * u_new(i, j))

        h_new(i, j) = 0.0_r8kind
        v_new(i, j) = 0.0_r8kind
        u_new(i, j) = 0.0_r8kind

        u(i+1, j) = u(i+1, j) + tempb5 + 2.0_r8kind * traj_u(i+1, j) * tempb13 / 2.0_r8kind + tempb12
        v(i+1, j) = v(i+1, j) + tempb10 + tempb8
        h(i+1, j) = h(i+1, j) + tempb1 + tempb + tempb16
      end if

    end do  ! Our interior j

    ! Take care of our sourthern neighbor's northernmost j+1
    if (yps /= 1) then
      j = yts-1
      do i = xte, xts, -1
        tempb = h_new(i, j) / 4.0_r8kind
        tempb2 = -(dtdy * 0.5_r8kind * h_new(i, j))
        tempb3 = traj_v(i, j) * tempb2
        tempb6 = -(dtdy * 0.5_r8kind * h_new(i, j))
        tempb7 = (traj_h(i, j) - b(i, j)) * tempb6
        tempb8 = v_new(i, j) / 4.0_r8kind
        tempb11 = -(g * 0.5_r8kind * dtdy * v_new(i, j))
        tempb12 = u_new(i, j) / 4.0_r8kind
        tempb14 = -(dtdy * 0.5_r8kind * u_new(i, j))
        tempb15 = traj_v(i, j) * tempb14

        h_new(i, j) = 0.0_r8kind
        v_new(i, j) = 0.0_r8kind
        u_new(i, j) = 0.0_r8kind

        u(i, j+1) = u(i, j+1) + tempb15 + tempb12
        v(i, j+1) = v(i, j+1) + tempb7 + tempb8
        h(i, j+1) = h(i, j+1) + tempb3 + tempb + tempb11
      end do
    end if

  end subroutine update_interior_adj


  !------------------------------------------------------------------
  ! Update boundaries_adj
  !
  ! Advance adjoint state one step in the future for the domain boundaries
  !------------------------------------------------------------------
  subroutine update_boundaries_adj(xps, xpe, yps, ype, xms, xme, yms, yme, north, south, west, east, traj_u, traj_v, traj_h, u, v, h, u_new, v_new, h_new)

    integer,      intent(   in) :: xps, xpe, yps, ype
    integer,      intent(   in) :: xms, xme, yms, yme
    integer,      intent(   in) :: north, south, west, east
    real(r8kind), intent(   in) :: traj_u(xms:xme, yms:yme)
    real(r8kind), intent(   in) :: traj_v(xms:xme, yms:yme)
    real(r8kind), intent(   in) :: traj_h(xms:xme, yms:yme)
    real(r8kind), intent(inout) :: u(xms:xme,yms:yme)
    real(r8kind), intent(inout) :: v(xms:xme,yms:yme)
    real(r8kind), intent(inout) :: h(xms:xme,yms:yme)
    real(r8kind), intent(inout) :: u_new(xms:xme,yms:yme)
    real(r8kind), intent(inout) :: v_new(xms:xme,yms:yme)
    real(r8kind), intent(inout) :: h_new(xms:xme,yms:yme)

    integer :: i, j

    ! Update eastern boundary if there is one
    if (east == -1) then
      do j = yps, ype
        u(xpe-1, j) = u(xpe-1, j) - u_new(xpe, j)
        v(xpe-1, j) = v(xpe-1, j) + v_new(xpe, j)
        h(xpe-1, j) = h(xpe-1, j) + h_new(xpe, j)
        u_new(xpe, j) = 0.0_r8kind
        v_new(xpe, j) = 0.0_r8kind
        h_new(xpe, j) = 0.0_r8kind
      end do
    end if

    ! Update western boundary if there is one
    if (west == -1) then
      do j = yps, ype
        u(xps+1, j) = u(xps+1, j) - u_new(xps, j)
        v(xps+1, j) = v(xps+1, j) + v_new(xps, j)
        h(xps+1, j) = h(xps+1, j) + h_new(xps, j)
        u_new(xps, j) = 0.0_r8kind
        v_new(xps, j) = 0.0_r8kind
        h_new(xps, j) = 0.0_r8kind
      end do
    end if

    ! Update northern boundary if there is one
    if (north == -1) then
      do i = xps, xpe
        u(i, ype-1) = u(i, ype-1) + u_new(i, ype)
        v(i, ype-1) = v(i, ype-1) - v_new(i, ype)
        h(i, ype-1) = h(i, ype-1) + h_new(i, ype)
        v_new(i, ype) = 0.0_r8kind
        u_new(i, ype) = 0.0_r8kind
        h_new(i, ype) = 0.0_r8kind
      end do
    end if

    ! Update southern boundary if there is one
    if (south == -1) then
      do i = xps, xpe
        u(i, yps+1) = u(i, yps+1) + u_new(i, yps)
        v(i, yps+1) = v(i, yps+1) - v_new(i, yps)
        h(i, yps+1) = h(i, yps+1) + h_new(i, yps)
      end do
    end if

  end subroutine update_boundaries_adj


  !------------------------------------------------------------------
  ! destructor_adj
  !
  ! Deallocates pointers used by a shallow_water_adj_type object (none currently)
  !------------------------------------------------------------------
  elemental subroutine destructor_adj(this)

    type(shallow_water_adj_type), intent(inout) :: this

    ! No pointers in shallow_water_adj_type object so we do nothing

  end subroutine


  !------------------------------------------------------------------
  ! get_config
  !
  ! Get model configuration
  !------------------------------------------------------------------
  pure function get_config(this) result(config)

    class(shallow_water_model_type), intent(in) :: this
    type(shallow_water_model_config_type) :: config

    config = this%config

  end function get_config


  !------------------------------------------------------------------
  ! get_geometry
  !
  ! Get model geometry
  !------------------------------------------------------------------
  pure function get_geometry(this) result(geometry)

    class(shallow_water_model_type), intent(in) :: this
    type(shallow_water_geometry_type) :: geometry

    geometry = this%geometry

  end function get_geometry


  !------------------------------------------------------------------
  ! get_dt
  !
  ! Get model dt
  !------------------------------------------------------------------
  pure function get_dt(this) result(dt)

    class(shallow_water_model_type), intent(in) :: this
    real(r8kind) :: dt

    dt = this%dt

  end function get_dt


end module Shallow_Water_Model

