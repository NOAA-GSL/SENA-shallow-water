submodule (Shallow_Water_Model) Shallow_Water_TL

implicit none

contains

  !------------------------------------------------------------------
  ! constructor
  !
  ! Returns an initialized shallow_water_tl_type object
  !------------------------------------------------------------------
  module function constructor_tl(config, geometry) result(this)

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
  elemental module subroutine destructor_tl(this)

    type(shallow_water_tl_type), intent(inout) :: this

    ! No pointers in shallow_water_tl_type object so we do nothing

  end subroutine


  !------------------------------------------------------------------
  ! adv_nsteps_tl
  !
  ! Advance tl state n steps
  !------------------------------------------------------------------
  module subroutine adv_nsteps_tl(this, state, trajectory ,nsteps)

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
  module subroutine update_interior_tl(xps, xpe, yps, ype, xts, xte, yts, yte, xms, xme, yms, yme, traj_u, traj_v, traj_h, u, v, h, b, u_new, v_new, h_new, dx, dy, dt)

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
  module subroutine update_boundaries_tl(xps, xpe, yps, ype, xms, xme, yms, yme, north, south, west, east, traj_u, traj_v, traj_h, u, v, h, u_new, v_new, h_new)

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


end submodule Shallow_Water_TL

