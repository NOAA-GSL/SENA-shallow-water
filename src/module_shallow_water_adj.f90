submodule (Shallow_Water_Model) Shallow_Water_ADJ

implicit none

contains

  !------------------------------------------------------------------
  ! constructor
  !
  ! Returns an initialized shallow_water_adj_type object
  !------------------------------------------------------------------
  module function constructor_adj(config, geometry) result(this)

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
  module subroutine adv_nsteps_adj(this, state, trajectory, nsteps)

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
  module subroutine update_interior_adj(xps, xpe, yps, ype, xts, xte, yts, yte, xms, xme, yms, yme, nx, ny, traj_u, traj_v, traj_h, u, v, h, b, u_new, v_new, h_new, dx, dy, dt)

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
  module subroutine update_boundaries_adj(xps, xpe, yps, ype, xms, xme, yms, yme, north, south, west, east, traj_u, traj_v, traj_h, u, v, h, u_new, v_new, h_new)

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
  elemental module subroutine destructor_adj(this)

    type(shallow_water_adj_type), intent(inout) :: this

    ! No pointers in shallow_water_adj_type object so we do nothing

  end subroutine


end submodule Shallow_Water_ADJ

