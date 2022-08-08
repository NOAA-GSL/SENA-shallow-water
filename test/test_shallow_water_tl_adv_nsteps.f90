program Test_Shallow_Water_TL_Adv_Nsteps

  use Shallow_Water_Kind,            only : r8kind
  use Shallow_Water_Geometry_Config, only : shallow_water_geometry_config_type
  use Shallow_Water_Model_Config,    only : shallow_water_model_config_type
  use Shallow_Water_Geometry,        only : shallow_water_geometry_type
  use Shallow_Water_State,           only : shallow_water_state_type
  use Shallow_Water_Model,           only : shallow_water_model_type, shallow_water_tl_type
  use mpi

  implicit none

  ! Config parameters
  integer, parameter :: nx = 101
  integer, parameter :: ny = 101
  real(r8kind), parameter :: xmax = 100000.0
  real(r8kind), parameter :: ymax = 100000.0
  real(r8kind), parameter :: u0 = 0.0
  real(r8kind), parameter :: v0 = 0.0
  real(r8kind), parameter :: b0 = 0.0
  real(r8kind), parameter :: h0 = 5030.0
  real(r8kind), parameter :: g = 9.81_r8kind
  real(r8kind), parameter :: dt = 0.8_r8kind

  ! Test parameters
  integer, parameter :: spinup_steps = 1000 ! model spinup steps
  integer, parameter :: digits = 8          ! Number of significant digits to check

  type(shallow_water_geometry_config_type) :: geometry_config
  type(shallow_water_model_config_type)    :: model_config
  type(shallow_water_geometry_type)        :: geometry
  type(shallow_water_state_type)           :: state, state_delta, state_tl, trajectory
  type(shallow_water_model_type)           :: shallow_water, shallow_water_delta
  type(shallow_water_tl_type)              :: shallow_water_tl

  real(r8kind)              :: dx, dy, xmid, ymid, dsqr, sigma
  real(r8kind), allocatable :: u(:,:), v(:,:), h(:,:)
  real(r8kind), allocatable :: udelta(:,:), vdelta(:,:), hdelta(:,:)
  real(r8kind), allocatable :: mu(:,:), mv(:,:), mh(:,:)
  real(r8kind), allocatable :: m_udelta(:,:), m_vdelta(:,:), m_hdelta(:,:)
  real(r8kind), allocatable :: mprime_udelta(:,:), mprime_vdelta(:,:), mprime_hdelta(:,:)
  real(r8kind)              :: uratio(digits), vratio(digits), hratio(digits)

  real(r8kind) :: lambda              ! scaling factor
  integer      :: d, n, i, j          ! Loop index
  integer      :: xps, xpe, yps, ype  ! Local domain index ranges

  ! MPI variables
  integer :: myrank, nranks, ierr, errors

  ! Start MPI
  call MPI_Init(ierr)

  ! Get our MPI rank
  call MPI_Comm_rank(MPI_COMM_WORLD, myrank, ierr)

  ! Get number of MPI ranks
  call MPI_Comm_size(MPI_COMM_WORLD, nranks, ierr)

  ! Initialize lambda
  lambda = 1.0_r8kind

  ! Create a geometry configuration as specified by the namelist
  geometry_config = shallow_water_geometry_config_type(nx, ny, xmax, ymax)

  ! Create a geometry from configuration
  geometry = shallow_water_geometry_type(geometry_config, MPI_COMM_WORLD)

  ! Get index ranges from geometry
  xps = geometry%get_xps()
  xpe = geometry%get_xpe()
  yps = geometry%get_yps()
  ype = geometry%get_ype()

  ! Allocate space for model states
  allocate(u(xps:xpe, yps:ype), v(xps:xpe, yps:ype), h(xps:xpe, yps:ype))
  allocate(udelta(xps:xpe, yps:ype), vdelta(xps:xpe, yps:ype), hdelta(xps:xpe, yps:ype))
  allocate(mu(xps:xpe, yps:ype), mv(xps:xpe, yps:ype), mh(xps:xpe, yps:ype))
  allocate(m_udelta(xps:xpe, yps:ype), m_vdelta(xps:xpe, yps:ype), m_hdelta(xps:xpe, yps:ype))
  allocate(mprime_udelta(xps:xpe, yps:ype), mprime_vdelta(xps:xpe, yps:ype), mprime_hdelta(xps:xpe, yps:ype))

  ! Create a state with a tsunami pulse in it to initialize field h
  dx = geometry%get_dx()
  dy = geometry%get_dy()
  xmid = geometry%get_xmax() / 2.0_r8kind
  ymid = geometry%get_ymax() / 2.0_r8kind
  sigma = floor(geometry%get_xmax() / 20.0_r8kind)
  do j = yps, ype
    do i = xps, xpe
      dsqr = (dble(i - 1) * dx - xmid)**2 + (dble(j - 1) * dy - ymid)**2
      h(i,j) = 5000.0_r8kind + exp(-dsqr / sigma**2) * (h0 - 5000.0_r8kind)
    end do
  end do
  state = shallow_water_state_type(geometry, h=h)

  ! Create a model configuration as specified by the namelist
  model_config = shallow_water_model_config_type(dt, u0, v0, b0, h0)

  ! Create a model with the namelist configuration
  shallow_water = shallow_water_model_type(model_config, geometry)

  ! Create a tangent linear model with the namelist configuration
  shallow_water_tl = shallow_water_tl_type(model_config, geometry)

  ! Spinup the forward model to avoid initial condition issues
  write(*,*) 'Integrating forward model spinup steps: ', spinup_steps
  call shallow_water%adv_nsteps(state, spinup_steps)

  ! Advance the forward model 100 steps and use it to compute a dx
  u = state%get_u()
  v = state%get_v()
  h = state%get_h()
  call shallow_water%adv_nsteps(state, 100)
  udelta = (state%get_u() - u)
  vdelta = (state%get_v() - v)
  hdelta = (state%get_h() - h)
  udelta = udelta * 100000.0
  vdelta = vdelta * 100000.0
  hdelta = hdelta * 100000.0

  ! Loop over digits of precision to calculate metric ratio
  do d = 1, digits

    ! Create a shallow_water state for M(x)
    state = shallow_water_state_type(geometry, u=u, v=v, h=h)

    ! Create a shallow_water state for M(x + lambda * dx)
    state_delta = shallow_water_state_type(geometry, u=u + lambda * udelta, v=v + lambda * vdelta, h=h + lambda * hdelta)

    ! Create a shallow_water_tl state and trajectory for M'(lambda * dx)
    trajectory = shallow_water_state_type(geometry, u=u, v=v, h=h)
    state_tl = shallow_water_state_type(geometry, u=lambda * udelta, v=lambda * vdelta, h=lambda * hdelta)

    ! Advance shallow_water, shallow_water_delta, and shallow_water_tl
    call shallow_water%adv_nsteps(state, 1)
    call shallow_water%adv_nsteps(state_delta, 1)
    call shallow_water_tl%adv_nsteps(state_tl, trajectory, 1)

    ! Calculate and print test metric
    mu = state%get_u()
    mv = state%get_v()
    mh = state%get_h()

    m_udelta = state_delta%get_u()
    m_vdelta = state_delta%get_v()
    m_hdelta = state_delta%get_h()

    mprime_udelta = state_tl%get_u()
    mprime_vdelta = state_tl%get_v()
    mprime_hdelta = state_tl%get_h()

    uratio(d) = (m_udelta(xps+(xpe-xps)/3,yps+(ype-yps)/3) - mu(xps+(xpe-xps)/3,yps+(ype-yps)/3)) / mprime_udelta(xps+(xpe-xps)/3,yps+(ype-yps)/3)
    vratio(d) = (m_vdelta(xps+(xpe-xps)/3,yps+(ype-yps)/3) - mv(xps+(xpe-xps)/3,yps+(ype-yps)/3)) / mprime_vdelta(xps+(xpe-xps)/3,yps+(ype-yps)/3)
    hratio(d) = (m_hdelta(xps+(xpe-xps)/3,yps+(ype-yps)/3) - mh(xps+(xpe-xps)/3,yps+(ype-yps)/3)) / mprime_hdelta(xps+(xpe-xps)/3,yps+(ype-yps)/3)

    ! Increase precision
    lambda = lambda / 10.0_r8kind

  end do

  ! Loop over each MPI rank to check for proper increase in precision of ratio for decrease in lambda
  do n = 0, nranks - 1

    if (myrank == n) then

      ! Write dx info
      write(*,*)
      write(*,'(2A15,2F18.10)') 'min udelta', 'max udelta', minval(udelta), maxval(udelta)
      write(*,'(2A15,2F18.10)') 'min vdelta', 'max vdelta', minval(vdelta), maxval(vdelta)
      write(*,'(2A15,2F18.10)') 'min hdelta', 'max hdelta', minval(hdelta), maxval(hdelta)
      write(*,*)

      ! Write column headers
      write(*,'(A13,11x,A)') "Lambda", "( M(x + lambda * dx) - M(x) ) / M'(lambda * dx)"
      write(*,'(A13,3A18)') "", "U", "V", "H"

      lambda = 1.0_r8kind
      errors = 0
      do d = 1, digits

        write(*,'(4F18.12)') lambda, uratio(d), vratio(d), hratio(d)

        if (d > 1) then

          ! Check precision of ratios
          if (abs(uratio(d) - 1.0_r8kind) > abs(uratio(d-1) - 1.0_r8kind)) then
            write(*, '(A)') "ERROR: Precision of u ratio not decreasing as lambda decreases.  "
            errors = errors + 1
          end if
          if (abs(vratio(d) - 1.0_r8kind) > abs(vratio(d-1) - 1.0_r8kind)) then
            write(*, '(A)') "ERROR: Precision of v ratio not decreasing as lambda decreases.  "
            errors = errors + 1
          end if
          if (abs(hratio(d) - 1.0_r8kind) > abs(hratio(d-1) - 1.0_r8kind)) then
            write(*, '(A)') "ERROR: Precision of h ratio not decreasing as lambda decreases.  "
            errors = errors + 1
          end if

        end if

        ! Increase precision
        lambda = lambda / 10.0_r8kind

      end do

    end if

    call MPI_Barrier(MPI_COMM_WORLD, ierr)

  end do

  if (errors > 0) then
    call MPI_Abort(MPI_COMM_WORLD, errors, ierr)
  end if

  ! Shut down MPI
  call MPI_Finalize(ierr)

end program Test_Shallow_Water_TL_Adv_Nsteps
