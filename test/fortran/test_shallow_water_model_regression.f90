program Test_Shallow_Water_Model_Regression

  use Shallow_Water_Kind,            only : r8kind
  use Shallow_Water_Geometry_Config, only : shallow_water_geometry_config_type
  use Shallow_Water_Model_Config,    only : shallow_water_model_config_type
  use Shallow_Water_Geometry,        only : shallow_water_geometry_type
  use Shallow_water_State,           only : shallow_water_state_type
  use Shallow_Water_Model,           only : shallow_water_model_type
  use Test_Utilities,                only : check_real_scalar
  use mpi

  implicit none

  ! Shallow water config object
  type(shallow_water_geometry_config_type) :: geometry_config
  type(shallow_water_model_config_type)    :: model_config
  type(shallow_water_geometry_type)        :: geometry
  type(shallow_water_state_type)           :: state
  type(shallow_water_model_type)           :: model

  ! Config parameters
  integer, parameter :: nx = 11
  integer, parameter :: ny = 11
  real(r8kind), parameter :: xmax = 10000.0
  real(r8kind), parameter :: ymax = 10000.0
  real(r8kind), parameter :: u0 = 0.0
  real(r8kind), parameter :: v0 = 0.0
  real(r8kind), parameter :: b0 = 0.0
  real(r8kind), parameter :: h0 = 5030.0
  real(r8kind), parameter :: g = 9.81_r8kind
  real(r8kind), parameter :: dt = 0.68_r8kind * (xmax / (dble(nx) - 1.0)) / (u0 + sqrt(g * (h0 - b0)))

  ! Initial Tsunami pulse
  real(r8kind), allocatable :: h(:,:)
  integer                   :: i, j
  real(r8kind)              :: dx, dy, xmid, ymid, dsqr, sigma

  ! Test variables
  real(r8kind), allocatable :: u_full(:,:), v_full(:,:), h_full(:,:)
  real(r8kind)              :: u_rms, v_rms, h_rms
  real(r8kind), parameter   :: u_rms_baseline = 0.00161019683016338_r8kind
  real(r8kind), parameter   :: v_rms_baseline = 0.00161183246804103_r8kind
  real(r8kind), parameter   :: h_rms_baseline = 5000.37196249264_r8kind
  integer                   :: errors

  ! MPI variables
  integer :: ierr

  ! Start MPI
  call MPI_Init(ierr)
  
  ! Initialize Serialbox
  !$ser init directory='./serialbox_data' prefix='test_shallow_water_model_regression'

  ! Initialize error count to 0
  errors = 0

  ! Create a shallow water geometry configuration
  geometry_config = shallow_water_geometry_config_type(nx, ny, xmax, ymax)

  ! Create a shallow water geometry from the configurate
  geometry = shallow_water_geometry_type(geometry_config, MPI_COMM_WORLD)

  ! Create a state with a tsunami pulse in it to initialize field h
  allocate(h(geometry%get_xps():geometry%get_xpe(), geometry%get_yps():geometry%get_ype()))
  dx = geometry%get_dx()
  dy = geometry%get_dy()
  xmid = geometry%get_xmax() / 2.0_r8kind
  ymid = geometry%get_ymax() / 2.0_r8kind
  sigma = floor(geometry%get_xmax() / 20.0_r8kind)
  do j = geometry%get_yps(), geometry%get_ype()
    do i = geometry%get_xps(), geometry%get_xpe()
      dsqr = (dble(i - 1) * dx - xmid)**2 + (dble(j - 1) * dy - ymid)**2
      h(i,j) = 5000.0_r8kind + exp(-dsqr / sigma**2) * (h0 - 5000.0_r8kind)
    end do
  end do
  state = shallow_water_state_type(geometry, h=h)

  ! Create a shallow water model configuration
  model_config = shallow_water_model_config_type(dt, u0, v0, b0, h0)

  ! Initialize shallow water model
  model = shallow_water_model_type(model_config, geometry)

  ! Advance the model 1 step
  call model%adv_nsteps(state, 100)

  ! Gather state to calculate RMS
  if (geometry%get_rank() == 0) then
    allocate(u_full(geometry%get_nx(),geometry%get_ny()))
    allocate(v_full(geometry%get_nx(),geometry%get_ny()))
    allocate(h_full(geometry%get_nx(),geometry%get_ny()))
  end if
  call state%gather(u_full, v_full, h_full)

  ! Compute rms
  if (geometry%get_rank() == 0) then
    u_full = u_full * u_full
    u_rms = sqrt(sum(u_full) / (geometry%get_nx() * geometry%get_ny()))
    v_full = v_full * v_full
    v_rms = sqrt(sum(v_full) / (geometry%get_nx() * geometry%get_ny()))
    h_full = h_full * h_full
    h_rms = sqrt(sum(h_full) / (geometry%get_nx() * geometry%get_ny()))

    ! Check rms values against baseline
    call check_real_scalar(u_rms, "u_rms", u_rms_baseline, 10E-12_r8kind, errors)
    call check_real_scalar(v_rms, "v_rms", v_rms_baseline, 10E-12_r8kind, errors)
    call check_real_scalar(h_rms, "h_rms", h_rms_baseline, 10E-12_r8kind, errors)

    if (errors > 0) then
      call MPI_Abort(MPI_COMM_WORLD, errors, ierr)
    end if

  end if

  ! Clean up Serialbox
  !$ser cleanup

  ! Shut down MPI
  call MPI_Finalize(ierr)
  
end program Test_Shallow_Water_Model_Regression
