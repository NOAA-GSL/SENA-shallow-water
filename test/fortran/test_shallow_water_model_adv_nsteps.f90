program Test_Shallow_Water_Model_Adv_Nsteps

  use Shallow_Water_Kind,            only : r8kind
  use Shallow_Water_Geometry_Config, only : shallow_water_geometry_config_type
  use Shallow_Water_Model_Config,    only : shallow_water_model_config_type
  use Shallow_Water_Geometry,        only : shallow_water_geometry_type
  use Shallow_Water_State,           only : shallow_water_state_type
  use Shallow_Water_Model,           only : shallow_water_model_type
  use Test_Utilities,                only : check_integer_scalar, check_real_scalar, check_min_max_real2d
  use mpi

  implicit none

  ! Shallow water config object
  type(shallow_water_geometry_config_type) :: geometry_config
  type(shallow_water_model_config_type)    :: model_config
  type(shallow_water_geometry_type)        :: geometry
  type(shallow_water_state_type)           :: state
  type(shallow_water_model_type)           :: model

  ! Config parameters
  integer, parameter :: step = 1000
  integer, parameter :: nx = 101
  integer, parameter :: ny = 101
  real(r8kind), parameter :: xmax = 100000.0
  real(r8kind), parameter :: ymax = 100000.0
  real(r8kind), parameter :: u0 = 0.0
  real(r8kind), parameter :: v0 = 0.0
  real(r8kind), parameter :: b0 = 0.0
  real(r8kind), parameter :: h0 = 10.06
  real(r8kind), parameter :: g = 9.81_r8kind
  real(r8kind), parameter :: dt = 0.68_r8kind * (xmax / (dble(nx) - 1.0)) / (u0 + sqrt(g * (h0 - b0)))

  ! Model variables
  integer                   :: i, j
  integer                   :: xps, xpe, yps, ype
  real(r8kind), allocatable :: u(:,:), v(:,:), h(:,:)

  ! Test variables
  integer :: ierr
  integer :: errors

  call MPI_Init(ierr)

  ! Initialize Serialbox
  !$ser init directory='./serialbox_data' prefix='test_shallow_water_model_adv_nsteps'

  ! Initialize error count to 0
  errors = 0

  ! Create a shallow water geometry configuration
  geometry_config = shallow_water_geometry_config_type(nx, ny, xmax, ymax)

  ! Create a shallow water geometry
  geometry = shallow_water_geometry_type(geometry_config, MPI_COMM_WORLD)

  ! Get index ranges from geometry
  xps = geometry%get_xps()
  xpe = geometry%get_xpe()
  yps = geometry%get_yps()
  ype = geometry%get_ype()

  ! Allocate u, v, h
  allocate(u(xps:xpe, yps:ype))
  allocate(v(xps:xpe, yps:ype))
  allocate(h(xps:xpe, yps:ype))

  ! Initialize u, v, h
  do j = yps, ype
    do i = xps, xpe
      u(i, j) = 0.0
      v(i, j) = 0.0
      h(i, j) = 0.0
    end do
  end do

  ! Create a shallow water model configuration
  model_config = shallow_water_model_config_type(dt, u0, v0, b0, h0)

  ! Initialize shallow water model
  model = shallow_water_model_type(model_config, geometry)

  ! Create a shallow water state
  state = shallow_water_state_type(geometry, u=u, v=v, h=h, clock=step * model%get_dt())

  ! Advance the model 1 step
  call model%adv_nsteps(state, 1)

  ! Check clock
  call check_real_scalar(state%get_clock(), "clock", step * model%get_dt() + model%get_dt(), 10E-12_r8kind, errors)

  ! Check u
  call check_min_max_real2d(xps, xpe, yps, ype, state%get_u(), "u", 0.0_r8kind, 0.0_r8kind, errors)

  ! Check v
  call check_min_max_real2d(xps, xpe, yps, ype, state%get_v(), "v", 0.0_r8kind, 0.0_r8kind, errors)

  ! Check h
  call check_min_max_real2d(xps, xpe, yps, ype, state%get_h(), "h", 0.0_r8kind, 0.0_r8kind, errors)

  ! Advance the model 2 steps
  call model%adv_nsteps(state, 2)

  ! Check clock
  call check_real_scalar(state%get_clock(), "clock", step * model%get_dt() + model%get_dt() + model%get_dt() + model%get_dt(), 10E-12_r8kind, errors)

  ! Initialize shallow water state
  h(:,:) = h(:,:) + 10.0_r8kind
  state = shallow_water_state_type(geometry, u=u, v=v, h=h)

  ! Initialize shallow water model
  model = shallow_water_model_type(model_config, geometry)

  ! Advance the model 1 step
  call model%adv_nsteps(state, 1)

  ! Check u
  call check_min_max_real2d(xps, xpe, yps, ype, state%get_u(), "u", 0.0_r8kind, 0.0_r8kind, errors)

  ! Check v
  call check_min_max_real2d(xps, xpe, yps, ype, state%get_v(), "v", 0.0_r8kind, 0.0_r8kind, errors)

  ! Check h
  call check_min_max_real2d(xps, xpe, yps, ype, state%get_h(), "h", 10.0_r8kind, 10.0_r8kind, errors)

  if (errors > 0) then
    call MPI_Abort(MPI_COMM_WORLD, errors, ierr)
  end if

  ! Clean up Serialbox
  !$ser cleanup

  call MPI_Finalize(ierr)

end program Test_Shallow_Water_Model_Adv_Nsteps
