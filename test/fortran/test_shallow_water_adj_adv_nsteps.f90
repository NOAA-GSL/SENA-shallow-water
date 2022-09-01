program Test_Shallow_Water_ADJ_Adv_Nsteps

  use Shallow_Water_Kind,            only : r8kind
  use Shallow_Water_Geometry_Config, only : shallow_water_geometry_config_type
  use Shallow_Water_Model_Config,    only : shallow_water_model_config_type
  use Shallow_Water_Model,           only : shallow_water_model_type, shallow_water_tl_type, shallow_water_adj_type
  use Shallow_Water_Geometry,        only : shallow_water_geometry_type
  use Shallow_Water_State,           only : shallow_water_state_type
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
  real(r8kind), parameter :: dt = 0.68_r8kind * (xmax / (dble(nx) - 1.0)) / (u0 + sqrt(g * (h0 - b0)))

  ! Test parameters
  integer, parameter :: spinup_steps = 1000 ! model spinup steps

  type(shallow_water_geometry_config_type) :: geometry_config
  type(shallow_water_model_config_type)    :: model_config
  type(shallow_water_geometry_type)        :: geometry
  type(shallow_water_model_type)           :: shallow_water
  type(shallow_water_tl_type)              :: shallow_water_tl
  type(shallow_water_adj_type)             :: shallow_water_adj
  type(shallow_water_state_type)           :: state, stated, stateb
  type(shallow_water_state_type)           :: statex, statey

  real(r8kind), allocatable :: u(:,:), v(:,:), h(:,:)
  real(r8kind), allocatable :: xu(:,:), xv(:,:), xh(:,:)
  real(r8kind), allocatable :: yu(:,:), yv(:,:), yh(:,:)
  real(r8kind), allocatable :: global_ud(:,:), global_vd(:,:), global_hd(:,:)
  real(r8kind), allocatable :: global_ub(:,:), global_vb(:,:), global_hb(:,:)
  real(r8kind), allocatable :: global_xu(:,:), global_xv(:,:), global_xh(:,:)
  real(r8kind), allocatable :: global_yu(:,:), global_yv(:,:), global_yh(:,:)
  real(r8kind)              :: dx, dy, xmid, ymid, dsqr, sigma
  real(r8kind)              :: dot_ratio
  integer                   :: xps, xpe, yps, ype  ! Local domain index ranges
  integer                   :: i, j

  ! MPI variables
  integer              :: ierr
  integer              :: myrank

  ! Initialize MPI
  call MPI_Init(ierr)

  ! Create a geometry configuration as specified by the namelist
  geometry_config = shallow_water_geometry_config_type(nx, ny, xmax, ymax)

  ! Create a model geometry with the config
  geometry = shallow_water_geometry_type(geometry_config, MPI_COMM_WORLD)

  ! Get index ranges from geometry
  xps = geometry%get_xps()
  xpe = geometry%get_xpe()
  yps = geometry%get_yps()
  ype = geometry%get_ype()

  ! Allocate space for model states
  allocate(u(xps:xpe, yps:ype), v(xps:xpe, yps:ype), h(xps:xpe, yps:ype))
  allocate(xu(xps:xpe, yps:ype), xv(xps:xpe, yps:ype), xh(xps:xpe, yps:ype))
  allocate(yu(xps:xpe, yps:ype), yv(xps:xpe, yps:ype), yh(xps:xpe, yps:ype))

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

  ! Spinup the forward model to avoid initial condition issues
  write(*,*) 'Integrating forward model spinup steps: ', spinup_steps
  call shallow_water%adv_nsteps(state, spinup_steps)

  ! Advance the forward model 100 steps and use it to compute x
  u = state%get_u()
  v = state%get_v()
  h = state%get_h()
  call shallow_water%adv_nsteps(state, 100)
  xu = (state%get_u() - u)
  xv = (state%get_v() - v)
  xh = (state%get_h() - h)
  xu = xu * 1000.0_r8kind
  xv = xv * 1000.0_r8kind
  xh = xh * 1000.0_r8kind
  statex = shallow_water_state_type(geometry, u=xu, v=xv, h=xh)
  stated = shallow_water_state_type(geometry, u=xu, v=xv, h=xh)

  ! Advance the forward model another 100 steps and use it to compute y
  call shallow_water%adv_nsteps(state, 100)
  yu = (state%get_u() - u)
  yv = (state%get_v() - v)
  yh = (state%get_h() - h)
  yu = yu * 1000.0_r8kind
  yv = yv * 1000.0_r8kind
  yh = yh * 1000.0_r8kind
  statey = shallow_water_state_type(geometry, u=yu, v=yv, h=yh)
  stateb = shallow_water_state_type(geometry, u=yu, v=yv, h=yh)

  ! Advance statex using the shallow_water_tl
  shallow_water_tl = shallow_water_tl_type(model_config, geometry)
  call shallow_water_tl%adv_nsteps(state=stated, trajectory=state, nsteps=1)

  ! Advance statey using the shallow_water_adj
  shallow_water_adj = shallow_water_adj_type(model_config, geometry)
  call shallow_water_adj%adv_nsteps(state=stateb, trajectory=state, nsteps=1)

  ! Get the MPI rank of this process
  myrank = geometry%get_rank()

  ! Gather M'(x)
  if (myrank == 0) then
    allocate(global_ud(nx,ny))
    allocate(global_vd(nx,ny))
    allocate(global_hd(nx,ny))
  end if
  call stated%gather(global_ud, global_vd, global_hd)

  ! Gather M*(y)
  if (myrank == 0) then
    allocate(global_ub(nx,ny))
    allocate(global_vb(nx,ny))
    allocate(global_hb(nx,ny))
  end if
  call stateb%gather(global_ub, global_vb, global_hb)

  ! Gather x
  if (myrank == 0) then
    allocate(global_xu(nx,ny))
    allocate(global_xv(nx,ny))
    allocate(global_xh(nx,ny))
  end if
  call statex%gather(global_xu, global_xv, global_xh)

  ! Gather y
  if (myrank == 0) then
    allocate(global_yu(nx,ny))
    allocate(global_yv(nx,ny))
    allocate(global_yh(nx,ny))
  end if
  call statey%gather(global_yu, global_yv, global_yh)

  ! Now compute ratio of dot product on rank 0 using global arrays that we just gathered
  if (myrank == 0) then
    dot_ratio = (dot_product(reshape(global_ud, (/nx*ny/)), reshape(global_yu, (/nx*ny/)))  + &
                 dot_product(reshape(global_vd, (/nx*ny/)), reshape(global_yv, (/nx*ny/)))  + &
                 dot_product(reshape(global_hd, (/nx*ny/)), reshape(global_yh, (/nx*ny/)))) / &
                (dot_product(reshape(global_xu, (/nx*ny/)), reshape(global_ub, (/nx*ny/))) + &
                 dot_product(reshape(global_xv, (/nx*ny/)), reshape(global_vb, (/nx*ny/))) + &
                 dot_product(reshape(global_xh, (/nx*ny/)), reshape(global_hb, (/nx*ny/))))

    ! Write out the value of <M'(x),y> / <x, M*(y)> for combined u, v, h
    write(*,'(A25,F20.15)') "<M'(x),y> / <x, M*(y)> = ", dot_ratio

    if (abs(dot_ratio - 1.0_r8kind) > 1.0E-14_r8kind) then
      write(*,*) "ERROR: Adjoint dot product test failed"
      call MPI_Abort(MPI_COMM_WORLD, 1, ierr)
    end if
  end if

  ! Shut down MPI
  call MPI_Finalize(ierr)

end program Test_Shallow_Water_ADJ_Adv_Nsteps
