program Tsunami

  ! Get unit numbers for stdin, stdout, stderr in a portable way
  use, intrinsic :: iso_fortran_env, only : stdin=>input_unit,   &
                                            stdout=>output_unit, &
                                            stderr=>error_unit

  use Shallow_Water_Kind,            only : r8kind
  use Shallow_Water_Geometry_Config, only : shallow_water_geometry_config_type
  use Shallow_Water_Model_Config,    only : shallow_water_model_config_type
  use Shallow_Water_Geometry,        only : shallow_water_geometry_type
  use Shallow_water_State,           only : shallow_water_state_type
  use Shallow_Water_Model,           only : shallow_water_model_type
  use mpi

  implicit none

  ! Shallow water config and model objects
  type(shallow_water_geometry_config_type) :: geometry_config
  type(shallow_water_model_config_type)    :: model_config
  type(shallow_water_geometry_type)        :: geometry
  type(shallow_water_state_type)           :: state
  type(shallow_water_model_type)           :: model

  ! Geometry parameters
  integer      :: nx, ny
  real(r8kind) :: xmax, ymax
  namelist /geometry_parm/ nx, ny, xmax, ymax

  ! Model parameters
  real(r8kind) :: dt, u0, v0, b0, h0
  namelist /model_parm/ dt, u0, v0, b0, h0

  ! Runtime parameters
  integer          :: start_step = 0
  integer          :: run_steps   = 1000
  integer          :: output_interval_steps = 100
  character(len=6) :: io_format = 'NETCDF'
  namelist /runtime/ start_step, run_steps, output_interval_steps, io_format

  integer :: nlunit              ! Unit of the namelist file
  integer :: myrank              ! Rank of this MPI process
  integer :: ierr                ! MPI error code
  integer :: t, i, j             ! Index variables
  character(len=64) :: filename  ! Output/input filename
  character(len=1024) :: nlfile  ! Name of the namelist file

  ! For initializing a tsunami pulse
  real(r8kind), allocatable :: h(:,:)
  real(r8kind)              :: dx, dy, xmid, ymid, dsqr, sigma
  integer                   :: yps_local, ype_local, xps_local, xpe_local

  ! Start up MPI
  call MPI_Init(ierr)

  ! Get our rank
  call MPI_Comm_rank(MPI_COMM_WORLD, myrank, ierr)
  
  ! Initialize Serialbox
  !$ser init directory='./serialbox_data' prefix='shallow_water' unique_id=.true.

  ! Read namelist from stdin
  if (myrank == 0) then
    call get_command_argument(1, nlfile)
    open(newunit=nlunit, file=trim(nlfile))

    ! Read geometry namelist
    read(nlunit, nml=geometry_parm)
    rewind(nlunit)

    ! Read model namelist
    read(nlunit, nml=model_parm)
    rewind(nlunit)

    ! Read runtime namelist from stdin
    read(nlunit, nml=runtime)
    rewind(nlunit)

    close(nlunit)
  end if

  ! Broadcast the runtime  namelist settings
  call MPI_Bcast(start_step, 1, MPI_INT, 0, MPI_COMM_WORLD, ierr)
  call MPI_Bcast(run_steps, 1, MPI_INT, 0, MPI_COMM_WORLD, ierr)
  call MPI_Bcast(output_interval_steps, 1, MPI_INT, 0, MPI_COMM_WORLD, ierr)
  call MPI_Bcast(io_format, 6, MPI_CHARACTER, 0, MPI_COMM_WORLD, ierr)

  ! Broadcast the geometry namelist settings
  call MPI_Bcast(nx, 1, MPI_INT, 0, MPI_COMM_WORLD, ierr)
  call MPI_Bcast(ny, 1, MPI_INT, 0, MPI_COMM_WORLD, ierr)
  call MPI_Bcast(xmax, 1, MPI_DOUBLE_PRECISION, 0, MPI_COMM_WORLD, ierr)
  call MPI_Bcast(ymax, 1, MPI_DOUBLE_PRECISION, 0, MPI_COMM_WORLD, ierr)

  ! Broadcast the model namelist settings
  call MPI_Bcast(dt, 1, MPI_DOUBLE_PRECISION, 0, MPI_COMM_WORLD, ierr)
  call MPI_Bcast(u0, 1, MPI_DOUBLE_PRECISION, 0, MPI_COMM_WORLD, ierr)
  call MPI_Bcast(v0, 1, MPI_DOUBLE_PRECISION, 0, MPI_COMM_WORLD, ierr)
  call MPI_Bcast(b0, 1, MPI_DOUBLE_PRECISION, 0, MPI_COMM_WORLD, ierr)
  call MPI_Bcast(h0, 1, MPI_DOUBLE_PRECISION, 0, MPI_COMM_WORLD, ierr)

  ! Create a shallow water geometry configuration from namelist
  geometry_config = shallow_water_geometry_config_type(nx, ny, xmax, ymax)

  ! Create a shallow water geometry from the configuration
  geometry = shallow_water_geometry_type(geometry_config, MPI_COMM_WORLD)

  ! If this is a restart, initialize model from a restart file
  if (start_step /= 0) then

    ! Initialize a default state
    state = shallow_water_state_type(geometry)

    ! Construct name of restart file
    write(filename,'(A,I0.7,A)') 'swout_', start_step, '.nc'

    ! Load restart file into state
    call state%read(filename)

  ! Otherwise create a new model from the namelist
  else

    ! Create a state with a tsunami pulse in it to initialize field h
    allocate(h(geometry%get_xps():geometry%get_xpe(), geometry%get_yps():geometry%get_ype()))
    dx = geometry%get_dx()
    dy = geometry%get_dy()
    xmid = geometry%get_xmax() / 2.0_r8kind
    ymid = geometry%get_ymax() / 2.0_r8kind
    sigma = floor(geometry%get_xmax() / 20.0_r8kind)
    yps_local = geometry%get_yps()
    ype_local = geometry%get_ype()
    xps_local = geometry%get_xps()
    xpe_local = geometry%get_xpe()

    do j = geometry%get_yps(), geometry%get_ype()
      do i = geometry%get_xps(), geometry%get_xpe()
        dsqr = (dble(i - 1) * dx - xmid)**2 + (dble(j - 1) * dy - ymid)**2
        h(i,j) = 5000.0_r8kind + exp(-dsqr / sigma**2) * (h0 - 5000.0_r8kind)
      end do
    end do
    state = shallow_water_state_type(geometry, h=h)
    
  end if

  ! Create a shallow water model configuration from namelist
  model_config = shallow_water_model_config_type(dt, u0, v0, b0, h0)


  ! Initialize shallow water model object
  model = shallow_water_model_type(model_config, geometry)

  ! Construct name of output file
  write(filename,'(A,I0.7,A)') 'swout_', nint(state%get_clock() / model%get_dt()), '.nc'

  ! Write out state if needed
  if (output_interval_steps <= run_steps) call state%write(filename)
  
  ! Capture nsteps of the model
  !$ser savepoint tsunami_pulse-IN
  !$ser data run_steps=run_steps

  ! Run the model
  do t = 0, run_steps, output_interval_steps

    ! Advance the model to next output interval
    call model%adv_nsteps(state, min(output_interval_steps, run_steps - t))

    ! Write out model state if needed
    if (output_interval_steps <= run_steps) then

      ! Construct name of output file
      write(filename,'(A,I0.7,A)') 'swout_', nint(state%get_clock() / model%get_dt()), '.nc'
      call state%write(filename)

    end if

!    call model%print()

  end do

  ! Clean up Serialbox
  !$ser cleanup

  ! Shut down MPI
  call MPI_Finalize(ierr)

end program tsunami
