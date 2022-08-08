program Test_Shallow_Water_Geometry

  use Shallow_Water_Kind,            only : r8kind
  use Shallow_Water_Geometry_Config, only : shallow_water_geometry_config_type
  use Shallow_Water_Geometry,        only : shallow_water_geometry_type
  use Test_Utilities,                only : check_real_scalar, check_integer_scalar, check_min_max_real2d
  use mpi

  implicit none

  ! Shallow water config object
  type(shallow_water_geometry_config_type) :: config
  type(shallow_water_geometry_type)        :: geometry

  ! Config parameters
  integer, parameter :: nx = 11
  integer, parameter :: ny = 11
  real(r8kind), parameter :: xmax = 10000.0
  real(r8kind), parameter :: ymax = 10000.0

  ! Test variables
  integer :: errors
  integer :: north, south, west, east
  integer :: npx, npy
  integer :: xps, xpe, yps, ype
  integer :: xts, xte, yts, yte
  integer :: xms, xme, yms, yme

  ! MPI variables
  integer :: ierr
  integer :: nranks, myrank

  ! Start MPI
  call MPI_Init(ierr)
  
  ! Initialize error count to 0
  errors = 0

  ! Get my MPI rank
  call MPI_Comm_rank(MPI_COMM_WORLD, myrank, ierr)

  ! Get number of MPI ranks
  call MPI_Comm_size(MPI_COMM_WORLD, nranks, ierr)

  ! Create a shallow water model configuration
  config = shallow_water_geometry_config_type(nx, ny, xmax, ymax)

  ! Create a shallow water geometry from the configurate
  geometry = shallow_water_geometry_type(config, MPI_COMM_WORLD)

  ! Set correct answers to compare geometry settings to
  select case (nranks)
    case(1)
      north = -1
      south = -1
      west = -1
      east = -1
      npx = nx
      npy = ny
      xps = 1
      xpe = nx
      yps = 1
      ype = ny
      xts = 2
      xte = nx -1
      yts = 2
      yte = ny - 1
      xms = 1
      xme = nx
      yms = 1
      yme = ny
    case(2)
      select case (myrank)
        case(0)
          north = 1
          south = -1
          west = -1
          east = -1
          npx = nx
          npy = 5
          xps = 1
          xpe = nx
          yps = 1
          ype = 5
          xts = 2
          xte = nx -1
          yts = 2
          yte = 5
          xms = 1
          xme = nx
          yms = 1
          yme = 6
        case(1)
          north = -1
          south = 0
          west = -1
          east = -1
          npx = nx
          npy = 6
          xps = 1
          xpe = nx
          yps = 6
          ype = ny
          xts = 2
          xte = nx -1
          yts = 6
          yte = ny - 1
          xms = 1
          xme = nx
          yms = 5
          yme = ny
      end select
    case(4)
      select case (myrank)
        case(0)
          north = 2
          south = -1
          west = -1
          east = 1
          npx = 5
          npy = 5
          xps = 1
          xpe = 5
          yps = 1
          ype = 5
          xts = 2
          xte = 5
          yts = 2
          yte = 5
          xms = 1
          xme = 6
          yms = 1
          yme = 6
        case(1)
          north = 3
          south = -1
          west = 0
          east = -1
          npx = 6
          npy = 5
          xps = 6
          xpe = nx
          yps = 1
          ype = 5
          xts = 6
          xte = nx -1
          yts = 2
          yte = 5
          xms = 5
          xme = nx
          yms = 1
          yme = 6
        case(2)
          north = -1
          south = 0
          west = -1
          east = 3
          npx = 5
          npy = 6
          xps = 1
          xpe = 5
          yps = 6
          ype = ny
          xts = 2
          xte = 5
          yts = 6
          yte = ny-1
          xms = 1
          xme = 6
          yms = 5
          yme = ny
        case(3)
          north = -1
          south = 1
          west = 2
          east = -1
          npx = 6
          npy = 6
          xps = 6
          xpe = nx
          yps = 6
          ype = ny
          xts = 6
          xte = nx -1
          yts = 6
          yte = ny - 1
          xms = 5
          xme = nx
          yms = 5
          yme = ny
      end select
    case(9)
      select case (myrank)
        case(0)
          north = 3
          south = -1
          west = -1
          east = 1
          npx = 3
          npy = 3
          xps = 1
          xpe = 3
          yps = 1
          ype = 3
          xts = 2
          xte = 3
          yts = 2
          yte = 3
          xms = 1
          xme = 4
          yms = 1
          yme = 4

        case(1)
          north = 4
          south = -1
          west = 0
          east = 2
          npx = 4
          npy = 3
          xps = 4
          xpe = 7
          yps = 1
          ype = 3
          xts = 4
          xte = 7
          yts = 2
          yte = 3
          xms = 3
          xme = 8
          yms = 1
          yme = 4

        case(2)
          north = 5
          south = -1
          west = 1
          east = -1
          npx = 4
          npy = 3
          xps = 8
          xpe = nx
          yps = 1
          ype = 3
          xts = 8
          xte = nx-1
          yts = 2
          yte = 3
          xms = 7
          xme = nx
          yms = 1
          yme = 4

        case(3)
          north = 6
          south = 0
          west = -1
          east = 4
          npx = 3
          npy = 4
          xps = 1
          xpe = 3
          yps = 4
          ype = 7
          xts = 2
          xte = 3
          yts = 4
          yte = 7
          xms = 1
          xme = 4
          yms = 3
          yme = 8

        case(4)
          north = 7
          south =1
          west = 3
          east = 5
          npx = 4
          npy = 4
          xps = 4
          xpe = 7
          yps = 4
          ype = 7
          xts = 4
          xte = 7
          yts = 4
          yte = 7
          xms = 3
          xme = 8
          yms = 3
          yme = 8

        case(5)
          north = 8
          south = 2
          west = 4
          east = -1
          npx = 4
          npy = 4
          xps = 8
          xpe = nx
          yps = 4
          ype = 7
          xts = 8
          xte = nx -1
          yts = 4
          yte = 7
          xms = 7
          xme = nx
          yms = 3
          yme = 8

        case(6)
          north = -1
          south = 3
          west = -1
          east = 7
          npx = 3
          npy = 4
          xps = 1
          xpe = 3
          yps = 8
          ype = ny
          xts = 2
          xte = 3
          yts = 8
          yte = ny-1
          xms = 1
          xme = 4
          yms = 7
          yme = ny

        case(7)
          north = -1
          south = 4
          west = 6
          east = 8
          npx = 4
          npy = 4
          xps = 4
          xpe = 7
          yps = 8
          ype = ny
          xts = 4
          xte = 7
          yts = 8
          yte = ny-1
          xms = 3
          xme = 8
          yms = 7
          yme = ny

        case(8)
          north = -1
          south = 5
          west = 7
          east = -1
          npx = 4
          npy = 4
          xps = 8
          xpe = nx
          yps = 8
          ype = ny
          xts = 8
          xte = nx -1
          yts = 8
          yte = ny -1
          xms = 7
          xme = nx
          yms = 7
          yme = ny
      end select
    case default
      write (*,'(A,I6,A)') "test_shallow_water_geometry does not support ", nranks, " cores"
      call MPI_Abort(MPI_COMM_WORLD, 1, ierr)
  end select

  ! Check MPI communicator
  call check_integer_scalar(geometry%get_communicator(), "communicator", MPI_COMM_WORLD, errors)

  ! Check rank
  call check_integer_scalar(geometry%get_rank(), "rank", myrank, errors)

  ! Check nranks
  call check_integer_scalar(geometry%get_nranks(), "nranks", nranks, errors)

  ! Check nx
  call check_integer_scalar(geometry%get_nx(), "nx", nx, errors)

  ! Check ny
  call check_integer_scalar(geometry%get_ny(), "ny", ny, errors)

  ! Check xmax
  call check_real_scalar(geometry%get_xmax(), "xmax", xmax, 0.0_r8kind, errors)

  ! Check ymax
  call check_real_scalar(geometry%get_ymax(), "ymax", ymax, 0.0_r8kind, errors)

  ! Check dx
  call check_real_scalar(geometry%get_dx(), "dx", xmax / (dble(nx) - 1.0_r8kind), 10E-12_r8kind, errors)

  ! Check dy
  call check_real_scalar(geometry%get_dy(), "dy", ymax / (dble(ny) - 1.0_r8kind), 10E-12_r8kind, errors)

  ! Check north
  call check_integer_scalar(geometry%get_north(), "north", north, errors)

  ! Check south
  call check_integer_scalar(geometry%get_south(), "south", south, errors)

  ! Check west
  call check_integer_scalar(geometry%get_west(), "west", west, errors)

  ! Check east
  call check_integer_scalar(geometry%get_east(), "east", east, errors)

  ! Check npx
  call check_integer_scalar(geometry%get_npx(), "npx", npx, errors)

  ! Check npy
  call check_integer_scalar(geometry%get_npy(), "npy", npy, errors)

  ! Check xps
  call check_integer_scalar(geometry%get_xps(), "xps", xps, errors)

  ! Check xpe
  call check_integer_scalar(geometry%get_xpe(), "xpe", xpe, errors)

  ! Check yps
  call check_integer_scalar(geometry%get_yps(), "yps", yps, errors)

  ! Check ype
  call check_integer_scalar(geometry%get_ype(), "ype", ype, errors)

  ! Check xts
  call check_integer_scalar(geometry%get_xts(), "xts", xts, errors)

  ! Check xte
  call check_integer_scalar(geometry%get_xte(), "xte", xte, errors)

  ! Check yts
  call check_integer_scalar(geometry%get_yts(), "yts", yts, errors)

  ! Check yte
  call check_integer_scalar(geometry%get_yte(), "yte", yte, errors)

  ! Check xms
  call check_integer_scalar(geometry%get_xms(), "xms", xms, errors)

  ! Check xme
  call check_integer_scalar(geometry%get_xme(), "xme", xme, errors)

  ! Check yms
  call check_integer_scalar(geometry%get_yms(), "yms", yms, errors)

  ! Check yme
  call check_integer_scalar(geometry%get_yme(), "yme", yme, errors)


  if (errors > 0) then
    call MPI_Abort(MPI_COMM_WORLD, errors, ierr)
  end if

  ! Shut down MPI
  call MPI_Finalize(ierr)

end program Test_Shallow_Water_Geometry
