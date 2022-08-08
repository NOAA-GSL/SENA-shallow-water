[![Linux GNU](https://github.com/NOAA-GSL/shallow-water-demo/actions/workflows/linux_gnu.yml/badge.svg?branch=develop)](https://github.com/NOAA-GSL/shallow-water-demo/actions/workflows/linux_gnu.yml)
[![MacOS GNU](https://github.com/NOAA-GSL/shallow-water-demo/actions/workflows/macos_gnu.yml/badge.svg?branch=develop)](https://github.com/NOAA-GSL/shallow-water-demo/actions/workflows/macos_gnu.yml)
[![Linux Intel](https://github.com/NOAA-GSL/shallow-water-demo/actions/workflows/linux_intel.yml/badge.svg?branch=develop)](https://github.com/NOAA-GSL/shallow-water-demo/actions/workflows/linux_intel.yml)
[![MacOS Intel](https://github.com/NOAA-GSL/shallow-water-demo/actions/workflows/macos_intel.yml/badge.svg?branch=develop)](https://github.com/NOAA-GSL/shallow-water-demo/actions/workflows/macos_intel.yml)
[![Docker Ubuntu Intel](https://github.com/NOAA-GSL/shallow-water-demo/actions/workflows/docker_intel.yml/badge.svg?branch=develop)](https://github.com/NOAA-GSL/shallow-water-demo/actions/workflows/docker_intel.yml)

```
This repository is a scientific product and is not official communication
of the National Oceanic and Atmospheric Administration, or the United States
Department of Commerce. All NOAA GitHub project code is provided on an ‘as
is’ basis and the user assumes responsibility for its use. Any claims against
the Department of Commerce or Department of Commerce bureaus stemming from
the use of this GitHub project will be governed by all applicable Federal
law. Any reference to specific commercial products, processes, or service
by service mark, trademark, manufacturer, or otherwise, does not constitute
or imply their endorsement, recommendation or favoring by the Department of
Commerce. The Department of Commerce seal and logo, or the seal and logo of
a DOC bureau, shall not be used in any manner to imply endorsement of any
commercial product or activity by DOC or the United States Government.
```

# Overview

NOTE: If you are reading this with a plain text editor, please note that this document is
formatted with Markdown syntax elements.  See https://www.markdownguide.org/cheat-sheet/
for more information.

This repository is intended to be an educational tool for demonstrating:

 - Use of modern Fortran language constructs
 - Creation of a portable build system
 - Use of test driven development (TDD) to build an application test suite
 - Automated testing and continuous integration (CI)
 - How to use [GT4Py](https://github.com/ai2cm/gt4py.git)

This demonstration project uses a "toy" shallow water model implementation
based on work by Steve McHale ([Shallow Water Wave CFD (Tsunami Modelling),
MATLAB Central File Exchange](
https://www.mathworks.com/matlabcentral/fileexchange/17716-shallow-water-wave-cfd-tsunami-modelling)
).

## Shallow Water Model Description

This is a "toy" model that simulates simplified shallow water equations in
a single layer rectangular "bathtub" domain using reflective boundary
conditions. The model is initialized with a gaussian pulse in the center of
the domain with the initial velocity of the surface set to zero. The surface
sloshes around in the tub until it becomes a flat surface. No external forcing
is used to keep the simulation going. This model comes with both a tangent
linear model and an adjoint model, which can be used for a variety of
applications including 4D variational data assimilation.

## Contributing

Please see the [Contributing Guide](https://github.com/NOAA-GSL/shallow-water-demo/blob/develop/CONTRIBUTING.md) for information about contributing to this respository.

## Dependencies

This repository requires the following

* C compiler
* Fortran compiler
* MPI (e.g. [openmpi](https://www.open-mpi.org/software/ompi/v4.1/), [mvapich2](http://mvapich.cse.ohio-state.edu/downloads/), [mpich](https://www.mpich.org/downloads/), [Intel MPI](https://software.intel.com/content/www/us/en/develop/tools/oneapi/components/mpi-library.html#gs.1cg64u))
* [netcdf-c](https://www.unidata.ucar.edu/downloads/netcdf/)
* [netcdf-fortran](https://www.unidata.ucar.edu/downloads/netcdf/)
* [cmake](https://cmake.org/download/) (version >= 3.10)

## Build Instructions

This code uses an out-of-source cmake build, meaning that the build must be done in directory that is not in the source tree.

### Basic build procedure (from the directory containing this file)

The basic build steps are as follows:

```bash
$ rm -rf build ; mkdir build ; cd build
$ export CC=<name of C compiler>
$ export FC=<name of fortran compiler> 
$ cmake .. -DCMAKE_BUILD_TYPE=<debug | release>
$ make VERBOSE=1
```

On many systems, the above will suffice. However, some systems will require you to help cmake
find dependencies, particularly if software depencencies are not installed in standard locations.
See below for more information.

### Machines that use modules to manage software

Most HPC systems use modules to manage software.  Make sure you have loaded the versions of
the compiler and software you want to use before running the build steps above.  This will allow build
dependencies to be found properly.  For example:

```bash
$ module load intel netcdf cmake
```

### Machines that do not use modules to manage software

If compilers and/or NetCDF is not installed in a standard location where cmake can find it, you
may need to add their installation paths to the `CMAKE_PREFIX_PATH` before running the steps
above. For example:

```bash
$ export CMAKE_PREFIX_PATH=$CMAKE_PREFIX_PATH:/path/to/netcdf:/path/to/netcdf-fortran
```

### Building on a Mac

By default, gcc points to the clang compiler on Mac.  To use the GNU compiler on Mac, depending
on how the GNU compiler was installed, you may need to specify the C compiler name as gcc-$version.
For example:

```bash
$ export CC=gcc-10
```

## Test Instructions

To run the test suite (from the build directory):

```bash
$ ctest
```

To run a specific test (for example):

```bash
$ ctest -R shallow_water_model_regression_1
```

To run a specific test with full output to get more information about a failure (for example):

```bash
$ ctest -VV -R shallow_water_model_regression_1
```

NOTE: The timings provided by `ctest` report how long each test took.  They are not rigorous
performance measurements.  Other tools will be needed to collect accurate performance data.

## Install and Run Instructions

The shallow water executable may be installed into the `exe/` directory after the build completes.  This make it easier to run. From the `build` directory:

```bash
$ make install
```

To run the code, call the executable and pass the namelist (located in `parm/`) as an argument

```bash
$ cd exe
$ ./shallow_water.x ../parm/shallow_water.nl
```

This will produce NetCDF files that can be inspected or visualized (e.g. with [ncview](http://meteora.ucsd.edu/~pierce/ncview_home_page.html)).

## Build and test script

For convenience, a build script is provided that builds the code and runs the test suite. This
script serves as a minimum example of how to build and test the code.  You will be required to
edit the script to modify compilers, load modules (if needed), and set the desired build type.

Once you have edited the script to your desired settings, run it.

```bash
$ ./build.sh
```

