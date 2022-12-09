[![Linux GNU](https://github.com/NOAA-GSL/SENA-shallow-water/actions/workflows/linux_gnu.yml/badge.svg?branch=develop)](https://github.com/NOAA-GSL/SENA-shallow-water/actions/workflows/linux_gnu.yml)
[![MacOS GNU](https://github.com/NOAA-GSL/SENA-shallow-water/actions/workflows/macos_gnu.yml/badge.svg?branch=develop)](https://github.com/NOAA-GSL/SENA-shallow-water/actions/workflows/macos_gnu.yml)
[![Linux Intel](https://github.com/NOAA-GSL/SENA-shallow-water/actions/workflows/linux_intel.yml/badge.svg?branch=develop)](https://github.com/NOAA-GSL/SENA-shallow-water/actions/workflows/linux_intel.yml)
[![MacOS Intel](https://github.com/NOAA-GSL/SENA-shallow-water/actions/workflows/macos_intel.yml/badge.svg?branch=develop)](https://github.com/NOAA-GSL/SENA-shallow-water/actions/workflows/macos_intel.yml)
[![Docker Ubuntu Intel](https://github.com/NOAA-GSL/SENA-shallow-water/actions/workflows/docker_intel.yml/badge.svg?branch=develop)](https://github.com/NOAA-GSL/SENA-shallow-water/actions/workflows/docker_intel.yml)

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
 - Example usage of [GT4Py](https://github.com/ai2cm/gt4py.git)

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

Please see the [Contributing Guide](https://github.com/NOAA-GSL/SENA-shallow-water/blob/main/CONTRIBUTING.md) for information about contributing to this respository.

# Fortran Version

## Dependencies

The Fortran implementation of this repository requires the following:

* C compiler
* Fortran compiler
* MPI (e.g. [openmpi](https://www.open-mpi.org/software/ompi/v4.1/), [mvapich2](http://mvapich.cse.ohio-state.edu/downloads/), [mpich](https://www.mpich.org/downloads/), [Intel MPI](https://software.intel.com/content/www/us/en/develop/tools/oneapi/components/mpi-library.html#gs.1cg64u))
* [netcdf-c](https://www.unidata.ucar.edu/downloads/netcdf/)
* [netcdf-fortran](https://www.unidata.ucar.edu/downloads/netcdf/)
* [cmake](https://cmake.org/download/) (version >= 3.12)

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
on how the GNU compiler was installed, you may need to specify the C compiler name as `gcc-$version`.
For example:

```bash
$ export CC=gcc-10
```

### Buidling with Serialbox (Optional)

After building [Serialbox](https://gridtools.github.io/serialbox/) [(stepts listed below)](#serialbox-install-instructions-optional) with the Fortran interface option, the only difference from the basic build procedure listed above is inserting `-DGT4PY=on` option to the `cmake` command, which should find the Serialbox installation. Make sure to use the same `CC` and `FC` environment variables used in the Serialbox build when building the Shallow Water Model. 

```bash
$ cmake -DCMAKE_BUILD_TYPE=debug -DGT4PY=on ..
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

To run the code, call the executable and pass the namelist (located in `parm/`) as an argument.

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

# Python (GT4Py) Version

## Dependencies

The Python implementation of this repository requires the following:

* Python (Version >= 3.8)

Underlying dependencies for Python packages include:

* MPI Implementation (e.g. [openmpi](https://www.open-mpi.org/software/ompi/v4.1/), [mvapich2](http://mvapich.cse.ohio-state.edu/downloads/), [mpich](https://www.mpich.org/downloads/), [Intel MPI](https://software.intel.com/content/www/us/en/develop/tools/oneapi/components/mpi-library.html#gs.1cg64u))
* [netcdf-c](https://www.unidata.ucar.edu/downloads/netcdf/)
* [netcdf-fortran](https://www.unidata.ucar.edu/downloads/netcdf/)


If Serialbox will be used for comparison tests, the [Fortran Version](#fortran-version) dependencies apply, in addition to `boost`, to include:
* C Compiler
* Fortran Compiler
* [cmake](https://cmake.org/download/) (version >= 3.12)
* [boost](https://www.boost.org/) (version >= 1.65.1)


## Install Python Dependencies (from the directory containing this file)

```bash
$ pip install -r ./requirements.txt
```

## Test Instructions

To run the full test suite (Both MPI & non-MPI tests): 

```bash
$ cd test/python
$ mpiexec -n 1 python -m pytest -s
$ mpiexec -n <number of processes> python -m pytest --with-mpi -s
```

Certain tests use mpi, while others do not. The commands to execute individual tests differ.

To run a specific test, which does not use MPI (for example):

```bash
$ pytest test_shallow_water_gt4py_config.py
```

To run a specific test using MPI (for example):

```bash
$ mpiexec -n 4 python -m pytest --with-mpi test_shallow_water_model.py
```
## Run Instructions

To run the code, call the `shallow_water` Python program and pass a `.yml` configuration file (located in `parm/`) as an argument.

```bash 
$ cd src/python
$ mpiexec -n <number of processors> python -m shallow_water ../../parm/shallow_water.yml
```

## Visualization Instructions

A Python [Visualization script](src/python/shallow_water_visualization.py) is included, which outputs a .gif file to view model animation with an inital gaussian pulse.  
Additional Python packages may need to be installed (`matplotlib`). 

```bash
$ cd src/python
$ mpiexec -n 1 python -m shallow_water_visualization ../../parm/shallow_water.yml
```

# Comparison Tests 

Two forms of comparison tests exist to validate the GT4Py port against the Fortran reference implementation. [Serialbox](https://github.com/GridTools/serialbox/) is a tool that enables granular verification of a GT4Py port. In placing preprocessor directives in the Fortran code then compiling and executing the code, savepoints (variables) can be captured, and compared against Python counterparts to validate stencil functions. Using Serialbox allows a piecemeal approach to validating a port, as comparisons can be done at the subroutine level, and does not depend on the added code needed to write output to netcdf files. The [`nccmp`](https://gitlab.com/remikz/nccmp) utility also provides the ability to compare results, since the functionality to write to netcdf files has been implementated in both versions. 

## Serialbox

1) Build Serialbox issuing the commands listed below. Additional Platform/Compiler specific examples can be found in the [Github Actions Workflow](https://github.com/NOAA-GSL/SENA-shallow-water/tree/develop/.github/workflows) files.  For further information and guidance, please refer to the [Serialbox 2.2.0 Documention](https://gridtools.github.io/serialbox/).


```bash
$ git clone https://github.com/eth-cscs/serialbox2.git serialbox
$ cd serialbox
$ mkdir build && cd build
$ export CC=<name of C compiler>
$ export FC=<name of Fortran Compiler>
$ cmake ../ -DSERIALBOX_ENABLE_FORTRAN=ON
$ make
$ make install
```

2) In some cases, you may have to set a couple of environment variables to allow Python to find the Serialbox library.

```bash 
$ export SERIALBOX_ROOT=path/to/serialbox2/install
$ export DYLD_LIBRARY_PATH=$SERIALBOX_ROOT/lib:$DYLD_LIBRARY_PATH
```

3) Follow the [Fortran Version - Buidling with Serialbox](#buidling-with-serialbox-optional) build.

4) Install the shallow_water_serialize executable from within the `/build` directory. 

```bash 
$ make install
```

5) Call the executable, passing the namelist as an argument. (The Python comparison script depends on the `serialbox_data/` being in the `src/python` directory).

```bash 
$ cd src/python
$ ../../exe/shallow_water_serialize.x ../../parm/shallow_water.nl
```

6) After installation of the Python Version requirements, execute the Python script to compare the Fortran Serialbox output with the GT4Py implementation. The `parm/shallow_water.yml` used in the script provides the same configuration as the `parm/shallow_water.nl` used to run the model on the Fortran side. 

Validate against the Fortran implementation by running the following script:

```bash
$ cd src/python
$ python shallow_water_serialize.py
```

## nccmp

`nccmp` is a command line utility which compares two NetCDF files bitwise, semantically, or with a user defined tolerance. Download instructions can be found [here](https://gitlab.com/remikz/nccmp#download).

After running both the [Fortran Version](#install-and-run-instructions) & [Python Version](#run-instructions) of the model, you can compare NetCDF output files (for example): 

```bash 
$ nccmp -mdfs <last/fortran/NetCDF/output/.nc> <last/python/NetCDF/output/.nc>
```
