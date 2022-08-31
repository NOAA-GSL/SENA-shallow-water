################################################################################
# shallow_water
################################################################################

cmake_minimum_required( VERSION 3.12 )
project( shallow_water LANGUAGES CXX Fortran )

list( APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake )

enable_testing()

## Build options
option(GT4PY "Enable GT4Py" off)

## Compiler options
include( ${PROJECT_NAME}_compiler_flags )

## Dependencies
find_package( OpenMP COMPONENTS CXX Fortran )
find_package( MPI REQUIRED COMPONENTS CXX Fortran )
find_package( NetCDF REQUIRED COMPONENTS C Fortran )

## Option dependencies
if (GT4PY)
  ## Boost
  find_package( Boost 1.65.1 REQUIRED)

  ## SerialBox
  set(SERIALBOX_USE_SHARED_LIBS ON)
  find_package( Serialbox 2.2.0 REQUIRED COMPONENTS C++ C Fortran)
  add_definitions(-DSERIALIZE)
endif()

add_subdirectory( src )
add_subdirectory( test )
