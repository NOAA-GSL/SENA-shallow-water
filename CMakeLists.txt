################################################################################
# shallow_water
################################################################################

cmake_minimum_required( VERSION 3.12 )
project( shallow_water LANGUAGES CXX Fortran )

list( APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake )

enable_testing()

## Compiler options
include( ${PROJECT_NAME}_compiler_flags )

## Dependencies
find_package( OpenMP COMPONENTS CXX Fortran )
find_package( MPI REQUIRED COMPONENTS CXX Fortran )
find_package( NetCDF REQUIRED COMPONENTS C Fortran )

add_subdirectory( src )
add_subdirectory( test )
