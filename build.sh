#!/bin/env bash

##################
# Set the compiler
##################

# For GNU
# export CC=gcc-11
# export FC=gfortran-11

# For Intel
#export CC=icc
#export FC=ifort

####################
# Set the build type
####################

export BUILD_TYPE=debug
#export BUILD_TYPE=release

############################
# Create the build directory
############################
rm -rf build
mkdir build
cd build

###########
# Run cmake
###########
cmake .. -DGT4PY=on -DCMAKE_BUILD_TYPE=${BUILD_TYPE} -DMPIEXEC_PREFLAGS=--oversubscribe

#######
# Build
#######
make -j4 VERBOSE=1

######
# Test
######
# ctest --output-on-failure
