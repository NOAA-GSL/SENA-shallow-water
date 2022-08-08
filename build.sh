#!/bin/env bash

##################
# Set the compiler
##################

# For GNU
export CC=gcc
export FC=gfortran

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
cmake .. -DCMAKE_BUILD_TYPE=${BUILD_TYPE}

#######
# Build
#######
make -j4 VERBOSE=1

######
# Test
######
ctest --output-on-failure
