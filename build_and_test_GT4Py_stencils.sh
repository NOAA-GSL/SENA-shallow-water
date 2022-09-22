#!/bin/env bash

##################
# Set the compiler
##################

# For GNU
export CC=gcc-12
export FC=gfortran-12

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

# ctest -VV -R shallow_water_model_adv_nsteps_1

######
# Create executable and Serialize data
######

rm -rf ../exe

make install 

######
#  Serialize data
######
cd ../src/python

rm -rf ./serialbox_data
rm -rf *.nc

../../exe/shallow_water.x ../../parm/serialize_test.nl

######
# Run GT4Py Stencils & Compare answers
######
python shallow_water_stencils.py
