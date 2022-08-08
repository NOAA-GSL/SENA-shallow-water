# (C) Copyright 2018 UCAR.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

####################################################################
# FLAGS COMMON TO ALL BUILD TYPES
####################################################################
set( CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -emf -rmoid -lhugetlbfs")

####################################################################
# RELEASE FLAGS
####################################################################

set( CMAKE_Fortran_FLAGS_RELEASE "-O3 -hfp3 -hscalar3 -hvector3 -hPIC" )

####################################################################
# DEBUG FLAGS
####################################################################

set( CMAKE_Fortran_FLAGS_DEBUG   "-O0 -Gfast -Ktrap=fp" )

####################################################################

# Meaning of flags
# ----------------
# -hfp3     : Special optimisation for floating points
# -Ktrap=fp : Abort on NaN
# -R b      : Bounds checking
# -hflex_mp=conservative -hfp1 : Obtain bit-reproducible results
# -hflex_mp=intolerant -hfp2   : Obtain bit-reproducible results (also)
# -hadd_paren : encourage left to right fp evaluation
# -hscalarN , -hvectorN : optimisation for scalar and vectorisation
# -homp/-hnoomp : Enable/Disable OpenMP
# -rmoi : create compiler listing
