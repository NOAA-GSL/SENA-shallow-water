# Copy testing namelists
file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/../test_input/test_shallow_water_config.nl" DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")
file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/../test_input/test_shallow_water_reader.nc" DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")

add_library(${PROJECT_NAME}_test_utilities SHARED test_utilities.f90)
target_link_libraries(${PROJECT_NAME}_test_utilities PUBLIC ${PROJECT_NAME})

list( APPEND shallow_water_test_src_files
test_shallow_water_geometry_config_arglist.f90
test_shallow_water_geometry_config_nlfile.f90
test_shallow_water_geometry_config_nlunit.f90
test_shallow_water_model_config_arglist.f90
test_shallow_water_model_config_nlfile.f90
test_shallow_water_model_config_nlunit.f90
test_shallow_water_geometry.f90
test_shallow_water_state_init_default.f90
test_shallow_water_state_init_optional.f90
test_shallow_water_state_halo.f90
test_shallow_water_state_read.f90
test_shallow_water_state_write.f90
test_shallow_water_model_init.f90
test_shallow_water_model_adv_nsteps.f90
test_shallow_water_model_regression.f90
test_shallow_water_tl_init.f90
test_shallow_water_tl_adv_nsteps.f90
test_shallow_water_adj_init.f90
test_shallow_water_adj_adv_nsteps.f90
)

if(GT4PY)
  set(PP_OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/serialbox_pp)
  serialbox_run_pp_ser(SOURCES ${shallow_water_test_src_files}
                       OUTPUT_DIR ${PP_OUTPUT})
endif()

foreach(test_src ${shallow_water_test_src_files})
  string(REPLACE ".f90" ".x" test_exe ${test_src})
  if(GT4PY)
    add_executable( ${test_exe} ${PP_OUTPUT}/${test_src} )
    target_link_libraries( ${test_exe} Serialbox::SerialboxFortranStatic)
  else()
    add_executable( ${test_exe} ${test_src} )
  endif()
  target_link_libraries( ${test_exe} shallow_water_test_utilities )
endforeach()

#######################################
## Shallow water geometry config tests
#######################################
add_test( NAME shallow_water_geometry_config_arglist
          COMMAND bash -c "${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 1 ${MPIEXEC_PREFLAGS} ./test_shallow_water_geometry_config_arglist.x" )

add_test( NAME shallow_water_geometry_config_nlfile
          COMMAND bash -c "${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 1 ${MPIEXEC_PREFLAGS} ./test_shallow_water_geometry_config_nlfile.x" )

add_test( NAME shallow_water_geometry_config_nlunit
          COMMAND bash -c "${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 1 ${MPIEXEC_PREFLAGS} ./test_shallow_water_geometry_config_nlunit.x" )

#######################################
## Shallow water model config tests
#######################################
add_test( NAME shallow_water_model_config_arglist
          COMMAND bash -c "${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 1 ${MPIEXEC_PREFLAGS} ./test_shallow_water_model_config_arglist.x" )

add_test( NAME shallow_water_model_config_nlfile
          COMMAND bash -c "${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 1 ${MPIEXEC_PREFLAGS} ./test_shallow_water_model_config_nlfile.x" )

add_test( NAME shallow_water_model_config_nlunit
          COMMAND bash -c "${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 1 ${MPIEXEC_PREFLAGS} ./test_shallow_water_model_config_nlunit.x" )

#######################################
## Shallow water geometry tests
#######################################
add_test( NAME shallow_water_geometry_1
          COMMAND bash -c "${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 1 ${MPIEXEC_PREFLAGS} ./test_shallow_water_geometry.x" )

add_test( NAME shallow_water_geometry_2
          COMMAND bash -c "${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} ./test_shallow_water_geometry.x" )

add_test( NAME shallow_water_geometry_4
          COMMAND bash -c "${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS} ./test_shallow_water_geometry.x" )

add_test( NAME shallow_water_geometry_9
          COMMAND bash -c "${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 9 ${MPIEXEC_PREFLAGS} ./test_shallow_water_geometry.x" )

#######################################
## Shallow water state tests
#######################################
add_test( NAME shallow_water_state_init_default
          COMMAND bash -c "${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 1 ${MPIEXEC_PREFLAGS} ./test_shallow_water_state_init_default.x" )

add_test( NAME shallow_water_state_init_optional
          COMMAND bash -c "${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 1 ${MPIEXEC_PREFLAGS} ./test_shallow_water_state_init_optional.x" )

add_test( NAME shallow_water_state_halo_1
          COMMAND bash -c "${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 1 ${MPIEXEC_PREFLAGS} ./test_shallow_water_state_halo.x" )

add_test( NAME shallow_water_state_halo_2
          COMMAND bash -c "${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} ./test_shallow_water_state_halo.x" )

add_test( NAME shallow_water_state_halo_4
          COMMAND bash -c "${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS} ./test_shallow_water_state_halo.x" )

add_test( NAME shallow_water_state_halo_9
          COMMAND bash -c "${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 9 ${MPIEXEC_PREFLAGS} ./test_shallow_water_state_halo.x" )

add_test( NAME shallow_water_state_read_1
          COMMAND bash -c "${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 1 ${MPIEXEC_PREFLAGS} ./test_shallow_water_state_read.x" )

add_test( NAME shallow_water_state_read_2
          COMMAND bash -c "${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} ./test_shallow_water_state_read.x" )

add_test( NAME shallow_water_state_read_4
          COMMAND bash -c "${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS} ./test_shallow_water_state_read.x" )

add_test( NAME shallow_water_state_read_9
          COMMAND bash -c "${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 9 ${MPIEXEC_PREFLAGS} ./test_shallow_water_state_read.x" )

add_test( NAME shallow_water_state_write_1
          COMMAND bash -c "${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 1 ${MPIEXEC_PREFLAGS} ./test_shallow_water_state_write.x" )

add_test( NAME shallow_water_state_write_2
          COMMAND bash -c "${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} ./test_shallow_water_state_write.x" )

add_test( NAME shallow_water_state_write_4
          COMMAND bash -c "${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS} ./test_shallow_water_state_write.x" )

add_test( NAME shallow_water_state_write_9
          COMMAND bash -c "${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 9 ${MPIEXEC_PREFLAGS} ./test_shallow_water_state_write.x" )

#######################################
## Shallow water model tests
#######################################
add_test( NAME shallow_water_model_init
          COMMAND bash -c "${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 1 ${MPIEXEC_PREFLAGS} ./test_shallow_water_model_init.x" )

add_test( NAME shallow_water_model_adv_nsteps_1
          COMMAND bash -c "${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 1 ${MPIEXEC_PREFLAGS} ./test_shallow_water_model_adv_nsteps.x" )

add_test( NAME shallow_water_model_adv_nsteps_2
          COMMAND bash -c "${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} ./test_shallow_water_model_adv_nsteps.x" )

add_test( NAME shallow_water_model_adv_nsteps_4
          COMMAND bash -c "${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS} ./test_shallow_water_model_adv_nsteps.x" )

add_test( NAME shallow_water_model_adv_nsteps_9
          COMMAND bash -c "${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 9 ${MPIEXEC_PREFLAGS} ./test_shallow_water_model_adv_nsteps.x" )

add_test( NAME shallow_water_model_regression_1
          COMMAND bash -c "${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 1 ${MPIEXEC_PREFLAGS} ./test_shallow_water_model_regression.x" )

add_test( NAME shallow_water_model_regression_2
          COMMAND bash -c "${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} ./test_shallow_water_model_regression.x" )

add_test( NAME shallow_water_model_regression_4
          COMMAND bash -c "${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS} ./test_shallow_water_model_regression.x" )

add_test( NAME shallow_water_model_regression_9
          COMMAND bash -c "${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 9 ${MPIEXEC_PREFLAGS} ./test_shallow_water_model_regression.x" )

#######################################
## Shallow water tangent linear tests
#######################################
add_test( NAME shallow_water_tl_init
          COMMAND bash -c "${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 1 ${MPIEXEC_PREFLAGS} ./test_shallow_water_tl_init.x" )

add_test( NAME shallow_water_tl_adv_nsteps_1
          COMMAND bash -c "${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 1 ${MPIEXEC_PREFLAGS} ./test_shallow_water_tl_adv_nsteps.x" )

add_test( NAME shallow_water_tl_adv_nsteps_2
          COMMAND bash -c "${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} ./test_shallow_water_tl_adv_nsteps.x" )

add_test( NAME shallow_water_tl_adv_nsteps_4
          COMMAND bash -c "${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS} ./test_shallow_water_tl_adv_nsteps.x" )

add_test( NAME shallow_water_tl_adv_nsteps_9
          COMMAND bash -c "${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 9 ${MPIEXEC_PREFLAGS} ./test_shallow_water_tl_adv_nsteps.x" )

#######################################
## Shallow water adjoint tests
#######################################
add_test( NAME shallow_water_adj_init
          COMMAND bash -c "${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 1 ${MPIEXEC_PREFLAGS} ./test_shallow_water_adj_init.x" )

add_test( NAME shallow_water_adj_adv_nsteps_1
          COMMAND bash -c "${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 1 ${MPIEXEC_PREFLAGS} ./test_shallow_water_adj_adv_nsteps.x" )

add_test( NAME shallow_water_adj_adv_nsteps_2
          COMMAND bash -c "${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} ./test_shallow_water_adj_adv_nsteps.x" )

add_test( NAME shallow_water_adj_adv_nsteps_4
          COMMAND bash -c "${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS} ./test_shallow_water_adj_adv_nsteps.x" )

add_test( NAME shallow_water_adj_adv_nsteps_9
          COMMAND bash -c "${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 9 ${MPIEXEC_PREFLAGS} ./test_shallow_water_adj_adv_nsteps.x" )
