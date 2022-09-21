&geometry_parm
 nx = 7,
 ny = 7,
 xmax = 100000.0,
 ymax = 100000.0,
/

&model_parm
 dt = 2.0,
 u0 = 0.0,
 v0 = 0.0,
 b0 = 0.0,
 h0 = 5030.0,
/

&runtime
 start_step = 0,
 run_steps = 100,
 output_interval_steps = 100,
 io_format = 'NETCDF',
/
