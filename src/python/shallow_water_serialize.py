#!/usr/bin/env python3                                                                                                                                                                                                                                                                                                          

import numpy as np
import sys
import os                                                                                                                                                                                                                                                     
from mpi4py import MPI
import yaml
import serialbox as ser
import gt4py.gtscript as gtscript
import gt4py.storage as gt_storage

from shallow_water_geometry_config import ShallowWaterGeometryConfig
from shallow_water_geometry import ShallowWaterGeometry
from shallow_water_gt4py_config import ShallowWaterGT4PyConfig
from shallow_water_model_config import ShallowWaterModelConfig
from shallow_water_state import ShallowWaterState
from shallow_water_model import ShallowWaterModel

# Gather all of the Serialbox data 
data_path = './serialbox_data'
if not os.path.isdir(data_path):
    raise Exception('Data directory does not exist', data_path)
serializer = ser.Serializer(ser.OpenModeKind.Read, data_path, 'shallow_water')

input_sp = serializer.get_savepoint('input_data')
u_in  = serializer.read('u',  input_sp[0])
v_in  = serializer.read('v',  input_sp[0])
h_in  = serializer.read('h',  input_sp[0])

output_sp = serializer.get_savepoint('output_data')
u_out  = serializer.read('u',  output_sp[0])
v_out  = serializer.read('v',  output_sp[0])
h_out  = serializer.read('h',  output_sp[0])

# Set the config file
config_file = "../../parm/shallow_water.yml"

# Initialize and Run the Python model 
comm = MPI.COMM_WORLD

# Create the geometry
gc = ShallowWaterGeometryConfig.from_YAML_filename(config_file)
g = ShallowWaterGeometry(gc, comm)

# Create the model
mc = ShallowWaterModelConfig.from_YAML_filename(config_file)
gtc = ShallowWaterGT4PyConfig.from_YAML_filename(config_file)
m = ShallowWaterModel(mc, gtc, g)

# Read the configuration settings from config file
with open(config_file, "r") as yamlFile: 
    try:
        config = yaml.full_load(yamlFile)
    except yaml.YAMLError as e:
        print(e)
runtime = config['runtime']

# Create a state with a tsunami pulse in it to initialize field h
h = np.empty((g.npx, g.npy), dtype=float)
xmid = g.xmax / 2.0
ymid = g.ymax / 2.0
sigma = np.floor(g.xmax / 20.0)
for i in range(g.xps, g.xpe + 1):
    for j in range(g.yps, g.ype + 1):
        dsqr = ((i-1) * g.dx - xmid)**2 + ((j-1) * g.dy - ymid)**2
        h[i - g.xps,j - g.yps] = 5000.0 + np.exp(-dsqr / sigma**2) * (mc.h0 - 5000.0)
s = ShallowWaterState(g, gtc, h=h)

# Initialize input variables 
u_in_gt  = np.zeros((g.npx, g.npy), dtype=float)
v_in_gt  = np.zeros((g.npx, g.npy), dtype=float)
h_in_gt  = np.zeros((g.npx, g.npy), dtype=float)

# Write out the initial state
if (runtime['output_interval_steps'] <= runtime['run_steps']):
    s.write(f"python_swout_{round(s.clock / m.dt):07d}.nc")

# Run the model
for t in range(0, runtime['run_steps'], runtime['output_interval_steps']):

    if(t == 0):
        u_in_gt[:,:] = s.get_u()
        v_in_gt[:,:] = s.get_v()
        h_in_gt[:,:] = s.get_h()

    m.adv_nsteps(s, min(runtime['output_interval_steps'], runtime['run_steps'] - t))

    # Gather full u, v, and h
    u_out_gt, v_out_gt, h_out_gt = s.gather()
    
    # Set the output arrays, transpose to match Fortran's column-major from Python's row-major 
    if (t == runtime['run_steps']):
        u_out_gt[:,:] = np.transpose(u_out_gt[:,:])
        v_out_gt[:,:] = np.transpose(v_out_gt[:,:])
        h_out_gt[:,:] = np.transpose(h_out_gt[:,:])

    if (runtime['output_interval_steps']):
        s.write(f"python_swout_{round(s.clock / m.dt):07d}.nc")

# Compare the input and output arrays, using an absolute tolerance of 1e-23. 
# The tolerance value can be omitted, and the comparison test will still pass. 
try:
    assert np.allclose(u_in, u_in_gt, atol=1e-23),   "** u input does not match!"
    assert np.allclose(v_in, v_in_gt, atol=1e-23),   "** v input does not match!"
    assert np.allclose(h_in, h_in_gt, atol=1e-23),   "** h input does not match!"
    assert np.allclose(u_out, u_out_gt, atol=1e-23), "** u output does not match!"
    assert np.allclose(v_out, v_out_gt, atol=1e-23), "** v output does not match!"
    assert np.allclose(h_out, h_out_gt, atol=1e-23), "** h output does not match!"
except AssertionError as msg:
    print(msg)
else:
    print("Finished running comparison tests, all PASSED!")
