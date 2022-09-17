import os
import sys
from telnetlib import FORWARD_X
sys.path.append(os.environ.get('SERIALBOX_ROOT') + '/python')
import serialbox as ser
import numpy as np
import gt4py.gtscript as gtscript
import gt4py.storage as gt_storage

SENA_SHALLOW_WATER = os.environ.get('SENA_SHALLOW_WATER')

Field = gtscript.Field[np.float64]
backend = "numpy"
F_TYPE = np.float64
I_TYPE = np.int32
FloatFieldIJ = gtscript.Field[gtscript.IJ, F_TYPE]

# Constant declared in module_shallow_water_geometry
g  = 9.81

# Define update boundaries stencil 
@gtscript.stencil(backend=backend)
def update_boundaries(north :  I_TYPE,
                      south :  I_TYPE,
                      east  :  I_TYPE,
                      west  :  I_TYPE,
                      u     :  FloatFieldIJ,
                      v     :  FloatFieldIJ,
                      h     :  FloatFieldIJ,
                      u_new :  FloatFieldIJ,
                      v_new :  FloatFieldIJ,
                      h_new :  FloatFieldIJ):
    with computation(FORWARD), interval(...):
        # Update southern boundary
        with horizontal(region[:, J[0]]):
            if (south == -1): 
                h_new =  h[0,1]
                u_new =  u[0,1]
                v_new = -v[0,1]

        # Update northern boundary
        with horizontal(region[:, J[-1]]):
            if (north == -1):
                h_new =  h[0,-1] 
                u_new =  u[0,-1] 
                v_new = -v[0,-1] 

        # Update western boundary
        with horizontal(region[I[0], :]):
            if (west == -1):
                h_new =  h[1,0]
                u_new = -u[1,0]
                v_new =  v[1,0]
        
        # Update eastern boundary
        with horizontal(region[I[-1], :]):
            if (east == -1):
                h_new =  h[-1,0]
                u_new = -u[-1,0]
                v_new =  v[-1,0]

# Define update interior compute stencil 
@gtscript.stencil(backend=backend)
def update_interior_model(
                          u:     FloatFieldIJ,
                          v:     FloatFieldIJ,
                          h:     FloatFieldIJ,
                          b:     FloatFieldIJ,
                          u_new: FloatFieldIJ,
                          v_new: FloatFieldIJ,
                          h_new: FloatFieldIJ,
                          dtdx:  F_TYPE,
                          dtdy:  F_TYPE,
                          g:     F_TYPE):

    with computation(FORWARD), interval(...):
        u_new = ((u[1,0] + u[-1,0] + u[0,1] + u[0,-1]) / 4.0)                    \
                        - 0.5 * dtdx * ((u[1,0]**2) / 2.0 - (u[-1,0]**2) / 2.0)  \
                        - 0.5 * dtdy * (v[0,0]) * (u[0,1] - u[0,-1])             \
                        - 0.5 * g * dtdx * (h[1,0] - h[-1,0])
        v_new = ((v[1,0] + v[-1,0] + v[0,1] + v[0,-1]) / 4.0)                    \
                        - 0.5 * dtdy * ((v[0,1]**2) / 2.0 - (v[0,1]**2) / 2.0)   \
                        - 0.5 * dtdx * (u[0,0]) * (v[1,0] - v[-1,0])             \
                        - 0.5 * g * dtdy * (h[0,1] - h[0,-1])
        h_new = ((h[1,0] + h[-1,0] + h[0,1] + h[0,-1]) / 4.0)                                \
                        - 0.5 * dtdx * (u[0,0]) * ((h[1,0] - b[1,0]) - (h[-1,0] - b[-1,0]))  \
                        - 0.5 * dtdy * (v[0,0]) * ((h[0,1] - b[0,1]) - (h[0,-1] - b[0,-1]))  \
                        - 0.5 * dtdx * (h[0,0] - b[0,0]) * (u[1,0] - u[-1,0])                \
                        - 0.5 * dtdy * (h[0,0] - b[0,0]) * (v[0,1] - v[0,-1])

# serializer = ser.Serializer(ser.OpenModeKind.Read, f'{SENA_SHALLOW_WATER}/exe/serialbox_data', 'update_model')
serializer = ser.Serializer(ser.OpenModeKind.Read, f'{SENA_SHALLOW_WATER}/exe/serialbox_data', 'shallow_water')

sp_IN = serializer.get_savepoint('update_interior-IN')

local_u = serializer.read('local_u', sp_IN[0]) 
local_v = serializer.read('local_v', sp_IN[0]) 
local_h = serializer.read('local_h', sp_IN[0])
local_b = serializer.read('local_b', sp_IN[0]) 

sp_BOUND = serializer.get_savepoint('update_boundaries-IN')

north = serializer.read('north', sp_BOUND[0])[0]
south = serializer.read('south', sp_BOUND[0])[0]
west  = serializer.read('west',  sp_BOUND[0])[0]
east  = serializer.read('east',  sp_BOUND[0])[0]

# nhalo serialized from s_w_model.f90 as xts (start index of interior point)-xps (start index of grid "patch")
nhalo = serializer.read('nhalo', sp_IN[0])[0]
dtdx = serializer.read('dtdx', sp_IN[0])[0]
dtdy = serializer.read('dtdy', sp_IN[0])[0]


# Get the full shape of one array (local_u, local_v, local_h, local_b all are dimensioned the same) and set the nx, ny, nz dimensions.
full_shape = local_u.shape
nx = full_shape[0] - 2 * nhalo
ny = full_shape[1] - 2 * nhalo

print(full_shape)

origin = (nhalo, nhalo)
domain = (nx,ny)

print(domain)

# Allocate storages 
u_gt = gt_storage.from_array(local_u, 
                             backend=backend, 
                             dtype=np.float64,
                             default_origin=(1,1),
                             shape=full_shape)
v_gt = gt_storage.from_array(local_v, 
                             backend=backend, 
                             dtype=np.float64,
                             default_origin=(1,1),
                             shape=full_shape)
h_gt = gt_storage.from_array(local_h, 
                             backend=backend, 
                             dtype=np.float64,
                             default_origin=(1,1),
                             shape=full_shape)
b_gt = gt_storage.from_array(local_b, 
                             backend=backend, 
                             dtype=np.float64,
                             default_origin=(1,1),
                             shape=full_shape)

u_new_gt = gt_storage.zeros(backend=backend, 
                            dtype=F_TYPE, 
                            shape=full_shape,
                            default_origin=origin)
v_new_gt = gt_storage.zeros(backend=backend, 
                            dtype=F_TYPE, 
                            shape=full_shape,
                            default_origin=origin)
h_new_gt = gt_storage.zeros(backend=backend, 
                            dtype=F_TYPE, 
                            shape=full_shape,
                            default_origin=origin)

sp_tsunami_pulse_IN = serializer.get_savepoint('tsunami_pulse-IN')

nsteps = serializer.read('run_steps', sp_tsunami_pulse_IN[0])[0]

# Compute 
for step in range(nsteps):
    update_interior_model(u=u_gt,
                          v=v_gt,
                          h=h_gt,
                          b=b_gt,
                          u_new=u_new_gt,
                          v_new=v_new_gt,
                          h_new=h_new_gt,
                          dtdx=dtdx,
                          dtdy=dtdy,
                          g=g,
                        #   domain=domain,
                          origin=origin)

    update_boundaries(south=south,
                      north=north,
                      west=west,
                      east=east,
                      u=u_gt,
                      v=v_gt,
                      h=h_gt,
                      u_new=u_new_gt,
                      v_new=v_new_gt,
                      h_new=h_new_gt,
                      origin=(0,0))


sp_OUT = serializer.get_savepoint('update_interior-OUT')

u_new_out = serializer.read('u_new_out_interior', sp_OUT[0]) 
v_new_out = serializer.read('v_new_out_interior', sp_OUT[0]) 
h_new_out = serializer.read('h_new_out_interior', sp_OUT[0])

# Compare answers - only the interior points (non-halo) which are computed 
try:
    assert np.array_equal(u_new_out, u_new_gt), "** u_new does not match!"
    assert np.array_equal(v_new_out, v_new_gt), "** v_new does not match!"
    assert np.array_equal(h_new_out, h_new_gt), "** h_new does not match!"
except AssertionError as msg:
    print(msg)
    
print("Finished running comparison tests!")
