import os
import sys
sys.path.append(os.environ.get('SERIALBOX_ROOT') + '/python')
import serialbox as ser
import numpy as np
import gt4py.gtscript as gtscript
import gt4py.storage as gt_storage

SENA_SHALLOW_WATER = os.environ.get('SENA_SHALLOW_WATER')

backend = "numpy"
F_TYPE = np.float64
I_TYPE = np.int32
FloatFieldIJ = gtscript.Field[gtscript.IJ, F_TYPE]

g  = 9.81

@gtscript.stencil(backend=backend)
def update_boundaries_example(north :  I_TYPE,
                              south :  I_TYPE,
                              west  :  I_TYPE,
                              east  :  I_TYPE,
                              u     :  FloatFieldIJ,
                              v     :  FloatFieldIJ,
                              h     :  FloatFieldIJ,
                              u_new :  FloatFieldIJ,
                              v_new :  FloatFieldIJ,
                              h_new :  FloatFieldIJ):
    with computation(FORWARD), interval(...):
        # Update southern boundary
        with horizontal(region[:, J[0]]):
            if south == -1: 
                h_new = 1.
                v_new = 1.
                u_new = 1.
        
        # # Update northern boundary
        with horizontal(region[:, J[-1]]):
            if north == -1:
                h_new = 2.
                u_new = 2.
                v_new = 2.

        # Update western boundary
        with horizontal(region[I[0], :]):
            if west == -1:
                h_new = 3.
                u_new = 3.
                v_new = 3.
        
        # Update eastern boundary
        with horizontal(region[I[-1], :]):
            if east == -1:
                h_new = 4.
                u_new = 4.
                v_new = 4.

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


serializer = ser.Serializer(ser.OpenModeKind.Read, f'{SENA_SHALLOW_WATER}/exe/serialbox_data', 'shallow_water')

sp_IN = serializer.get_savepoint('update_boundaries-IN')

north = serializer.read('north', sp_IN[0])[0]
south = serializer.read('south', sp_IN[0])[0]
west  = serializer.read('west',  sp_IN[0])[0]
east  = serializer.read('east',  sp_IN[0])[0]

print(type(north), type(south), east, west)

local_u = serializer.read('local_u', sp_IN[0]) 
local_v = serializer.read('local_v', sp_IN[0]) 
local_h = serializer.read('local_h', sp_IN[0])

u_new = serializer.read('u_new', sp_IN[0]) 
v_new = serializer.read('v_new', sp_IN[0]) 
h_new = serializer.read('h_new', sp_IN[0])  

# nhalo serialized from s_w_model.f90 as xts (start index of interior point)-xps (start index of grid "patch")
nhalo = serializer.read('nhalo', sp_IN[0])[0]

origin = (0, 0)

# Get the full shape of one array (local_u, local_v, local_h, local_b all are dimensioned the same) and set the nx, ny, nz dimensions.
full_shape = local_u.shape
nx = full_shape[0] - 2 * nhalo
ny = full_shape[1] - 2 * nhalo
# nz = full_shape[2]

# Standard compute domain 
domain = (nx,ny)

# Allocate storages 
u_gt = gt_storage.from_array(local_u, 
                             backend=backend, 
                             dtype=np.float64,
                             default_origin=(0,0),
                             shape=full_shape)
v_gt = gt_storage.from_array(local_v, 
                             backend=backend, 
                             dtype=np.float64,
                             default_origin=(0,0),
                             shape=full_shape)
h_gt = gt_storage.from_array(local_h, 
                             backend=backend, 
                             dtype=np.float64,
                             default_origin=(0,0),
                             shape=full_shape)

u_new_gt = gt_storage.zeros(backend=backend, 
                            dtype=F_TYPE, 
                            shape=full_shape,
                            default_origin=(0,0))
v_new_gt = gt_storage.zeros(backend=backend, 
                            dtype=F_TYPE, 
                            shape=full_shape,
                            default_origin=(0,0))
h_new_gt = gt_storage.zeros(backend=backend, 
                            dtype=F_TYPE, 
                            shape=full_shape,
                            default_origin=(0,0))

sp_tsunami_pulse_IN = serializer.get_savepoint('tsunami_pulse-IN')

nsteps = serializer.read('run_steps', sp_tsunami_pulse_IN[0])[0]

for step in range(nsteps):
    # update_boundaries_example(north=north,
    #                           south=south,
    #                           west=west,
    #                           east=east,
    #                           u=u_gt,
    #                           v=v_gt,
    #                           h=h_gt,
    #                           u_new=u_new_gt,
    #                           v_new=v_new_gt,
    #                           h_new=h_new_gt,
    #                           origin=origin,
    #                           # domain=domain
    #                           )
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
                      origin=origin,
                      )

sp_OUT = serializer.get_savepoint('update_boundaries-OUT')

u_new_out = serializer.read('u_new_out_boundaries', sp_OUT[0]) 
v_new_out = serializer.read('v_new_out_boundaries', sp_OUT[0]) 
h_new_out = serializer.read('h_new_out_boundaries', sp_OUT[0])

# print("u_new_out", u_new_out)
print("u_new_gt \n", u_new_gt)

# Compare answers - only the interior points (non-halo) which are computed 
try:
    assert np.array_equal(u_new_out, u_new_gt), "** u_new does not match!"
    assert np.array_equal(v_new_out, v_new_gt), "** v_new does not match!"
    assert np.array_equal(h_new_out, h_new_gt), "** h_new does not match!"
except AssertionError as msg:
    print(msg)
    
print("Finished running comparison tests!")