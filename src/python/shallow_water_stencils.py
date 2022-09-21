import os
import sys
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
# Use a 2D Field for the GT4Py Stencils
FloatFieldIJ = gtscript.Field[gtscript.IJ, F_TYPE]

data_path = f'{SENA_SHALLOW_WATER}/src/python/serialbox_data'

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

serializer = ser.Serializer(ser.OpenModeKind.Read, data_path, 'shallow_water')

sp_IN = serializer.get_savepoint('update_interior-IN')

local_u = serializer.read('local_u', sp_IN[0]) 
local_v = serializer.read('local_v', sp_IN[0]) 
local_h = serializer.read('local_h', sp_IN[0])
local_b = serializer.read('local_b', sp_IN[0]) 

u_new = serializer.read('u_new', sp_IN[0])
v_new = serializer.read('v_new', sp_IN[0])
h_new = serializer.read('h_new', sp_IN[0])

xps  = serializer.read('xps', sp_IN[0])[0]
xpe  = serializer.read('xpe', sp_IN[0])[0]
yps  = serializer.read('yps', sp_IN[0])[0]
ype  = serializer.read('ype', sp_IN[0])[0]

xts  = serializer.read('xts', sp_IN[0])[0]
xte  = serializer.read('xte', sp_IN[0])[0]
yts  = serializer.read('yts', sp_IN[0])[0]
yte  = serializer.read('yte', sp_IN[0])[0]

# dtdx and dtdy are computed during serialization (!$ser data dtdx=local_dt/dx dtdy=local_dt/dy) 
dtdx = serializer.read('dtdx', sp_IN[0])[0]
dtdy = serializer.read('dtdy', sp_IN[0])[0]

sp_BOUND = serializer.get_savepoint('update_boundaries-IN')

north = serializer.read('north', sp_BOUND[0])[0]
south = serializer.read('south', sp_BOUND[0])[0]
west  = serializer.read('west',  sp_BOUND[0])[0]
east  = serializer.read('east',  sp_BOUND[0])[0]

# Set up nhalo, origin, and domain (nhalo is xts - xps - start of interior points minus start of grid patch)
nhalo=xts-xps
nx = xte - xts + 1
ny = yte - yts + 1
origin=(nhalo, nhalo)
domain=(nx, ny,1)


# Allocate storages 
u_gt = gt_storage.from_array(local_u, backend=backend, dtype=np.float64,default_origin=(1,1))
v_gt = gt_storage.from_array(local_v, backend=backend, dtype=np.float64,default_origin=(1,1))
h_gt = gt_storage.from_array(local_h, backend=backend, dtype=np.float64,default_origin=(1,1))
b_gt = gt_storage.from_array(local_b, backend=backend, dtype=np.float64,default_origin=(1,1))

u_new_gt = gt_storage.from_array(u_new, backend=backend, dtype=np.float64,default_origin=(1,1))
v_new_gt = gt_storage.from_array(v_new, backend=backend, dtype=np.float64,default_origin=(1,1))
h_new_gt = gt_storage.from_array(h_new, backend=backend, dtype=np.float64,default_origin=(1,1))

# Get the N time steps from the tsunami program- shallow_water.f90
sp_tsunami_pulse_IN = serializer.get_savepoint('tsunami_pulse-IN')
nsteps = serializer.read('run_steps', sp_tsunami_pulse_IN[0])[0]

# Move model forward and compute
for step in range(nsteps):
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
                      origin=(0,0),
                      domain=(nx+2, ny+2, 1))

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
                          domain=domain,
                          origin=origin)


# Retrieve savepoint output data 
sp_OUT = serializer.get_savepoint('update_interior-OUT')

u_new_out = serializer.read('u_new_out_interior', sp_OUT[0]) 
v_new_out = serializer.read('v_new_out_interior', sp_OUT[0]) 
h_new_out = serializer.read('h_new_out_interior', sp_OUT[0])

# Compare answers (after calling both stencils)
# Comparison is only done at the end of advancing the model through the specified timesteps
try:
    assert np.array_equal(u_new_out, u_new_gt), "** u_new does not match!"
    assert np.array_equal(v_new_out, v_new_gt), "** v_new does not match!"
    assert np.array_equal(h_new_out, h_new_gt), "** h_new does not match!"
except AssertionError as msg:
    print(msg)
    
print("Finished running comparison tests!")
