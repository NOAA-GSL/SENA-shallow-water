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

# Constant declared in module_shallow_water_geometry
g  = 9.81

# Define stencil 
@gtscript.stencil(backend=backend)
def update_interior_model(
                          u:     gtscript.Field[np.float64],
                          v:     gtscript.Field[np.float64],
                          h:     gtscript.Field[np.float64],
                          b:     gtscript.Field[np.float64],
                          u_new: gtscript.Field[np.float64],
                          v_new: gtscript.Field[np.float64],
                          h_new: gtscript.Field[np.float64],
                          dtdx:  np.float64,
                          dtdy:  np.float64,
                          g:     np.float64
                        ):
    with computation(PARALLEL), interval(...):
        u_new[0,0,0] = 1.0
        # ((u[1,0,0] + u[-1,0,0] + u[0,1,0] + u[0,-1,0]) / 4.0)         \
        #                 - 0.5 * dtdx * ((u[1,0,0]**2) / 2.0 - (u[-1,0,0]**2) / 2.0)  \
        #                 - 0.5 * dtdy * (v[0,0,0]) * (u[0,1,0] - u[0,-1,0])           \
        #                 - 0.5 * g * dtdx * (h[1,0,0] - h[-1,0,0])
        v_new[0,0,0] = 1.0
        # ((v[1,0,0] + v[-1,0,0] + v[0,1,0] + v[0,-1,0]) / 4.0)         \
        #                 - 0.5 * dtdy * ((v[0,1,0]**2) / 2.0 - (v[0,1,0]**2) / 2.0)   \
        #                 - 0.5 * dtdx * (u[0,0,0]) * (v[1,0,0] - v[-1,0,0])           \
        #                 - 0.5 * g * dtdy * (h[0,1,0] - h[0,-1,0])
        h_new[0,0,0] = 1.0
        # ((h[1,0,0] + h[-1,0,0] + h[0,1,0] + h[0,-1,0]) / 4.0)                           \
        #                 - 0.5 * dtdx * (u[0,0,0]) * ((h[1,0,0] - b[1,0,0]) - (h[-1,0,0] - b[-1,0,0]))  \
        #                 - 0.5 * dtdy * (v[0,0,0]) * ((h[0,1,0] - b[0,1,0]) - (h[0,-1,0] - b[0,-1,0]))  \
        #                 - 0.5 * dtdx * (h[0,0,0] - b[0,0,0]) * (u[1,0,0] - u[-1,0,0])                  \
        #                 - 0.5 * dtdy * (h[0,0,0] - b[0,0,0]) * (v[0,1,0] - v[0,-1,0])

# serializer = ser.Serializer(ser.OpenModeKind.Read, f'{SENA_SHALLOW_WATER}/exe/serialbox_data', 'update_model')
serializer = ser.Serializer(ser.OpenModeKind.Read, f'{SENA_SHALLOW_WATER}/exe/serialbox_data', 'shallow_water')

sp_IN = serializer.get_savepoint('update_interior-IN')

local_u = serializer.read('local_u', sp_IN[0]) 
local_v = serializer.read('local_v', sp_IN[0]) 
local_h = serializer.read('local_h', sp_IN[0])
local_b = serializer.read('local_b', sp_IN[0]) 

# Add a third dimension to allow ability to run the GT4Py stencil.
local_u  = local_u[..., np.newaxis]
local_v  = local_v[..., np.newaxis]
local_h  = local_h[..., np.newaxis]
local_b  = local_b[..., np.newaxis]


# nhalo serialized from s_w_model.f90 as xts (start index of interior point)-xps (start index of grid "patch")
nhalo = serializer.read('nhalo', sp_IN[0])[0]
dtdx = serializer.read('dtdx', sp_IN[0])[0]
dtdy = serializer.read('dtdy', sp_IN[0])[0]

origin = (nhalo, nhalo, 0)

# Get the full shape of one array (local_u, local_v, local_h, local_b all are dimensioned the same) and set the nx, ny, nz dimensions.
full_shape = local_u.shape
nx = full_shape[0] - 2 * nhalo
ny = full_shape[1] - 2 * nhalo
nz = full_shape[2]
# domain = (xpe - nhalo * 2, ype - nhalo * 2, 0)
# domain = (shape[0] - 1, shape[1] - 1, shape[2])
domain = (nx,ny,nz)

# Allocate storages 
u_gt = gt_storage.from_array(local_u, 
                             backend=backend, 
                             dtype=np.float64,
                             default_origin=(0,0,0),
                             shape=full_shape)
v_gt = gt_storage.from_array(local_v, 
                             backend=backend, 
                             dtype=np.float64,
                             default_origin=(0,0,0),
                             shape=full_shape)
h_gt = gt_storage.from_array(local_h, 
                             backend=backend, 
                             dtype=np.float64,
                             default_origin=(0,0,0),
                             shape=full_shape)
b_gt = gt_storage.from_array(local_b, 
                             backend=backend, 
                             dtype=np.float64,
                             default_origin=(0,0,0),
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

# Instead of dimensioning two seperate arrays to get the i and j indices to pass to GT4Py, we'll keep this is a pure python funciton
def initialize_h(h:     np.ndarray, 
                 sigma: np.float64, 
                 xmid:  np.float64, 
                 ymid:  np.float64, 
                 h0:    np.float64, 
                 dx:    np.float64, 
                 dy:    np.float64,
                 yps:   np.int32, 
                 ype:   np.int32, 
                 xps:   np.int32, 
                 xpe:   np.int32):  
    for j in range(yps,ype):
        for i in range(xps,xpe):
            dsqr = (np.float64(i - 1) * dx - xmid)**2 + (np.float64(j - 1) * dy - ymid)**2
            h[i,j] = 5000.0 + np.exp(-dsqr / sigma**2) * (h0 - 5000.0)

sp_tsunami_pulse_IN = serializer.get_savepoint('tsunami_pulse-IN')

nsteps = serializer.read('run_steps', sp_tsunami_pulse_IN[0])[0]
yps_local = serializer.read('yps_local', sp_tsunami_pulse_IN[0])[0]
ype_local = serializer.read('ype_local', sp_tsunami_pulse_IN[0])[0]
xps_local = serializer.read('xps_local', sp_tsunami_pulse_IN[0])[0]
xpe_local = serializer.read('xpe_local', sp_tsunami_pulse_IN[0])[0]

sigma = serializer.read('sigma', sp_tsunami_pulse_IN[0])[0]
xmid = serializer.read('xmid', sp_tsunami_pulse_IN[0])[0]
ymid = serializer.read('ymid', sp_tsunami_pulse_IN[0])[0]
h0 = serializer.read('h0', sp_tsunami_pulse_IN[0])[0]
dx = serializer.read('dx', sp_tsunami_pulse_IN[0])[0]
dy = serializer.read('dy', sp_tsunami_pulse_IN[0])[0]


# print(type(h0), "sigma type", type(sigma), "yps_local type", type(yps_local), type(xmid), "dx type", type(dx), ymid, nsteps)

# print("h_gt before initialize", local_h[:,:,0])

# local_h = local_h[:,:,0]

# initialize_h(h=local_h,
#              sigma=sigma,
#              xmid=xmid, 
#              ymid=ymid, 
#              h0=h0, 
#              dx=dx,
#              dy=dy,
#              yps=yps_local,
#              ype=ype_local,
#              xps=xps_local,
#              xpe=xpe_local )

# print("h_gt after initialize", local_h)

sp_tsunami_pulse_OUT = serializer.get_savepoint('tsunami_pulse-OUT')

# local_h = local_h[:,:,0]


# h_out = serializer.read('h', sp_tsunami_pulse_OUT[0])

# if np.allclose(local_h,h_out):
#     print("Solution is valid!")
#     print('Max error:', np.max(np.abs(local_h - h_out)))
# else:
#     raise Exception("Solution is NOT valid")

# try:
#     # assert np.array_equal(local_h, h_out), "** h does not match!"
#     assert np.array_equal(local_h, h_out), "** h does not match!"
# except AssertionError as msg:
#     print(msg)

# print(local_h.shape, h_out.shape)

# # Compute 
for step in range(nsteps):
    update_interior_model(
                          u=u_gt,
                          v=v_gt,
                          h=h_gt,
                          b=b_gt,
                          u_new=u_new_gt,
                          v_new=v_new_gt,
                          h_new=h_new_gt,
                          dtdx=dtdx,
                          dtdy=dtdy,
                          g=g,
                          origin=origin,
                          domain=domain)

# # Remove 3rd dimension of gt arrays to compare to original Fortran 2d output arrays
u_new_gt = u_new_gt[:,:,0]
v_new_gt = v_new_gt[:,:,0]
h_new_gt = h_new_gt[:,:,0]

sp_OUT = serializer.get_savepoint('update_interior-OUT')

u_new_out = serializer.read('u_new', sp_OUT[0]) 
v_new_out = serializer.read('v_new', sp_OUT[0]) 
h_new_out = serializer.read('h_new', sp_OUT[0])

print("u_new_gt", u_new_gt)
print("u_new_out", u_new_out)

# Compare answers
try:
    # assert np.array_equal(u_new_out, u_new_gt), "** u_new does not match!"
    assert np.array_equal(v_new_out, v_new_gt), "** v_new does not match!"
    assert np.array_equal(h_new_out, h_new_gt), "** h_new does not match!"
except AssertionError as msg:
    print(msg)
    
print("Finished running comparison tests!")
