#!/usr/bin/env python3

import numpy as np
import sys
import os
#sys.path.append(os.environ.get('SERIALBOX_ROOT') + '/python')
import serialbox as ser
import gt4py.gtscript as gtscript
import gt4py.storage as gt_storage
#from gt4py.gtscript import stencil, function, PARALLEL, computation, interval, Field, parallel, region, I, J, K

backend="numpy"
F_TYPE = np.float64
I_TYPE = np.int32
FloatFieldIJ = gtscript.Field[gtscript.IJ, F_TYPE]

# Path to the serialbox data
data_path = './serialbox_data'

# Define constants
g = 9.81

@gtscript.stencil(backend=backend)
def interior_update(u     : FloatFieldIJ,
                    v     : FloatFieldIJ,
                    h     : FloatFieldIJ,
                    b     : FloatFieldIJ,
                    u_new : FloatFieldIJ,
                    v_new : FloatFieldIJ,
                    h_new : FloatFieldIJ,
                    dtdx  : F_TYPE,
                    dtdy  : F_TYPE):
    # NOTE: FORWARD ordering is required here to disambiguate the missing k dimension
    #       for assignment into our 2D arrays.
    with computation(FORWARD), interval(...):
        u_new = ((u[1,0] + u[-1,0] + u[0,1] + u[0,-1]) / 4.0)           \
                - 0.5 * dtdx * ((u[1,0]**2) / 2.0 - (u[-1,0]**2) / 2.0) \
                - 0.5 * dtdy * v * (u[0,1] - u[0,-1])                   \
                - 0.5 * g * dtdx * (h[1,0] - h[-1,0])

        v_new = ((v[1,0] + v[-1,0] + v[0,1] + v[0,-1]) / 4.0)           \
                - 0.5 * dtdy * ((v[0,1]**2) / 2.0 - (v[0,1]**2) / 2.0)  \
                - 0.5 * dtdx * u * (v[1,0] - v[-1,0])                   \
                - 0.5 * g * dtdy * (h[0,1] - h[0,-1])
  
        h_new = ((h[1,0] + h[-1,0] + h[0,1] + h[0,-1]) / 4.0)                \
                - 0.5 * dtdx * u * ((h[1,0] - b[1,0]) - (h[-1,0] - b[-1,0])) \
                - 0.5 * dtdy * v * ((h[0,1] - b[0,1]) - (h[0,-1] - b[0,-1])) \
                - 0.5 * dtdx * (h - b) * (u[1,0] - u[-1,0])                  \
                - 0.5 * dtdy * (h - b) * (v[0,1] - v[0,-1])


@gtscript.stencil(backend=backend)
def boundary_update(u     : FloatFieldIJ,
                    v     : FloatFieldIJ,
                    h     : FloatFieldIJ,
                    u_new : FloatFieldIJ,
                    v_new : FloatFieldIJ,
                    h_new : FloatFieldIJ,
                    north : I_TYPE,
                    south : I_TYPE,
                    west  : I_TYPE,
                    east  : I_TYPE,
                    dtdx  : F_TYPE,
                    dtdy  : F_TYPE):
    # NOTE: FORWARD ordering is required here to disambiguate the missing k dimension
    #       for assignment into our 2D arrays.
    # NOTE: Requires origin/domain set such that I[0], I[-1], J[0], J[-1] are the
    #       first/last elements of the compute domain (which they are in the original
    #       Fortran).
    with computation(FORWARD), interval(...):
        # Update southern boundary if there is one
        with horizontal(region[:, J[0]]):
            if (south == -1):
                h_new = h[0, 1]
                u_new = u[0, 1]
                v_new = -v[0, 1]

        # Update northern boundary if there is one
        with horizontal(region[:, J[-1]]):
            if (north == -1):
                h_new = h[0,-1]
                u_new = u[0,-1]
                v_new = -v[0,-1]

        # Update western boundary if there is one
        with horizontal(region[I[0], :]):
            if (west == -1):
                h_new = h[1,0]
                u_new = -u[1,0]
                v_new = v[1,0]

        # Update eastern boundary if there is one
        with horizontal(region[I[-1], :]):
            if (east == -1):
                h_new = h[-1,0]
                u_new = -u[-1,0]
                v_new = v[-1,0]

# Instantiate the serializer
if not os.path.isdir(data_path):
    raise Exception('Data directory does not exist', data_path)
serializer = ser.Serializer(ser.OpenModeKind.Read, data_path, 'shallow_water')

# Get savepoint for the input data
input_sp = serializer.get_savepoint('input_data')

# Set savepoint for the boundaries output data
boundaries_output_sp = serializer.get_savepoint('boundaries_output')

# Set savepoint for the boundaries output data
interior_output_sp = serializer.get_savepoint('interior_output')

# Extract the input data dimensions
xps  = serializer.read('xps',  input_sp[0])[0]
xpe  = serializer.read('xpe',  input_sp[0])[0]
yps  = serializer.read('yps',  input_sp[0])[0]
ype  = serializer.read('ype',  input_sp[0])[0]

xms  = serializer.read('xms',  input_sp[0])[0]
xme  = serializer.read('xme',  input_sp[0])[0]
yms  = serializer.read('yms',  input_sp[0])[0]
yme  = serializer.read('yme',  input_sp[0])[0]

xts  = serializer.read('xts',  input_sp[0])[0]
xte  = serializer.read('xte',  input_sp[0])[0]
yts  = serializer.read('yts',  input_sp[0])[0]
yte  = serializer.read('yte',  input_sp[0])[0]

print(xps, xpe, yps, ype)
print(xms, xme, yms, yme)
print(xts, xte, yts, yte)

# Extract the MPI ranks of the neighbors
north  = serializer.read('north',  input_sp[0])[0]
south  = serializer.read('south',  input_sp[0])[0]
west  = serializer.read('west',  input_sp[0])[0]
east  = serializer.read('east',  input_sp[0])[0]

print(north, south, west, east)

# Extract grid spacing and time step
dx  = serializer.read('dx',  input_sp[0])[0]
dy  = serializer.read('dy',  input_sp[0])[0]
dt  = serializer.read('dt',  input_sp[0])[0]

print(dx, dy, dt)

# Extract model state variable inputs
u = serializer.read('u',  input_sp[0])
v = serializer.read('v',  input_sp[0])
h = serializer.read('h',  input_sp[0])
b = serializer.read('b',  input_sp[0])

print(u.shape, v.shape, h.shape, b.shape)

# Extract model state variable input to interior stencil
u_new = serializer.read('u_new',  input_sp[0])
v_new = serializer.read('v_new',  input_sp[0])
h_new = serializer.read('h_new',  input_sp[0])

print(u_new.shape, v_new.shape, h_new.shape)

# Compute dxdt and dydt
dtdx = dt / dx
dtdy = dt / dy

print(dtdx, dtdy)

# Set up halo, origin, and domain
nhalo=1
nx = xte - xts + 1
ny = yte - yts + 1
origin=(nhalo, nhalo)
domain=(nx, ny, 1)

print(origin)
print(domain)

# Create storages for the model state
u_gt = gt_storage.from_array(u, backend=backend, default_origin=origin)
v_gt = gt_storage.from_array(v, backend=backend, default_origin=origin)
h_gt = gt_storage.from_array(h, backend=backend, default_origin=origin)
b_gt = gt_storage.from_array(b, backend=backend, default_origin=origin)
u_new_gt = gt_storage.from_array(u_new, backend=backend, default_origin=origin)
v_new_gt = gt_storage.from_array(v_new, backend=backend, default_origin=origin)
h_new_gt = gt_storage.from_array(h_new, backend=backend, default_origin=origin)

# Get new boundaries
boundary_update(u=u_gt,
                v=v_gt,
                h=h_gt,
                u_new=u_new_gt,
                v_new=v_new_gt,
                h_new=h_new_gt,
                north=north,
                south=south,
                west=west,
                east=east,
                dtdx=dtdx,
                dtdy=dtdy,
                origin=(0,0,0),
                domain=(nx+2, ny+2, 1))

# Extract model state variable input to interior stencil
u_new_ref = serializer.read('u_new',  boundaries_output_sp[0])
v_new_ref = serializer.read('v_new',  boundaries_output_sp[0])
h_new_ref = serializer.read('h_new',  boundaries_output_sp[0])

try:
    assert np.array_equal(u_new_ref, u_new_gt), "boundary_update: u does not match!"
    assert np.array_equal(v_new_ref, v_new_gt), "boundary_update: v does not match!"
    assert np.array_equal(h_new_ref, h_new_gt), "boundary_update: h does not match!"
except AssertionError as msg:
    print(msg)

#u_new_gt = gt_storage.from_array(u_new_ref, backend=backend, default_origin=origin)
#v_new_gt = gt_storage.from_array(v_new_ref, backend=backend, default_origin=origin)
#h_new_gt = gt_storage.from_array(h_new_ref, backend=backend, default_origin=origin)

# Get new interior points
interior_update(u=u_gt,
                v=v_gt,
                h=h_gt,
                b=b_gt,
                u_new=u_new_gt,
                v_new=v_new_gt,
                h_new=h_new_gt,
                dtdx=dtdx,
                dtdy=dtdy,
                origin=origin,
                domain=domain)

# Extract model state variable output from interior stencil
u_new_ref = serializer.read('u_new',  interior_output_sp[0])
v_new_ref = serializer.read('v_new',  interior_output_sp[0])
h_new_ref = serializer.read('h_new',  interior_output_sp[0])

try:
    assert np.array_equal(u_new_ref, u_new_gt), "interior_update: u does not match!"
    assert np.array_equal(v_new_ref, v_new_gt), "interior_update: v does not match!"
    assert np.array_equal(h_new_ref, h_new_gt), "interior_update: h does not match!"
except AssertionError as msg:
    print(msg)
