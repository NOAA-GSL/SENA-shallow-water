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

# Define stencil 
#@gtscript.stencil(backend=backend)
#def update_interior_model(u     : gtscript.Field[F_TYPE],
#                          v     : gtscript.Field[F_TYPE],
#                          h     : gtscript.Field[F_TYPE],
#                          u_new : gtscript.Field[F_TYPE],
#                          v_new : gtscript.Field[F_TYPE],
#                          h_new : gtscript.Field[F_TYPE],
#                          dx    : np.float64,
#                          dy    : np.float64,
#                          dt    : np.float64 ):



#    return

serializer = ser.Serializer(ser.OpenModeKind.Read, f'{SENA_SHALLOW_WATER}/exe/serialbox_data', 'update_model')

sp_IN = serializer.get_savepoint('update_interior-IN')

xps   = serializer.read('xps',   sp_IN[0])[0]
xpe   = serializer.read('xpe',   sp_IN[0])[0]
yps   = serializer.read('yps',   sp_IN[0])[0]
ype   = serializer.read('ype',   sp_IN[0])[0]
xms   = serializer.read('xms',   sp_IN[0])[0]
xme   = serializer.read('xme',   sp_IN[0])[0]
yms   = serializer.read('yms',   sp_IN[0])[0]
yme   = serializer.read('yme',   sp_IN[0])[0]

local_u = serializer.read('local_u', sp_IN[0]) 
local_v = serializer.read('local_v', sp_IN[0]) 
local_h = serializer.read('local_h', sp_IN[0])
local_b = serializer.read('local_b', sp_IN[0]) 

u_new_in = serializer.read('u_new', sp_IN[0]) 
v_new_in = serializer.read('v_new', sp_IN[0]) 
h_new_in = serializer.read('h_new', sp_IN[0])

dx = serializer.read('dx', sp_IN[0])[0]
dy = serializer.read('dy', sp_IN[0])[0]
local_dt = serializer.read('local_dt', sp_IN[0])[0]



full_shape = v_new_in.shape


print(local_dt, full_shape, local_v, yps, ype)




# Compute 




sp_OUT = serializer.get_savepoint('update_interior-OUT')

u_new_out = serializer.read('u_new', sp_OUT[0]) 
v_new_out = serializer.read('v_new', sp_OUT[0]) 
h_new_out = serializer.read('h_new', sp_OUT[0])