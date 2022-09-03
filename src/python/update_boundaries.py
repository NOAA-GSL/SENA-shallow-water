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

#@gtscript.stencil(backend=backend)
# def update_boundaries_model(north :  ,
#                             south :  ,
#                             west  :  ,
#                             east  :  ,
#                             u     :  ,
#                             v     :  ,
#                             h     :  ,
#                             u_new :  ,
#                             v_new :  ,
#                             h_new :  ):

#   return

serializer = ser.Serializer(ser.OpenModeKind.Read, f'{SENA_SHALLOW_WATER}/exe/serialbox_data', 'update_model')

sp = serializer.get_savepoint('update_boundaries-IN')

xps   = serializer.read('xps', sp[0])[0]
xpe   = serializer.read('xpe', sp[0])[0]
yps   = serializer.read('yps', sp[0])[0]
ype   = serializer.read('ype', sp[0])[0]
xms   = serializer.read('xms', sp[0])[0]
xme   = serializer.read('xme', sp[0])[0]
yms   = serializer.read('yms', sp[0])[0]
yme   = serializer.read('yme', sp[0])[0]
north = serializer.read('north', sp[0])[0]
south = serializer.read('south', sp[0])[0]
west  = serializer.read('west', sp[0])[0]
east  = serializer.read('east', sp[0])[0]

u_new = serializer.read('u_new', sp[0]) 
v_new = serializer.read('v_new', sp[0]) 
h_new = serializer.read('h_new', sp[0])  

test_r8kind_python_type  = serializer.read('test_r8kind_python_type', sp[0])[0]

full_shape = v_new.shape


print(xps, xpe, yps, ype)
