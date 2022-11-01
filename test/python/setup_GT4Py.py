#################################
#
# Test numpy and matplotlib setup
#
#################################

import numpy as np
import matplotlib.pyplot as plt


#plt.imshow(np.random.randn(128, 128))
plt.imsave('./image.png', np.random.randn(128, 128))

#################################
#
# Test gt4py setup
#
#################################
import os
import numpy as np
import gt4py
from gt4py import gtscript

@gtscript.stencil(backend="numpy")
def copy_stencil(in_storage: gtscript.Field[float], out_storage: gtscript.Field[float]):
    with computation(FORWARD):
        with interval(...):
            out_storage = in_storage

shape = (1, 10, 10)
in_storage = gt4py.storage.ones(
    shape=shape, default_origin=(0, 0, 0), dtype=float, backend="numpy"
)
in_storage.data[:, 2:7, :] = 2
out_storage = gt4py.storage.zeros(
    shape=shape, default_origin=(0, 0, 0), dtype=float, backend="numpy"
)
copy_stencil(
    in_storage, out_storage,origin=(0, 1, 1), domain=(1, 8, 8)
)
print(out_storage)

@gtscript.stencil(backend="gt:cpu_ifirst")
def copy_stencil_x86(in_storage: gtscript.Field[float], out_storage: gtscript.Field[float]):
    with computation(FORWARD):
        with interval(...):
            out_storage = in_storage

shape = (1, 10, 10)
in_storage_x86 = gt4py.storage.ones(
    shape=shape, default_origin=(0, 0, 0), dtype=float, backend="gt:cpu_ifirst"
)
in_storage_x86.data[:, 2:7, :] = 3
out_storage_x86 = gt4py.storage.zeros(
    shape=shape, default_origin=(0, 0, 0), dtype=float, backend="gt:cpu_ifirst"
)
copy_stencil_x86(
    in_storage_x86, out_storage_x86, origin=(0, 1, 1), domain=(1, 8, 8)
)
print(out_storage_x86)

#@gtscript.stencil(backend="gt:gpu")
#def copy_stencil_gpu(in_storage: gtscript.Field[float], out_storage: gtscript.Field[float]):
#    with computation(FORWARD):
#        with interval(...):
#            out_storage = in_storage
#
#shape = (1, 10, 10)
#in_storage_gpu = gt4py.storage.ones(
#    shape=shape, default_origin=(0, 0, 0), dtype=float, backend="gt:gpu", managed_memory=True
#)
##in_storage_gpu.data[:, 2:7, :] = 4
#in_storage_gpu[:, 2:7, :] = 4
#out_storage_gpu = gt4py.storage.zeros(
#    shape=shape, default_origin=(0, 0, 0), dtype=float, backend="gt:gpu", managed_memory=True
#)
#copy_stencil_gpu(
#    in_storage_gpu, out_storage_gpu, origin=(0, 1, 1), domain=(1, 8, 8)
#)
#import cupy as cp
#cp.cuda.runtime.deviceSynchronize()
#print(out_storage_gpu)

#################################
#
# Test Serialbox setup
#
#################################
import sys
import os
import numpy as np
#sys.path.append(os.environ.get('SERIALBOX_ROOT') + '/python')
import serialbox as ser

print(ser.__version__)
config = ser.Config()
print(config.get_dict())

# write test
serializer = ser.Serializer(ser.OpenModeKind.Write, "./", "test-file")
savepoint = ser.Savepoint('test-savepoint')
serializer.write('test-field', savepoint, np.asarray(out_storage))

# read test
serializer = ser.Serializer(ser.OpenModeKind.Read, "./", "test-file")
savepoint_list = serializer.savepoint_list()
savepoint = savepoint_list[0]
field = serializer.read('test-field', savepoint)
assert(np.all(field == np.asarray(out_storage)))
