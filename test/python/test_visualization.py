#!/usr/bin/env python

import sys
sys.path.append("../../src/python")

import numpy as np
from mpi4py import MPI
from matplotlib import pyplot as plt, animation
import gt4py.gtscript as gtscript
import gt4py.storage as gt_storage

from shallow_water_geometry_config import ShallowWaterGeometryConfig
from shallow_water_geometry import ShallowWaterGeometry
from shallow_water_model_config import ShallowWaterModelConfig
from shallow_water_state import ShallowWaterState
from shallow_water_model import ShallowWaterModel


comm = MPI.COMM_WORLD
gc = ShallowWaterGeometryConfig(yamlpath='../../parm/shallow_water.yml')
g = ShallowWaterGeometry(geometry=gc, mpi_comm=comm)
mc = ShallowWaterModelConfig('../../parm/shallow_water.yml')
m = ShallowWaterModel(mc, g)

# Create a state with a tsunami pulse in it to initialize field h
h = np.empty((g.npx, g.npy), dtype=float)
xmid = g.xmax / 2.0
ymid = g.ymax / 2.0
sigma = np.floor(g.xmax / 20.0)
for i in range(g.xps, g.xpe + 1):
    for j in range(g.yps, g.ype + 1):
        dsqr = (i * g.dx - xmid)**2 + (j * g.dy - ymid)**2
        h[i - g.xps, j - g.yps] = 5000.0 + np.exp(-dsqr / sigma**2) * (mc.h0 - 5000.0)
s = ShallowWaterState(g, clock=0.0, h=h)

# Write the initial state
# s.write_NetCDF("../test_output/state_0.nc")

# Advance the model state
# m.adv_nsteps(s, 200)

# Write the final state
# s.write_NetCDF("../test_output/state_200.nc")

def animate(n):
    m.adv_nsteps(s, 1)
    pc.set_array(s.h.data)

import matplotlib.colors as colors
class ColormapNormalize(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, v1=None, v2=None, clip=False):
        self.v1 = v1
        self.v2 = v2
        super().__init__(vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        # Note also that we must extrapolate beyond vmin/vmax
        x, y = [self.vmin, self.v1, self.v2, self.vmax], [0, 0.1, 0.9, 1.]
        return np.ma.masked_array(np.interp(value, x, y,
                                            left=-np.inf, right=np.inf))

    def inverse(self, value):
        y, x = [self.vmin, self.v1, self.v2, self.vmax], [0, 0.1, 0.9, 1]
        return np.interp(value, x, y, left=-np.inf, right=np.inf)



# Plot animation
X, Y = np.meshgrid(np.linspace(0, g.xmax, g.npx), np.linspace(0, g.ymax, g.npy))
fig, axs = plt.subplots()
norm = ColormapNormalize(vmin=4990, vmax=5030, v1=4995, v2=5005)
pc = axs.pcolormesh(X, Y, s.h.data, cmap='nipy_spectral', norm=norm)
cb = fig.colorbar(pc, ax=axs, extend='both')
fig.savefig('../test_output/test0.png')
anim = animation.FuncAnimation(fig, animate, interval=125, frames=500)
anim.save('../test_output/test.gif')
fig.savefig('../test_output/test500.png')
# s.write_NetCDF("../test_output/state_500.nc")