#!/usr/bin/env python

from mpi4py import MPI
from matplotlib import pyplot as plt, animation, colors
import numpy as np
import yaml
import argparse

from shallow_water_geometry_config import ShallowWaterGeometryConfig
from shallow_water_geometry import ShallowWaterGeometry
from shallow_water_model_config import ShallowWaterModelConfig
from shallow_water_model import ShallowWaterModel
from shallow_water_gt4py_config import ShallowWaterGT4PyConfig
from shallow_water_state import ShallowWaterState


class ColormapNormalize(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, v1=None, v2=None, clip=False):
        self.v1 = v1
        self.v2 = v2
        super().__init__(vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        # Note also that we must extrapolate beyond vmin/vmax
        x, y = [self.vmin, self.v1, self.v2, self.vmax], [0, 0.05, 0.95, 1.]
        return np.ma.masked_array(np.interp(value, x, y,
                                            left=-np.inf, right=np.inf))

    def inverse(self, value):
        y, x = [self.vmin, self.v1, self.v2, self.vmax], [0, 0.1, 0.9, 1]

        return np.interp(value, x, y, left=-np.inf, right=np.inf)


def plot_animation(config_file: str, filename=None):

    comm = MPI.COMM_WORLD

    # config_file = "../../parm/shallow_water_test.yml"

    def animate(n):
        m.adv_nsteps(s, 1)
        pc.set_array(s.h.data)

    # Create the geometry
    gc = ShallowWaterGeometryConfig.from_YAML_filename(config_file)
    g = ShallowWaterGeometry(gc, comm)

    # Create the model
    mc = ShallowWaterModelConfig.from_YAML_filename(config_file)
    gtc = ShallowWaterGT4PyConfig.from_YAML_filename(config_file)
    m = ShallowWaterModel(mc, gtc, g)

    # Get the runtime config
    with open(config_file, 'r') as yamlFile:
        try:
            config = yaml.full_load(yamlFile)
        except yaml.YAMLError as e:
            print(e)
    runtime = config['runtime']

    # If this is a restart, initialze model from restart file
    if (runtime['start_step'] != 0 or filename is not None):

        # Initialize a default state
        s = ShallowWaterState(g, gtc)

        # Read restart file into state
        s.read(filename)

    # Otherwise create a new model from the configuration
    else:

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

    ## Plot animation
    X, Y = np.meshgrid(np.linspace(0, g.xmax, g.npx), np.linspace(0, g.ymax, g.npy))
    fig, axs = plt.subplots()
    norm = ColormapNormalize(vmin=4990, vmax=5030, v1=4995, v2=5005)
    pc = axs.pcolormesh(X, Y, s.h.data, cmap='nipy_spectral', norm=norm)
    cb = fig.colorbar(pc, ax=axs, extend='both')
    cb.set_ticks([4990, 4995, 4996, 4997, 4998, 4999, 5000, 5001, 5002, 5003, 5004, 5005, 5030])
    anim = animation.FuncAnimation(fig, animate, interval=125, frames=runtime['run_steps'])
    anim.save('shallow_water.gif')

def main():
    """Visualization tool to output a .gif file for the shallow water model.
    Parses config file or restart file as command line arguments."""
    
    parser = argparse.ArgumentParser(description='Run the shallow water model.')
    
    parser.add_argument('config_file', type=str, help='yaml configuration file')

    parser.add_argument('--filename', type=str, help='NetCDF file for a restart')
   
    args = parser.parse_args()

    if args.filename:
        plot_animation(args.config_file, filename=args.filename)
    else:
        plot_animation(args.config_file)


if __name__ == "__main__":
    main()