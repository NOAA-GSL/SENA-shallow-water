#!/usr/bin/env python

"""Main driver for the Shallow Water Model."""

import argparse
import yaml
from mpi4py import MPI
import numpy as np

from shallow_water_geometry_config import ShallowWaterGeometryConfig
from shallow_water_geometry import ShallowWaterGeometry
from shallow_water_gt4py_config import ShallowWaterGT4PyConfig
from shallow_water_model_config import ShallowWaterModelConfig
from shallow_water_state import ShallowWaterState
from shallow_water_model import ShallowWaterModel


def run_shallow_water(config_file: str, filename=None):

    comm = MPI.COMM_WORLD

    # Create the geometry
    gc = ShallowWaterGeometryConfig.from_YAML_filename(config_file)
    g = ShallowWaterGeometry(gc, comm)

    # Create the model
    mc = ShallowWaterModelConfig.from_YAML_filename(config_file)
    gtc = ShallowWaterGT4PyConfig.from_YAML_filename(config_file)
    m = ShallowWaterModel(mc, gtc, g)

    # Read the configuration settings from config file
    with open(config_file, "r") as yamlFile: 
        try:
            config = yaml.full_load(yamlFile)
        except yaml.YAMLError as e:
            print(e)
    runtime = config['runtime']


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


    # Write out the initial state if needed
    if (runtime['output_interval_steps'] <= runtime['run_steps'] and filename is None):
        s.write(f"swout_{round(s.clock / m.dt):07d}.nc")

    # Run the model
    for t in range(0, runtime['run_steps'], runtime['output_interval_steps']):
        m.adv_nsteps(s, min(runtime['output_interval_steps'], runtime['run_steps'] - t))
        if (runtime['output_interval_steps'] <= runtime['run_steps']):
            s.write(f"swout_{round(s.clock / m.dt):07d}.nc")
        print(f"t = {t}")


def main():
    """Driver for the shallow water model that parses config file or restart file
    as command line arguments."""
    
    parser = argparse.ArgumentParser(description='Run the shallow water model.')
    
    parser.add_argument('config_file', type=str, help='yaml configuration file')

    parser.add_argument('--filename', type=str, help='NetCDF file for a restart')
   
    args = parser.parse_args()

    if args.filename:
        run_shallow_water(args.config_file, filename=args.filename)
    else:
        run_shallow_water(args.config_file)


if __name__ == "__main__":
    main()

