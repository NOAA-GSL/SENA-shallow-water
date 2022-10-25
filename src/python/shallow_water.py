"""Main driver for the Shallow Water Model."""

import argparse
import yaml
from mpi4py import MPI
import numpy as np
from netCDF4 import Dataset
import gt4py.storage as gt_storage

from shallow_water_geometry_config import ShallowWaterGeometryConfig
from shallow_water_model_config import ShallowWaterModelConfig
from shallow_water_geometry import ShallowWaterGeometry
from shallow_water_state import ShallowWaterState
from shallow_water_model import ShallowWaterModel


def run_shallow_water(config_file: str, filename=None) -> ShallowWaterModel:

    comm = MPI.COMM_WORLD 

      # Read the configuration settings from config file
    with open(config_file, mode="r") as f: 
        config = yaml.safe_load(f)

    # Set run time config variables 
    _start_step = config['runtime']['start_step']
    _run_steps = config['runtime']['run_steps']
    _output_interval_steps = config['runtime']['output_interval_steps']

    # Create a shallow water geometry configuration from yaml configuration
    geometry_config = ShallowWaterGeometryConfig(yamlpath=config_file)
    
    # Create a shallow water geometry from the configuration
    geometry = ShallowWaterGeometry(geometry=geometry_config, mpi_comm=comm)

    # Create a shallow water model configuration from yaml configuration
    model_config = ShallowWaterModelConfig(yamlpath=config_file)

    if (_start_step != 0 or filename is not None):

        state = ShallowWaterState(geometry=geometry, backend=model_config.backend, clock=0.0)

        state.read_NetCDF(filename)

    else:
        # Create a state with a tsunami pulse in it to initialize field h
        _h = gt_storage.zeros(model_config.backend, default_origin=(1,1), shape=(geometry.npx, geometry.npy), dtype=model_config.F_TYPE)
        xmid = geometry.xmax / 2.0
        ymid = geometry.ymax / 2.0
        sigma = np.floor(geometry.xmax / 20.0)
        for i in range(geometry.xps, geometry.xpe + 1):
            for j in range(geometry.yps, geometry.ype + 1):
                dsqr = (i * geometry.dx - xmid)**2 + (j * geometry.dy - ymid)**2
                _h[i - geometry.xps, j - geometry.yps] = 5000.0 + np.exp(-dsqr / sigma**2) * (model_config.h0 - 5000.0)
        
        state = ShallowWaterState(geometry, clock=0.0, backend=model_config.backend, h=_h)

    # Initialize shallow water model object
    model = ShallowWaterModel(model_config, geometry)

    # Write out the initial state
    if _output_interval_steps <= _run_steps: 
        state.write_NetCDF(f"swout_{_start_step}.nc")

    # Run the model
    for t in range(_start_step, _run_steps, _output_interval_steps):
       
        # Advance the model to next output interval
        model.adv_nsteps(state, min(_output_interval_steps, _run_steps - t))

        # Write out model state if needed 
        if _output_interval_steps <= _run_steps:
            state.write_NetCDF(f"swout_{int(state.clock / model.dt)}.nc")


def main():
    """Driver for the shallow water model that parses config file as a cla."""
    
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