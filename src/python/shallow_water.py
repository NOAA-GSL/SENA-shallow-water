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

    _rank = comm.Get_rank()


    # Read the configuration settings from config file
    with open(config_file, mode="r") as f: 
        config = yaml.safe_load(f)

    # Broadcast the runtime configuration settings 
    # comm.Bcast(np.full(1, config['runtime']['start_step']))
    # comm.Bcast(np.full(1, config['runtime']['run_steps']))
    # comm.Bcast(np.full(1, config['runtime']['output_interval_steps']))
    # comm.Bcast(np.full(1, config['runtime']['io_format']))

    # # Broadcast the geometry namelist settings
    # comm.Bcast(np.full(1, config['geometry']['nx']))
    # comm.Bcast(np.full(1, config['geometry']['ny']))
    # comm.Bcast(np.full(1, config['geometry']['xmax']))
    # comm.Bcast(np.full(1, config['geometry']['ymax']))

    # # Broadcast the model namelist settings 
    # comm.Bcast(np.full(1, config['model']['dt']))
    # comm.Bcast(np.full(1, config['model']['u0']))
    # comm.Bcast(np.full(1, config['model']['v0']))
    # comm.Bcast(np.full(1, config['model']['b0']))
    # comm.Bcast(np.full(1, config['model']['h0']))

    # Create a shallow water geometry configuration from yaml configuration
    geometry_config = ShallowWaterGeometryConfig(yamlpath=config_file)
    
    # Create a shallow water geometry from the configuration
    geometry = ShallowWaterGeometry(geometry=geometry_config, mpi_comm=comm)

    # Create a shallow water model configuration from yaml configuration
    model_config = ShallowWaterModelConfig(filename)

    if (config['runtime']['start_step'] != 0 or filename is not None):

        state = ShallowWaterState(geometry=geometry, clock=0.0)

        state.read_NetCDF(filename)

    else:

        _h = gt_storage.zeros(config['gt4py_vars']['backend'], default_origin=(1,1), shape=(geometry.npx, geometry.npy), dtype=np.float64)
        xmid = geometry.xmax /2.0
        ymid = geometry.ymax /2.0
        sigma = np.floor(geometry.xmax / 20.0)
        for i in range(geometry.xps, geometry.xpe + 1):
            for j in range(geometry.yps, geometry.ype + 1):
                dsqr = (i * geometry.dx - xmid)**2 + (j * geometry.dy - ymid)**2
                _h[i - geometry.xps, j - geometry.yps] = 5000.0 + np.exp(-dsqr / sigma**2) * (model_config.h0 - 5000.0)
        
        state = ShallowWaterState(geometry, clock=0.0, h=_h)

    # Initialize shallow water model object
    model = ShallowWaterModel(model_config, geometry)

    # Write out the initial state
    state.write_NetCDF(f"swout_{config['runtime']['start_step']}.nc")

    # TODO Write out state if needed

    # Run the model
    model.adv_nsteps(state, config['runtime']['run_steps'])

    # Write out the final state
    state.write_NetCDF(f"swout_{config['runtime']['run_steps']}.nc")


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