import yaml
import numpy as np


class ShallowWaterModelConfig:

    def __init__(self, yamlpath=None, dt=None, u0=None, v0=None, b0=None, h0=None):


        if (yamlpath):
            try:
                with open(yamlpath, 'r') as file:
                    param = yaml.safe_load(file)
            except IOError as e:
                print('Error reading file', file)
                raise e
            except yaml.YAMLError as e:
                print('Error parsing file', file)
                raise e

            self.dt = param['model']['dt']
            self.u0 = param['model']['u0']
            self.v0 = param['model']['v0']
            self.b0 = param['model']['b0']
            self.h0 = param['model']['h0']

            self.backend = param['gt4py_vars']['backend']

        else: 
            # Set default values
            self.dt = dt if dt is not None else np.float64(0.0)
            self.u0 = u0 if u0 is not None else np.float64(0.0)
            self.v0 = v0 if v0 is not None else np.float64(0.0)
            self.b0 = b0 if b0 is not None else np.float64(0.0)
            self.h0 = h0 if h0 is not None else np.float64(5030.0)
            self.backend = "numpy"


    def __str__(self):
        return (
            f'dt = {self.dt}\n'
            f'u0 = {self.u0}\n'
            f'v0 = {self.v0}\n'
            f'b0 = {self.b0}\n'
            f'h0 = {self.h0}\n'
            f'backend = {self.backend}\n'
            f'dtype   = {self.dtype}\n'
            f'F_TYPE  = {self.F_TYPE}\n'
            f'I_TYPE  = {self.I_TYPE}'
        )
