import yaml
import numpy as np

class ShallowWaterGeometryConfig:
    """Geometry config
    Can take the arguments nx, ny, xmax, ymax directy to assign to the class, or accept a yaml filepath.
    """
    
    def __init__(self, yamlpath=None, nx=None, ny=None, xmax=None, ymax=None):
        """
        Set variables for the ShallowWaterGeometryConfig class passed during instantiation.
        These can be instantiated from a yaml file (parm/shallow_water.yml), or passed directly,
        if nothing is passed, set default values.
        
            Arguments:
                yamlpath:    string,   A filepath to a yaml file with a "geometry_parm" collection,
                                           containing key value pairs of nx, ny, xmax, and ymax. 
                nx, ny:      integer,  Number of gridpoints in x and y directions. 
                xmax, ymax:  integer,  Maximum extent of the domain in the x and y directions. 
            Return:
                An initialized ShallowWaterGeometryConfig class
        """

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

            self.nx   = param['geometry']['nx']
            self.ny   = param['geometry']['ny']
            self.xmax = param['geometry']['xmax']
            self.ymax = param['geometry']['ymax']
        
        else:     
        
            self.nx   = nx if nx is not None else np.int32(151)
            self.ny   = ny if ny is not None else np.int32(151)
            self.xmax = xmax if xmax is not None else np.float64(100000.0)
            self.ymax = ymax if ymax is not None else np.float64(100000.0)

    def __str__(self):
        return (
            f'nx   = {self.nx}\n'
            f'ny   = {self.ny}\n'
            f'xmax = {self.xmax}\n'
            f'ymax = {self.ymax}'
        )
