import yaml
import numpy as np

class shallow_water_geometry_config:
    """Geometry config
    Can take the arguments nx, ny, xmax, ymax directy to assign to the class, or accept a yaml filepath.

    """
    def __init__(self, yamlpath=None, nx=None, ny=None, xmax=None, ymax=None):
        """Set variables for the shallow_water_geometry_config class passed during instantiation.
        These can be instantiated from a yaml file (parm/shallow_water.yml), or passed directly,
        if nothing is passed, set default values.
        
        Arguments:
            yamlpath:    string,   A filepath to a yaml file with a "geometry_parm" collection,
                                       containing key value pairs of nx, ny, xmax, and ymax. 
            nx, ny:      integer,  Number of gridpoints in x and y directions. 
            xmax, ymax:  integer,  Maximum extent of the domain in the x and y directions. 
        Return:
            An initialized shallow_water_geometry_config class
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

            self.nx   = self.get_nx(param)
            self.ny   = self.get_ny(param)
            self.xmax = self.get_xmax(param)
            self.ymax = self.get_ymax(param)
        
        else:     
        
            self.nx   = nx if nx is not None else np.int32(151)
            self.ny   = ny if ny is not None else np.int32(151)
            self.xmax = xmax if xmax is not None else np.float64(100000.0)
            self.ymax = ymax if ymax is not None else np.float64(100000.0)



    def get_nx(self, data):
        nx = np.int32(data['geometry_parm']['nx'])
        self.nx = nx 
        return nx

    def get_ny(self, data):
        ny = np.int32(data['geometry_parm']['ny'])
        self.ny = ny 
        return ny

    def get_xmax(self, data):
        xmax = np.float64(data['geometry_parm']['xmax'])
        self.xmax = xmax 
        return xmax

    def get_ymax(self,data):
        ymax = np.float64(data['geometry_parm']['ymax'])
        self.ymax = ymax 
        return ymax

    def __str__(self):
        return (
            f'nx   = {self.nx}\n'
            f'ny   = {self.ny}\n'
            f'xmax = {self.xmax}\n'
            f'ymax = {self.ymax}'
        )

    # def __call__(self, yamlpath: str):
    #     """        
    #     Set variables for the shallow_water_geometry_config class.
    #     These can be instantiated from a yaml file (parm/shallow_water.yml).


    #     """

    #     # 
    #     try:
    #         with open(yamlpath, 'r') as file:
    #             param = yaml.safe_load(file)
    #     except IOError as e:
    #         print('Error reading file', file)
    #         raise e
    #     except yaml.YAMLError as e:
    #         print('Error parsing file', file)
    #         raise e

    #     self.nx   = self.get_nx(param)
    #     self.ny   = self.get_ny(param)
    #     self.xmax = self.get_xmax(param)
    #     self.ymax = self.get_ymax(param)
