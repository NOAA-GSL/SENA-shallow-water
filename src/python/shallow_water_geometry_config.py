import yaml
import numpy as np

class shallow_water_geometry_config:
    """Geometry config
    Can take the arguments nx, ny, xmax, ymax directy to assign to the class, or by
    calling the shallow_water_geometry_config class with a yaml filepath.

    """
    def __init__(self, data: str, nx=None, ny=None, xmax=None, ymax=None):
        """Set variables for the shallow_water_geometry_config class if passed during instantiation.
        If not passed, provide default values.
        """

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
            f'nx   = {self.nx} \n'
            f'ny   = {self.ny} \n'
            f'xmax = {self.xmax} \n'
            f'ymax = {self.ymax}'
        )

    def __call__(self, data: str):
        """        
        Set variables for the shallow_water_geometry_config class.
        These will be instantiated from a yaml file (parm/shallow_water.yml).

        :param data: string,  a filepath to a yaml file with a "geometry_parm" collection,
        containing key value pairs of nx, ny, xmax, and ymax. 
        """

        # 
        try:
            with open(data, 'r') as file:
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


# Test the class above 

# t = shallow_water_geometry_config()

# print(t)
# t("../../parm/shallow_water.yml")
# print("from geometry_config.py \n", t)