import yaml
import numpy as np


# with open('../../parm/shallow_water.yml', 'r') as file:
#     parm = yaml.safe_load(file)

class shallow_water_geometry_config:
    """Geometry config"""
    def __init__(self, data):
        """Create the expected variables for the shallow_water_geometry_config class.
        These will be instantiated from a yaml file (parm/shallow_water.yml).
        """
        self.nx   = np.int32(data['geometry_parm']['nx'])
        self.ny   = np.int32(data['geometry_parm']['ny'])
        self.xmax = np.float64(data['geometry_parm']['xmax'])
        self.ymax = np.float64(data['geometry_parm']['ymax'])

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

    def __call__(self, data):
        """Create the expected variables for the shallow_water_geometry_config class.
        These will be instantiated from a yaml file (parm/shallow_water.yml).
        """
        self.nx   = self.get_nx(data)
        self.ny   = self.get_ny(data)
        self.xmax = self.get_xmax(data)
        self.ymax = self.get_ymax(data)

# t = shallow_water_geometry_config()

# print(t)
# t(parm)
# print(t)