import numpy  # Needed to parse/resolve numpy types specified in YAML
import yaml

class ShallowWaterGeometryConfig:
    """Geometry config
    Can take the arguments nx, ny, xmax, ymax directy to assign to the class, or accept a yaml filepath.
    """

    def __init__(self, nx: int, ny: int, xmax: int, ymax: int):
        self.nx = nx
        self.ny = ny
        self.xmax = xmax
        self.ymax = ymax
    
  
    def __str__(self):
        newline = '\n'
        return(
            f'ny = {self.ny}{newline}'
            f'nx = {self.nx}{newline}'
            f'xmax = {self.xmax}{newline}'
            f'ymax = {self.ymax}'
        )
    
    @classmethod
    def from_YAML_filename(cls, filename: str):
        with open(filename, 'r') as yamlFile:
            try:
                config = yaml.full_load(yamlFile)
            except yaml.YAMLError as e:
                print(e)
        geometry = config['geometry']
        swgc = cls(geometry['nx'], geometry['ny'], geometry['xmax'], geometry['ymax'])
        return swgc

    @classmethod
    def from_YAML_file_object(cls, fileObject):
        with fileObject:
            try:
                config = yaml.full_load(fileObject)
            except yaml.YAMLError as e:
                print(e)
        geometry = config['geometry']
        swgc = cls(geometry['nx'], geometry['ny'], geometry['xmax'], geometry['ymax'])
        return swgc
