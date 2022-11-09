import numpy  # Needed to parse/resolve numpy types specified in YAML
import yaml

class ShallowWaterGeometryConfig:

    def __init__(self, nx: int, ny: int, xmax: int, ymax: int):
        self.nx = nx
        self.ny = ny
        self.xmax = xmax
        self.ymax = ymax
    
  
    def __str__(self):
        newline = '\n'
        s = ''
        s += f'nx = {self.nx}{newline}'
        s += f'ny = {self.ny}{newline}'
        s += f'xmax = {self.xmax}{newline}'
        s += f'ymax = {self.ymax}'
        return(s)
    
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

