import numpy   # Needed to parse/resolve numpy types specified in YAML
import yaml

class ShallowWaterGT4PyConfig:
    """Geometry config
    Can take the arguments backend & float_type directy to assign to the class, or accept a yaml filepath.
    """
    
    def __init__(self, backend: str, float_type: type, gpus_per_node=0):
        self.backend = backend
        self.float_type = float_type
        self.gpus_per_node = gpus_per_node
  
    def __str__(self):
        newline = '\n'
        return(
            f'    backend = {self.backend}{newline}'
            f' float_type = {self.float_type}{newline}'
        )
    
    @classmethod
    def from_YAML_filename(cls, filename: str):
        with open(filename, 'r') as yamlFile:
            try:
                config = yaml.full_load(yamlFile)
            except yaml.YAMLError as e:
                print(e)
        gt4py = config['gt4py']
        swgtc = cls(gt4py['backend'], gt4py['float_type'], gt4py['gpus_per_node'])
        return swgtc

    @classmethod
    def from_YAML_file_object(cls, fileObject):
        with fileObject:
            try:
                config = yaml.full_load(fileObject)
            except yaml.YAMLError as e:
                print(e)
        gt4py = config['gt4py']
        swgtc = cls(gt4py['backend'], gt4py['float_type'], gt4py['gpus_per_node'])
        return swgtc
