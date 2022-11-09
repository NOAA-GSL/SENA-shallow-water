import numpy   # Needed to parse/resolve numpy types specified in YAML
import yaml

class ShallowWaterGT4PyConfig:

    def __init__(self, backend: str, float_type: type):
        self.backend = backend
        self.float_type = float_type
  
    def __str__(self):
        newline = '\n'
        s = ''
        s += f'    backend = {self.backend}{newline}'
        s += f' float_type = {self.float_type}{newline}'
        return(s)
    
    @classmethod
    def from_YAML_filename(cls, filename: str):
        with open(filename, 'r') as yamlFile:
            try:
                config = yaml.full_load(yamlFile)
            except yaml.YAMLError as e:
                print(e)
        gt4py = config['gt4py']
        swgtc = cls(gt4py['backend'], gt4py['float_type'])
        return swgtc

    @classmethod
    def from_YAML_file_object(cls, fileObject):
        with fileObject:
            try:
                config = yaml.full_load(fileObject)
            except yaml.YAMLError as e:
                print(e)
        gt4py = config['gt4py']
        swgtc = cls(gt4py['backend'], gt4py['float_type'])
        return swgtc
