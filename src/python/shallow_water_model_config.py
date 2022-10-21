#!/usr/bin/env python

class ShallowWaterModelConfig:

    def __init__(self, dt: float, u0: float, v0: float, b0: float, h0: float):
        self.dt = dt
        self.u0 = u0
        self.v0 = v0
        self.b0 = b0
        self.h0 = h0
    
  
    def __str__(self):
        newline = '\n'
        s = ''
        s += f'dt = {self.dt}{newline}'
        s += f'u0 = {self.u0}{newline}'
        s += f'v0 = {self.v0}{newline}'
        s += f'b0 = {self.b0}{newline}'
        s += f'h0 = {self.h0}'
        return(s)
    
    @classmethod
    def from_YAML_filename(cls, filename: str):
        import yaml
        with open(filename, 'r') as yamlFile:
            try:
                config = yaml.safe_load(yamlFile)
            except yaml.YAMLError as e:
                print(e)
        model = config['model']
        swmc = cls(model['dt'], model['u0'], model['v0'], model['b0'], model['h0'])
        return swmc

    @classmethod
    def from_YAML_file_object(cls, fileObject):
        import yaml
        with fileObject:
            try:
                config = yaml.safe_load(fileObject)
            except yaml.YAMLError as e:
                print(e)
        model = config['model']
        swmc = cls(model['dt'], model['u0'], model['v0'], model['b0'], model['h0'])
        return swmc
  
#swmc1 = ShallowWaterModelConfig(1.0, 2.0, 3.0, 4.0, 5.0)
#print(swmc1)
#
#swmc2 = ShallowWaterModelConfig.from_YAML_filename('foo.yml')
#print(swmc2)
#
#fp = open('foo.yml', 'r')
#swmc3 = ShallowWaterModelConfig.from_YAML_file_object(fp)
#print(swmc3)

