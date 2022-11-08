class ShallowWaterGT4PyConfig:

    def __init__(self, backend: str):
        self.backend = backend
  
    def __str__(self):
        newline = '\n'
        s = ''
        s += f' backend = {self.backend}{newline}'
        return(s)
    
    @classmethod
    def from_YAML_filename(cls, filename: str):
        import yaml
        with open(filename, 'r') as yamlFile:
            try:
                config = yaml.safe_load(yamlFile)
            except yaml.YAMLError as e:
                print(e)
        gt4py = config['gt4py']
        swgtc = cls(gt4py['backend'])
        return swgtc

    @classmethod
    def from_YAML_file_object(cls, fileObject):
        import yaml
        with fileObject:
            try:
                config = yaml.safe_load(fileObject)
            except yaml.YAMLError as e:
                print(e)
        gt4py = config['gt4py']
        swgtc = cls(gt4py['backend'])
        return swgtc
