import sys
import pytest

sys.path.append("../../src/python")

from shallow_water_model_config import ShallowWaterModelConfig

@pytest.mark.mpi_skip
def test_shallow_water_model_config_arglist():
    dt = 1.0
    u0 = 2.0
    v0 = 3.0
    b0 = 4.0
    h0 = 5.0

    config = ShallowWaterModelConfig(dt, u0, v0, b0, h0)

    assert config.dt == dt
    assert config.u0 == u0
    assert config.v0 == v0
    assert config.b0 == b0
    assert config.h0 == h0

@pytest.mark.mpi_skip
def test_shallow_water_model_config_yaml_file():
    import yaml
    filename = '../test_input/test_shallow_water_config.yml'
    with open(filename, 'r') as yamlFile:
        try:
            yamlConfig = yaml.safe_load(yamlFile)
        except yaml.YAMLError as e:
            print(e)
    yamlModel = yamlConfig['model']

    config = ShallowWaterModelConfig.from_YAML_filename(filename)

    assert config.dt == yamlModel['dt']
    assert config.u0 == yamlModel['u0']
    assert config.v0 == yamlModel['v0']
    assert config.b0 == yamlModel['b0']
    assert config.h0 == yamlModel['h0']

@pytest.mark.mpi_skip
def test_shallow_water_model_config_yaml_fp():
    import yaml
    filename = '../test_input/test_shallow_water_config.yml'
    with open(filename, 'r') as yamlFile:
        try:
            yamlConfig = yaml.safe_load(yamlFile)
        except yaml.YAMLError as e:
            print(e)
    yamlModel = yamlConfig['model']

    fp = open(filename, 'r')
    config = ShallowWaterModelConfig.from_YAML_file_object(fp)

    assert config.dt == yamlModel['dt']
    assert config.u0 == yamlModel['u0']
    assert config.v0 == yamlModel['v0']
    assert config.b0 == yamlModel['b0']
    assert config.h0 == yamlModel['h0']
