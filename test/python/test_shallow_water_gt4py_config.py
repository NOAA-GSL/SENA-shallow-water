import sys
import pytest
import numpy  # Needed to parse/resolve numpy types specified in YAML
import yaml

sys.path.append("../../src/python")

from shallow_water_gt4py_config import ShallowWaterGT4PyConfig

@pytest.mark.mpi_skip
def test_shallow_water_gt4py_config_arglist():
    backend = "numpy"

    config = ShallowWaterGT4PyConfig(backend, numpy.float64)

    assert config.backend == 'numpy'
    assert config.float_type == numpy.float64

@pytest.mark.mpi_skip
def test_shallow_water_gt4py_config_yaml_file():
    filename = '../test_input/test_shallow_water_config.yml'
    with open(filename, 'r') as yamlFile:
        try:
            yamlConfig = yaml.full_load(yamlFile)
        except yaml.YAMLError as e:
            print(e)
    yamlGt4Py = yamlConfig['gt4py']

    config = ShallowWaterGT4PyConfig.from_YAML_filename(filename)

    assert config.backend == yamlGt4Py['backend']
    assert config.float_type == yamlGt4Py['float_type']

@pytest.mark.mpi_skip
def test_shallow_water_gt4py_config_yaml_fp():
    filename = '../test_input/test_shallow_water_config.yml'
    with open(filename, 'r') as yamlFile:
        try:
            yamlConfig = yaml.full_load(yamlFile)
        except yaml.YAMLError as e:
            print(e)
    yamlGt4Py = yamlConfig['gt4py']

    fp = open(filename, 'r')
    config = ShallowWaterGT4PyConfig.from_YAML_file_object(fp)

    assert config.backend == yamlGt4Py['backend']
    assert config.float_type == yamlGt4Py['float_type']

