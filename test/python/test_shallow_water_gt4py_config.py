import sys
import pytest

sys.path.append("../../src/python")

from shallow_water_gt4py_config import ShallowWaterGT4PyConfig

@pytest.mark.mpi_skip
def test_shallow_water_gt4py_config_arglist():
    backend = "numpy"

    config = ShallowWaterGT4PyConfig(backend)

    assert config.backend == 'numpy'

@pytest.mark.mpi_skip
def test_shallow_water_gt4py_config_yaml_file():
    import yaml
    filename = '../test_input/test_shallow_water_config.yml'
    with open(filename, 'r') as yamlFile:
        try:
            yamlConfig = yaml.safe_load(yamlFile)
        except yaml.YAMLError as e:
            print(e)
    yamlGt4Py = yamlConfig['gt4py']

    config = ShallowWaterGT4PyConfig.from_YAML_filename(filename)

    assert config.backend == yamlGt4Py['backend']

@pytest.mark.mpi_skip
def test_shallow_water_gt4py_config_yaml_fp():
    import yaml
    filename = '../test_input/test_shallow_water_config.yml'
    with open(filename, 'r') as yamlFile:
        try:
            yamlConfig = yaml.safe_load(yamlFile)
        except yaml.YAMLError as e:
            print(e)
    yamlGt4Py = yamlConfig['gt4py']

    fp = open(filename, 'r')
    config = ShallowWaterGT4PyConfig.from_YAML_file_object(fp)

    assert config.backend == yamlGt4Py['backend']
