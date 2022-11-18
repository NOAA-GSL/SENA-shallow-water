import sys
import pytest
import numpy  # Needed to parse/resolve numpy types specified in YAML
import yaml

sys.path.append("../../src/python")

from shallow_water_geometry_config import ShallowWaterGeometryConfig

@pytest.mark.mpi_skip
def test_shallow_water_geometry_config_arglist():
    nx = 100
    ny = 200
    xmax = 500.0
    ymax = 600.0

    config = ShallowWaterGeometryConfig(nx, ny, xmax, ymax)

    assert config.nx == nx
    assert config.ny == ny
    assert config.xmax == xmax
    assert config.ymax == ymax

@pytest.mark.mpi_skip
def test_shallow_water_geometry_config_yaml_file():
    filename = '../test_input/test_shallow_water_config.yml'
    with open(filename, 'r') as yamlFile:
        try:
            yamlConfig = yaml.full_load(yamlFile)
        except yaml.YAMLError as e:
            print(e)
    yamlGeometry = yamlConfig['geometry']

    config = ShallowWaterGeometryConfig.from_YAML_filename(filename)

    assert config.nx == yamlGeometry['nx']
    assert config.ny == yamlGeometry['ny']
    assert config.xmax == yamlGeometry['xmax']
    assert config.ymax == yamlGeometry['ymax']

@pytest.mark.mpi_skip
def test_shallow_water_geometry_config_yaml_fp():
    filename = '../test_input/test_shallow_water_config.yml'
    with open(filename, 'r') as yamlFile:
        try:
            yamlConfig = yaml.full_load(yamlFile)
        except yaml.YAMLError as e:
            print(e)
    yamlGeometry = yamlConfig['geometry']

    fp = open(filename, 'r')
    config = ShallowWaterGeometryConfig.from_YAML_file_object(fp)

    assert config.nx == yamlGeometry['nx']
    assert config.ny == yamlGeometry['ny']
    assert config.xmax == yamlGeometry['xmax']
    assert config.ymax == yamlGeometry['ymax']

