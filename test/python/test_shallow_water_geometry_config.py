import sys
sys.path.append("../../src/python")

from shallow_water_geometry_config import shallow_water_geometry_config


test_default = shallow_water_geometry_config()

print("Test __init__ method of shallow_water_geometry_config class \n", test_default)

test_yamlpath = shallow_water_geometry_config(yamlpath="../../parm/shallow_water_test.yml")

print("Test __call__ method of shallow_water_geometry_config class \n", test_yamlpath)

test_args = shallow_water_geometry_config(nx=5,ny=5,xmax=25,ymax=25)

print("Test __init__ method of shallow_water_geometry_config class with nx, ny, xmax, ymax args \n", test_args)
