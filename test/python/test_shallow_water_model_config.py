import sys
sys.path.append("../../src/python")

from shallow_water_model_config import ShallowWaterModelConfig


test_default = ShallowWaterModelConfig()

print("Test __init__ method of ShallowWaterModelConfig class for instantiating default values")
print(test_default)

test_yamlpath = ShallowWaterModelConfig(yamlpath="../test_input/test_shallow_water_config.yml")

print("Test __init__ method of ShallowWaterModelConfig class with a yaml file")
print(test_yamlpath)

test_args = ShallowWaterModelConfig(dt=1.0, u0=1.0, v0=1.0, b0=1.0, h0=4000.0)

print("Test __init__ method of ShallowWaterModelConfig class, passing dt, u0, v0, b0, h0 arguments")
print(test_args)

