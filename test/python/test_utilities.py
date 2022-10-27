# Test Utilities 
import numpy as np

class TestUtilities:

    def check_real_scalar(data: float, name: str, expected: float, tolerance: float, errors: int):
        if (np.abs(data - expected) > tolerance):
            print(f"ERROR: Expected {name} = {expected}, but got, {name}  =  {data}")
            errors = errors + 1
        
    def check_integer_scalar(data: int, name: str, expected: int, errors: int):
        if (data != expected):
            print(f"ERROR: Expected {name} = {expected}, but got, {name}  =  {data}")
            errors = errors + 1

    def check_min_max_real(data: np.ndarray, name: str, expected_min: float, expected_max: float, errors: int):
        # Test variables
        _float_value = np.min(data)

        if(_float_value != expected_min):
            print(f"ERROR: Expected minval( {name} = {expected_min}, but got minval( {name} ) = {_float_value}")
            errors = errors + 1

        _float_value = np.max(data)
        if(_float_value != expected_max):
            print(f"ERROR: Expected maxval( {name} = {expected_min}, but got minval( {name} ) = {_float_value}")    
            errors = errors + 1

    # def check_min_max_real2d(data, name, expected_min, expected_max, errors):
    #     # Test variables
    #     _float_value = np.min(data)

    #     if(_float_value != expected_min):
    #         print(f"ERROR: Expected minval(, {name}, ) = , {expected_min}, but got minval(, {name}, ) = , {_float_value}")
    #         errors = errors + 1

    #     _float_value = np.max(data)
    #     if(_float_value != expected_max):
    #         print(f"ERROR: Expected maxval(, {name}, ) = , {expected_min}, but got minval(, {name}, ) = , {_float_value}")    
    #         errors = errors + 1
