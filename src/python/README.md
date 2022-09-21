# Steps to build the model & executables, serialize input/output data,  run the Fortran model, run GT4Py stencils and compare results 


Build the code with cmake GT4Py option on:

## Create the build directory
```
rm -rf build
mkdir build
cd build
```

```
cmake .. -DGT4PY=on -DCMAKE_BUILD_TYPE=debug -DMPIEXEC_PREFLAGS=--oversubscribe
```

```
rm -rf ../exe
make install
```

```
cd ../src/python
```

## Run the executable and generate serialized data with the serialize_test namelist

```
../../exe/shallow_water.x ../../parm/serialize_test.nl
```

## To run comparison tests with the serialized data:

```
python shallow_water_stencils.py
```
