# Conjugate_Gradient

This benchmark implements different versions of the Conjugate Gradient (CG) algorithm using various features of SYCL.

# Building 

CMake is used to build this benchmark. 

```
mkdir build && cd build
cmake .. -DSYCL_COMPILE= DPCPP|HIPSYCL -DOMP_COMPILE=true|false
```
if `HIPSYCL` is specified as a SYCL implementation then `-DHIPSYCL_INSTALL_DIR` need to be specified. Similarly, when `OMP_COMPILE` is true then `OMP_LIBRARY` need to be specified. 

For optimal performance `OMP_PROC_BIND` is set to true. 

### Example

```
./binary 
   -s <problem size>
   -b <block size (optional)>
   -a <atomics (yes or no) initialized with yes>
```
