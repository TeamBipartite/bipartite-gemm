# CSC485b Fall 2024 A4
Johnathan Warawa & Emily Martins

# Prerequisites
You should have a functional CUDA environment installed with a GPU of compute
capability 7.5 or higher. Our code is has been tested on both the Turing and Ampere
architectures.

OpenBLAS is helpful for checking correctness of output, but not necessary.
See the 'Build' section for more details.

# Build
A makefile is provided in  the top-level directory which handles building the application. 
The default target is `sm_75` (eg, Compute Capability 7.5 graphics cards such as the Turing T4). 
For accurate GFLOPs/SM calculations, make sure to specify the number of SMs your
GPU has with the NUM_SMS environment variable (default is 28)
```bash
$ make NUM_SMS=40
```

To specify the `arch` string for your target, use the `TARGET` variable when
calling `make`. For example, for a Compute Capability 8.6 graphics card such as the RTX 3060, use:
```bash
$ make TARGET=sm_86 NUM_SMS=28
```

The following options may also be specified using environment variables:

* `USE_OPENBLAS=yes`: By default, the OpenBLAS library is called to
  perform a CPU matrix multiplication to serve as a baseline to check for
  correctness. If you do not have OpenBLAS on your system, set this variable
  to `no` to use a naive provided n^3 CPU implementation.
* `OPENBLAS_NUM_THREADS=$(nproc)`: If `USE_OPENBLAS=yes`, this option can be
  specified to reduce the number of threads used by OpenBLAS.

A clean is required before switching configurations.

# Run

A single executable, `a4` is generated. Simply run this file.

For debugging and output inspection, `a4` provides a `-p` argument. When this
argument is provided, both the expected and actual outputs is printed.

**PRECISION NOTE:** We provide an FP16 version of our tensor matrix multiplication,
but since FP16 only has a 10-bit mantissa, it can be quite inaccurate for larger
matrices (or matrices with large values). We use a fixed epsilon (currently set
to `0.00001`) for output checking. This works on the provided example, but if the
parameters in `main.cu` are changed, **the test may report a fail for the FP16 runs
even though the output is as accurate as practical for FP16**. Changing the
`FIXED_EPSILON` in `gemm_experiment.h` may alleviate these issues, but note that
for large matrices (or large values) the accumulated error of many FP16 additions
can be quite large (in the order of 3 or 4 decimal digits in the worst possible
cases). **This is a practical limitation with FP16 that accounting for
is outside the scope of this project.**
