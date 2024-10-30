# CSC485b Fall 2024 A2
Johnathan Warawa & Emily Martins

# Prerequisites
You should have a functional CUDA environment installed with a GPU of compute
capability 7.5 or higher. Our code is best optimized for the Turing architecture,
but has been tested to work on Ampere.

OpenBLAS is helpful for checking correctness of output, but not necessary.
See the 'Build' section for more details.

# Build
A makefile is provided in  the top-level directory which handles building the application. 
The default target is `sm_75` (eg, Compute Capability 7.5 graphics cards such as the Turing T4). 
```bash
$ make
```

To specify the `arch` string for your target, use the `TARGET` variable when
calling `make`. For example, for a Compute Capability 8.6 graphics card such as the RTX 3060, use:
```bash
$ make TARGET=sm_86
```

The following options may also be specified using environment variables:

* `USE_OPENBLAS=yes`: By default, the OpenBLAS library is called to
  perform a CPU matrix multiplication to serve as a baseline to check for
  correctness. If you do not have OpenBLAS on your system, set this variable
  to `no` to use a naive provided n^3 CPU implementation.
* `OPENBLAS_NUM_THREADS=$(nproc)`: If `USE_OPENBLAS=yes`, this option can be
  specified to reduce the number of threads used by OpenBLAS. 
* `USE_FULL_MULTIPLY=no`: Set to `yes` to use the less-optimized full-multiply
   dense two_hop_reachability, otherwise use the more-optimized 
   `just-set-one` implementation (the one described as 'clever' in our video) for
   dense.
* `USE_WARP_PRIMITIVES=yes`: If using full multiply, set to `no` to use an
  unoptimized solution which does not exploit warp primitives.

A clean is required before switching configurations.

# Run

A single executable, `a2` is generated. Simply run this file.

For debugging and output inspection, `a2` provides a `-p` argument. When this 
argument is provided, a printed output of the generated edge list in addition to
both expected and actual outputs for both dense and CSR implementations is produced.
