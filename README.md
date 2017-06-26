# sph_opengl

http://bromat.fr/2017/gpu-sph-fluid-solver/

CUDA implementation of

* https://cg.informatik.uni-freiburg.de/publications/2014_EG_SPH_STAR.pdf
* https://cg.informatik.uni-freiburg.de/publications/2013_CASA_elasticSolids.pdf

# Dependencies

* GLM
* CUDA
* Assimp
* GLFW
* GLEW
* https://github.com/Mathiasb17/sph_boundary_particles

# Installation

Currently, it will only build on linux based systems (else you need to tweak CMakeLists.txt at least)

Get the project :

```bash
git clone git://github.com/mathiasb17/sph_opengl
cd sph_opengl
git submodule update --init --recursive
```

Build the project :

```bash
mkdir build
cd build
cmake ..
make -j8
```

# Configuration

## Precision

Natively, this simulator was developped to work in single-precision. However, you may want to work using double precision floating numbers. This was made possible with preprocessor. 
In **CMakeLists.txt**, just replace this line 

```
add_definitions(-DDOUBLE_PRECISION=0)
```

by

```
add_definitions(-DDOUBLE_PRECISION=1)
```

I largely insist on the fact you will have poor performances using double precision because of bad memory coalescence with CUDA. In most cases you'll only need single precision.

## Recording

If you want to record the animation, you'll first need **ffmpeg**. Then replace

```
add_definitions(-DRECORD_SIMULATION=0)
```

by

```
add_definitions(-DRECORD_SIMULATION=1)
```

# Future Works

I plan to implement a lot of physics/performance improvements. However, doing this on my free time, i'm not quite sure when ! If you are interested in this project, please feel free to contact me and maybe
we can work together. Here are some improvement i would like to implement :

* FLIP Solver
* PBF Solver
* Two way coupling
* Nanogui integration for better user control

# License

Please see LICENSE.txt
