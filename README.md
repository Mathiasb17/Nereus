(June 2018) I've been off this project for months. Now i'm planning to do a brand new code base.

# Nereus

Nereus is a CUDA parallelized library for particle based fluid dynamics.

It includes a CUDA implementation of :

* State-Equation SPH
* Implicit Incompressible SPH
* Predictive-Corrective Incompressible SPH (soon finished)
* Much more incoming : see Future Works below

# Dependencies

* GLM
* CUDA
* Assimp
* GLFW
* GLEW
* [SPH boundary particles](https://github.com/Mathiasb17/sph_boundary_particles)

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

## SPH Kernels

In SPH there are mainly two sets of kernels : the ones introduced by Monaghan, and the one introduced by Mueller et al. [MCG03]. There is no consensus about what set of kernels to use and both
gives pretty good results. In this project you can choose the set of kernel you want at compile time by tuning this define instruction :

```
add_definitions(-DKERNEL_SET=1) #0 for monaghan, 1 for muller kernels
```

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

# Cite

If you found this project useful for academic research projects, you can cite this work with the following bibtex reference :

```bibtex
@misc{Nereus,
   Author = {Mathias Brousset},
   Year = {2016},
   Note = {https://github.com/Mathiasb17/Nereus},
   Title = {Nereus library}
} 
```

Thanks a lot ! :^)

# License

Please see LICENSE.txt
