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

# License

Please see LICENSE.txt
