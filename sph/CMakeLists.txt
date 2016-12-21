include_directories(./sph/)

set(CUDA_SPH_FILES 
	./sph/sph.cuh
	./sph/sph_cuda.cu
	./sph/sph_kernel_impl.cuh)

set(SPH_FILES
	./sph/sph.h
	./sph/sph.cpp)
