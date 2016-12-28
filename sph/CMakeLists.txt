include_directories(./sph/)
include(${CMAKE_CURRENT_SOURCE_DIR}/sph/pcisph/CMakeLists.txt)

set(CUDA_SPH_FILES 
	./sph/sph_cuda.cu
	${CUDA_PCISPH_FILES})

set(SPH_FILES
	./sph/sph.cpp
	${PCISPH_FILES})
