cmake_minimum_required(VERSION 3.5.1)
project(sph)

#c++ 11
set (CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_COMPILER "/usr/bin/g++-4.9")

#for autocomplete database
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

include_directories(/usr/local/cuda-7.5/include/)
include_directories(/usr/local/cuda-7.5/samples/common/inc)

#LIBS
find_package(glfw3 3.2 REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_package(CUDA REQUIRED)

file(GLOB SYSTEM_SOURCES sph.h sph.cpp sph_cuda.cu)

set(SOURCES ${SYSTEM_SOURCES})

#CUDA OPTIONS
set(CUDA_64_BIT_DEVICE_CODE ON)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -gencode arch=compute_52,code=sm_52)
set(CUDA_SOURCES sph_kernel.cuh sph_cuda.cu)

CUDA_INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR})
CUDA_COMPILE(CUDA_OBJ ${CUDA_SOURCES})

add_executable(sph main.cpp ${SOURCES} ${CUDA_OBJ})

#target_include_directories(sph ${OPENGL_INCLUDE_DIR})
target_link_libraries(sph ${OPENGL_gl_LIBRARY})
target_link_libraries(sph ${OPENGL_LIBRARIES})
target_link_libraries(sph glfw ${GLFW_LIBRARIES})
target_link_libraries(sph ${GLEW_LIBRARIES})
target_link_libraries(sph ${CUDA_LIBRARIES})
