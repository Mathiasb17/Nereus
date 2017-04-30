cmake_minimum_required(VERSION 3.5.1)
project(sph)

#c++ 11
set (CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_COMPILER "/usr/bin/g++-4.9")

#for autocomplete database
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

#maybe copy used files in project to avoid doing this horror
include_directories(/usr/local/cuda-7.5/include/)
include_directories(/usr/local/cuda-7.5/samples/common/inc)

#LIBS
find_package(glfw3 3.2 REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_package(CUDA REQUIRED)

#include project subdirs
include(${CMAKE_CURRENT_SOURCE_DIR}/sph/CMakeLists.txt)
include(${CMAKE_CURRENT_SOURCE_DIR}/common/CMakeLists.txt)

#project sources
set(SOURCES ${SPH_FILES})

#CUDA OPTIONS
set(CUDA_64_BIT_DEVICE_CODE ON)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-gencode arch=compute_35,code=sm_35 -gencode arch=compute_52,code=sm_52 --use_fast_math --maxrregcount 192") 

message(${CUDA_NVCC_FLAGS})

CUDA_INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR})

##COMPILE SIMULATION KERNELS
set(CUDA_SOURCES  ${CUDA_SPH_FILES})
CUDA_COMPILE(CUDA_OBJ ${CUDA_SOURCES}) 

add_executable(sph main.cpp ${SPH_FILES} ${CUDA_OBJ})

#target_include_directories(sph ${OPENGL_INCLUDE_DIR})
target_link_libraries(sph ${OPENGL_gl_LIBRARY})
target_link_libraries(sph ${OPENGL_LIBRARIES})
target_link_libraries(sph glfw ${GLFW_LIBRARIES})
target_link_libraries(sph ${GLEW_LIBRARIES})
target_link_libraries(sph ${CUDA_LIBRARIES})
