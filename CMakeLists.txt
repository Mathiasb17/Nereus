cmake_minimum_required(VERSION 3.5.1)

#c++ 11, always before project to avoid infinite loop
set (CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_COMPILER "/usr/bin/g++-4.9")

project(sph)

#BEGIN SUBMODULES DEPENDENCIES
if (NOT IS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/external/sph_boundary_particles/src")
  message(FATAL_ERROR "The sph boundary particles dependency repositories (GLFW, etc.) are missing! "
    "You probably did not clone the project with --recursive. It is possible to recover "
    "by calling \"git submodule update --init --recursive\"")
endif()

add_subdirectory("./external/sph_boundary_particles" "ext_build/sph_boundary_particles")
include_directories(./external/sph_boundary_particles/include)

#END SUBMODULES DEPENDENCIES


#for autocomplete database
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

#maybe copy used files in project to avoid doing this horror
include_directories(/usr/local/cuda/include/)
include_directories(/usr/local/cuda/samples/common/inc)

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
set(NVCC_FLAGS_EXTRA ${NVCC_FLAGS_EXTRA} -D_FORCE_INLINES)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-gencode arch=compute_35,code=sm_35 -gencode arch=compute_52,code=sm_52 --use_fast_math -Xptxas -O3 --maxrregcount 40") 

message(${CUDA_NVCC_FLAGS})

CUDA_INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR})

##COMPILE SIMULATION KERNELS
set(CUDA_SOURCES  ${CUDA_SPH_FILES})
CUDA_COMPILE(CUDA_OBJ ${CUDA_SOURCES}) 

add_executable(sph main.cpp ${SPH_FILES} ${CUDA_OBJ})

target_link_libraries(sph ${OPENGL_gl_LIBRARY})
target_link_libraries(sph ${OPENGL_LIBRARIES})
target_link_libraries(sph glfw ${GLFW_LIBRARIES})
target_link_libraries(sph ${GLEW_LIBRARIES})
target_link_libraries(sph ${CUDA_LIBRARIES})
target_link_libraries(sph ss)
