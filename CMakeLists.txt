cmake_minimum_required(VERSION 3.5.1)
project(sph)

#c++ 11
set (CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_COMPILER "/usr/bin/g++-4.9")

#for autocomplete database
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

#LIBS
find_package(glfw3 3.2 REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)

file(GLOB SYSTEM_SOURCES sph.h sph.cpp)

set(SOURCES ${SYSTEM_SOURCES})

add_executable(sph main.cpp ${SOURCES})

#target_include_directories(sph ${OPENGL_INCLUDE_DIR})
target_link_libraries(sph ${OPENGL_gl_LIBRARY})
target_link_libraries(sph ${OPENGL_LIBRARIES})
target_link_libraries(sph glfw ${GLFW_LIBRARIES})
target_link_libraries(sph ${GLEW_LIBRARIES})
