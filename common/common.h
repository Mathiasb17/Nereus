#ifndef COMMON_H
#define COMMON_H

#include <cuda_runtime.h>

/****************************
*  KERNEL SET IDENTIFIERS  *
****************************/
#define MONAGHAN 0
#define MULLER   1

typedef unsigned int SUint;

/**********************************
*  SIMPLE OR DOUBLE PRECISION	  *	
*  (TUNE IN CMakeLists.txt)       *
***********************************/
#if DOUBLE_PRECISION == 1

typedef double SReal;
typedef double3 SVec3;
typedef double4 SVec4;
#define make_SVec3 make_double3
#define make_SVec4 make_double4
//OpenGL primitives
#define GL_REAL GL_DOUBLE

#else

typedef float SReal;
typedef float3 SVec3;
typedef float4 SVec4;
#define make_SVec3 make_float3
#define make_SVec4 make_float4
//OpenGL primitives
#define GL_REAL GL_FLOAT

#endif

/***********************
*  NAMESPACES MACROS  *
***********************/
#define NEREUS_NAMESPACE_BEGIN namespace NEREUS {
#define NEREUS_NAMESPACE_END   }

#define EXTERN_C_BEGIN extern "C" {
#define EXTERN_C_END   }

#endif /* ifndef COMMON_H */
