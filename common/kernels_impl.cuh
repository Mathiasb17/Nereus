#ifndef KERNELS_IMPL_CUH
#define KERNELS_IMPL_CUH

#include <cuda_runtime_api.h>
#include <cuda.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <helper_cuda.h>
#include <helper_cuda_gl.h>

#include <helper_functions.h>


#include <helper_math.h>
#include <math.h>

#include "common.h"

EXTERN_C_BEGIN
//==================================================================================================== 
//==================================================================================================== 
//==================================================================================================== 

__forceinline__ __device__ void debug3(const char* valname, SVec3 val)
{
	if(( length(val) != length(val) ) || isinf(length(val)))
	{
		printf("%s = %8f %8f %8f\n", valname, val.x, val.y, val.z);
	}
}

__forceinline__ __device__ void debug(const char* valname, SReal val)
{	
	if ( (val != val) || isinf(val)) 
	{
		printf("%s = %f\n", valname, val);
	}
}

//==================================================================================================== 
//==================================================================================================== 
//==================================================================================================== 
struct comp
{
	__host__ __device__ bool operator()(SVec3 p1, SVec3 p2)
	{
		return length(p1) < length(p2);
	}
};

//==================================================================================================== 
//==================================================================================================== 
//==================================================================================================== 
__forceinline__ __device__ __host__ SReal Wdefault(SVec3 r, SReal h, SReal kpoly)
{
	SReal l_r = length(r);

	if (l_r > h)
	{
		return 0.f ;
	}

	SReal b = pow((h * h - l_r * l_r), 3);

	return (kpoly * b);
}

//==================================================================================================== 
//==================================================================================================== 
//==================================================================================================== 
__forceinline__ __device__ __host__ SVec3 Wdefault_grad(SVec3 r, SReal h, SReal kpoly_grad)
{
	SReal l_r = length(r);

	if (l_r > h) 
	{
		return make_SVec3(0.f, 0.f, 0.f);
	}

	SReal b = powf(h*h - l_r*l_r, 2);

	return kpoly_grad*r*b;
}

//==================================================================================================== 
//==================================================================================================== 
//==================================================================================================== 
__forceinline__ __device__ __host__ SVec3 Wpressure_grad(SVec3 r, SReal h, SReal kpress_grad)
{
	SReal l_r = length(r);

	if (l_r > h) 
	{
		return make_SVec3(0.f, 0.f, 0.f);
	}

	SReal c = (h - l_r)*(h - l_r);

	return kpress_grad * (r/l_r) * c;
}

//==================================================================================================== 
//==================================================================================================== 
//==================================================================================================== 
__forceinline__ __device__ __host__ SVec3 Wviscosity_grad(SVec3 r, SReal h, SReal kvisc_grad, SReal kvisc_denum)
{
	SReal l_r = length(r);

	if (l_r > h) 
	{
		return make_SVec3(0.f, 0.f, 0.f);
	}

	SReal c = -(3*l_r / kvisc_denum ) + ( 2/(h*h) ) - ( h / (2*l_r*l_r*l_r));

	return kvisc_grad * r * c;
}

//==================================================================================================== 
//==================================================================================================== 
//==================================================================================================== 
__forceinline__ __device__ __host__ SReal Wmonaghan(SVec3 r, SReal h)
{
	SReal value = 0.f;
	SReal m_invH = 1.f  / h;
	SReal m_v = 1.0/(4.0*M_PI*h*h*h);
    SReal q = length(r)*m_invH;
    if( q >= 0 && q < 1 )
    {
        value = m_v*( (2-q)*(2-q)*(2-q) - 4.0f*(1-q)*(1-q)*(1-q));
    }
    else if ( q >=1 && q < 2 )
    {
        value = m_v*( (2-q)*(2-q)*(2-q) );
    }
    else
    {
        value = 0.0f;
    }
    return value;
}

//==================================================================================================== 
//==================================================================================================== 
//==================================================================================================== 
__forceinline__ __device__ __host__ SVec3 Wmonaghan_grad(SVec3 r, SReal h)
{

    SReal m_g = 1.0/(4.0*M_PI*h*h*h);
	SReal dist = length(r);
	SReal m_invH = 1.f/h;
    SReal q = dist*m_invH;
    SVec3 gradient = make_SVec3(0.f, 0.f, 0.f);
    if( q >= 0 && q < 1 )
    {
        SReal scalar = -3.0f*(2-q)*(2-q);
        scalar += 12.0f*(1-q)*(1-q);
        gradient = (m_g*m_invH*scalar/dist)*r;
    }
    else if ( q >=1 && q < 2 )
    {
        SReal scalar = -3.0f*(2-q)*(2-q);
        gradient = (m_g*scalar*m_invH/dist)*r;
    }
	return gradient;
}

//==================================================================================================== 
//==================================================================================================== 
//==================================================================================================== 
__forceinline__ __device__ __host__ SReal Cakinci(SVec3 r, SReal h, SReal ksurf1, SReal ksurf2)
{
	SReal len = length(r);
	SReal poly = ksurf1;
	SReal hr = h - len;
	if (2.f*len > h && len <= h) 
	{
		SReal a = (hr*hr*hr) * (len*len*len);
		return poly*a;
	}
	else if (len > 0.f && 2*len <= h) 
	{
		SReal a = 2 * (hr*hr*hr) * (len*len*len);
		SReal b = ksurf2;
		return  poly * (a-b);
	}
	else
	{
		return 0.f;
	}
}

//==================================================================================================== 
//==================================================================================================== 
//==================================================================================================== 
__forceinline__ __device__ __host__ SReal Aboundary(SVec3 r, SReal h, SReal bpol)
{
	SReal rl = length(r);
	if (2.f*rl > h && rl <= h) 
	{
		SReal a = -((4*(rl*rl))/(h));
		SReal b = (6.f*rl - 2.f*h);
		SReal res = powf(a + b, 1.f/4.f);
		return bpol*res;
	}
	else
	{
		return 0.f;
	}
}

EXTERN_C_END

#endif /* ifndef KERNELS_IMPL_CUH */
