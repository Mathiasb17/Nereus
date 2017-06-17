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

extern "C"
{

//==================================================================================================== 
//==================================================================================================== 
//==================================================================================================== 

__device__ void debug3(const char* valname, float3 val)
{
	if(( length(val) != length(val) ) || isinf(length(val)))
	{
		printf("%s = %8f %8f %8f\n", valname, val.x, val.y, val.z);
	}
}

__device__ void debug(const char* valname, float val)
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
	__host__ __device__ bool operator()(float3 p1, float3 p2)
	{
		return length(p1) < length(p2);
	}
};

//==================================================================================================== 
//==================================================================================================== 
//==================================================================================================== 
__device__ __host__ float Wdefault(float3 r, float h, float kpoly)
{
	float l_r = length(r);

	if (l_r > h)
	{
		return 0.f ;
	}

	float b = pow((h * h - l_r * l_r), 3);

	return (kpoly * b);
}

//==================================================================================================== 
//==================================================================================================== 
//==================================================================================================== 
__device__ __host__ float3 Wdefault_grad(float3 r, float h, float kpoly_grad)
{
	float l_r = length(r);

	if (l_r > h) 
	{
		return make_float3(0.f, 0.f, 0.f);
	}

	float b = powf(h*h - l_r*l_r, 2);

	return kpoly_grad*r*b;
}

//==================================================================================================== 
//==================================================================================================== 
//==================================================================================================== 
__device__ __host__ float3 Wpressure_grad(float3 r, float h, float kpress_grad)
{
	float l_r = length(r);

	if (l_r > h) 
	{
		return make_float3(0.f, 0.f, 0.f);
	}

	float c = (h - l_r)*(h - l_r);

	return kpress_grad * (r/l_r) * c;
}

//==================================================================================================== 
//==================================================================================================== 
//==================================================================================================== 
__device__ __host__ float3 Wviscosity_grad(float3 r, float h, float kvisc_grad, float kvisc_denum)
{
	float l_r = length(r);

	if (l_r > h) 
	{
		return make_float3(0.f, 0.f, 0.f);
	}

	float c = -(3*l_r / kvisc_denum ) + ( 2/(h*h) ) - ( h / (2*l_r*l_r*l_r));

	return kvisc_grad * r * c;
}

//==================================================================================================== 
//==================================================================================================== 
//==================================================================================================== 
__device__ __host__ float Wmonaghan(float3 r, float h, float kp)
{
	float value = 0.f;
	float m_invH = 1.f  / h;
	float m_v = 1.0/(4.0*M_PI*h*h*h);
    float q = length(r)*m_invH;
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
__device__ __host__ float3 Wmonaghan_grad(float3 r, float h, float kpg)
{

    float m_g = 1.0/(4.0*M_PI*h*h*h);
	float dist = length(r);
	float m_invH = 1.f/h;
    float q = dist*m_invH;
    float3 gradient = make_float3(0.f, 0.f, 0.f);
    if( q >= 0 && q < 1 )
    {
        float scalar = -3.0f*(2-q)*(2-q);
        scalar += 12.0f*(1-q)*(1-q);
        gradient = (m_g*m_invH*scalar/dist)*r;
    }
    else if ( q >=1 && q < 2 )
    {
        float scalar = -3.0f*(2-q)*(2-q);
        gradient = (m_g*scalar*m_invH/dist)*r;
    }
	return gradient;
}

//==================================================================================================== 
//==================================================================================================== 
//==================================================================================================== 
__device__ __host__ float Cakinci(float3 r, float h, float ksurf1, float ksurf2)
{
	float len = length(r);
	float poly = ksurf1;
	float hr = h - len;
	if (2.f*len > h && len <= h) 
	{
		float a = (hr*hr*hr) * (len*len*len);
		return poly*a;
	}
	else if (len > 0.f && 2*len <= h) 
	{
		float a = 2 * (hr*hr*hr) * (len*len*len);
		float b = ksurf2;
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
__device__ __host__ float Aboundary(float3 r, float h, float bpol)
{
	float rl = length(r);
	if (2.f*rl > h && rl <= h) 
	{
		float a = -((4*(rl*rl))/(h));
		float b = (6.f*rl - 2.f*h);
		float res = powf(a + b, 1.f/4.f);
		return bpol*res;
	}
	else
	{
		return 0.f;
	}
}

}

#endif /* ifndef KERNELS_IMPL_CUH */
