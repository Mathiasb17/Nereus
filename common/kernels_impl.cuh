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

__device__ __host__ float3 Wdefault_grad(float3 r, float h, float kpoly_grad)
{
	float l_r = length(r);

	float b = powf(h*h - l_r*l_r, 2);

	return kpoly_grad*r*b;
}

__device__ __host__ float3 Wpressure_grad(float3 r, float h, float kpress_grad)
{
	float l_r = length(r);

	float c = (h - l_r)*(h - l_r);

	return kpress_grad * (r/l_r) * c;
}

__device__ __host__ float3 Wviscosity_grad(float3 r, float h, float kvisc_grad, float kvisc_denum)
{
	float l_r = length(r);

	float c = -(3*l_r / kvisc_denum ) + ( 2/(h*h) ) - ( h / (2*l_r*l_r*l_r));

	return kvisc_grad * r * c;
}

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

}

#endif /* ifndef KERNELS_IMPL_CUH */
