#ifndef KERNELS_IMPL_CUH
#define KERNELS_IMPL_CUH

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <helper_cuda.h>
#include <helper_cuda_gl.h>

#include <helper_functions.h>


#include <helper_math.h>
#include <math.h>

__device__ float Wdefault(float3 r, float h)
{
	float l_r = length(r);
	if (l_r > h) return 0.f ;
	float a = 315 / (64 * M_PI * pow(h, 9));
	float b = pow((h * h - l_r * l_r), 3);
	return (a * b);
}

__device__ float3 Wdefault_grad(float3 r, float h)
{
	float l_r = length(r);
	float a = -945.f/(32.f*M_PI*powf(h,9));
	float b = powf(h*h - l_r*l_r, 2);
	return a*r*b;
}

/*__device__ float3 Wpressure_grad(float3 r, float h)*/
//{
	//float l_r = length(r);
	//if(l_r > h) return make_float3(0.f,0.f,0.f);
	//float a = -(45.f/ (M_PI * powf(h,6.f)));
	//float3 b = r / l_r;
	//float c = (h-l_r)*(h-l_r);
	//return a*b*c;
//}

//__device__ float Wviscosity_laplacian(float3 r, float h)
//{
	//float l_r = length(r);
	//if(l_r > h) return 0.f;
	//float a = 45.f / (M_PI * powf(h,6));
	//float b = h-l_r;
	////printf("Wdefault %5f\n", a);
	//return a*b;
/*}*/

#endif /* ifndef KERNELS_IMPL_CUH */
