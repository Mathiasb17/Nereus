#ifndef KERNELS_IMPL_CUH
#define KERNELS_IMPL_CUH

__device__ float Wdefault(float3 r, float h)
{
	float l_r = length(r);
	if(l_r > h) return 0.f;
	float a = (315.f / (64.f * M_PI * powf(h,9.f)));
	float b = powf(h*h - l_r*l_r,3.f);
	return a * b;
}

/*__device__ float3 Wdefault_grad(float3 r, float h)*/
//{

//}

//__device__ float3 Wpressure_grad(float3 r, float h)
//{

//}

//__device__ float Wviscosity_laplacian(float3 r, float h)
//{

//}

//float SPH::Wdefault(glm::vec3 r, float h)
//{
	//}

//glm::vec3 SPH::Wdefault_grad(glm::vec3 r, float h)
//{
	//float l_r = length(r);
	//float a = -945/(32*M_PI*powf(h,9));
	//float b = powf(h*h - l_r*l_r, 2);
	//return a*r*b;
//}

//glm::vec3 SPH::Wpressure_grad(glm::vec3 r, float h)
//{
	//float l_r = glm::length(r);
	//float a = -(45.f/ (M_PI * powf(h,6.f)));
	//glm::vec3 b = r / l_r;
	//float c = (h-l_r)*(h-l_r);
	//return a*b*c;
//}

//float SPH::Wviscosity_laplacian(glm::vec3 r, float h)
//{
	//float l_r = glm::length(r);
	//float a = 45 / (M_PI * powf(h,6));
	//float b = h-l_r;
	//return a*b;
/*}*/



#endif /* ifndef KERNELS_IMPL_CUH */
