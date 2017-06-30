#include "pcisph.h"

#include "sph.cuh"
#include "sph_kernel.cuh"

#include <iostream>
#include <algorithm>

#include <glm/glm.hpp>

#include <thrust/host_vector.h>
#include <thrust/functional.h>
#include <thrust/extrema.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <thrust/sort.h>

#include <helper_functions.h>
#include <helper_cuda.h>

#include <cuda_runtime.h>

namespace CFD
{
//==================================================================================================== 
//==================================================================================================== 
//==================================================================================================== 
PCISPH::PCISPH():
	SPH()
{
	std::cout << GREEN << "construction of pcisph based system" << RESET << std::endl;

	//OVERIDING THE FOLLOWING AS FOR NOW FIXME
	/********************
	*  SPH PARAMETERS  *
	********************/
	m_params.restDensity = 1000.0;
	m_params.particleRadius = 0.02;
	m_params.timestep = 1e-3;
	m_params.viscosity = 0.005;
	m_params.surfaceTension = 0.0001;

	m_params.gravity.x = 0.0;
	m_params.gravity.y = 0.0;
	m_params.gravity.y = -9.81f;
	m_params.gravity.z = 0.0;

	m_params.interactionRadius = 0.0537;//better !
	m_params.particleMass = powf(m_params.interactionRadius, 3)*m_params.restDensity;

	m_params.beta = 650.0;

	const SReal eta = 0.01;
    const SReal H = 0.1;
    const SReal vf = sqrt( 2*9.81*H );
    m_params.soundSpeed = vf/(sqrt(eta));

	/*********************
	*  GRID PARAMETERS  *
	*********************/
	m_params.worldOrigin = make_SVec3(-1.2,-1.2,-1.2); //slight offset to avoid particles off the domain
	m_params.gridSize = make_uint3(128,128,128); // power of 2

	m_params.cellSize = make_SVec3(m_params.interactionRadius, m_params.interactionRadius, m_params.interactionRadius);
	m_params.numCells = m_params.gridSize.x * m_params.gridSize.y * m_params.gridSize.z;

	/****************************************
	*  SMOOTHING KERNELS PRE-COMPUTATIONS  *
	****************************************/
	m_params.kpoly = 315.0 / (64.0 * M_PI * powf(m_params.interactionRadius, 9.0));
	
	m_params.kpoly_grad  = -945.0/(32.0*M_PI*powf(m_params.interactionRadius, 9.0));
	m_params.kpress_grad = -45.0/(M_PI*powf(m_params.interactionRadius, 6.0));

	m_params.kvisc_grad  = 15.0 / (2*M_PI*powf(m_params.interactionRadius, 3.0));
	m_params.kvisc_denum = 2.0*powf(m_params.interactionRadius, 3.0);

	m_params.ksurf1 = 32.0/(M_PI * powf(m_params.interactionRadius,9));
	m_params.ksurf2 = powf(m_params.interactionRadius,6)/64.0;
	m_params.bpol = 0.007f / (powf(m_params.interactionRadius, 3.25));
	
	/**********
	*  INIT  *
	**********/
	_initialize();
	m_numParticles = 0;

}
//=====================================================================================================   
//=====================================================================================================   
//=====================================================================================================   
PCISPH::PCISPH (SphSimParams params):
	SPH(params)
{
	/****************************************
	*  SMOOTHING KERNELS PRE-COMPUTATIONS  *
	****************************************/
	m_params.kpoly = 315.0 / (64.0 * (SReal)M_PI  * powf(m_params.interactionRadius, 9.0));
	
	m_params.kpoly_grad = -945.0/(32.0*(SReal)M_PI *powf(m_params.interactionRadius, 9.0));
	m_params.kpress_grad = -45.0/((SReal)M_PI *powf(m_params.interactionRadius, 6.0));

	m_params.kvisc_grad = 15.0 / (2*(SReal)M_PI *powf(m_params.interactionRadius, 3.0));
	m_params.kvisc_denum = 2.0*powf(m_params.interactionRadius, 3.0);

	m_params.ksurf1 = 32.0/((SReal)M_PI  * powf(m_params.interactionRadius,9));
	m_params.ksurf2 = powf(m_params.interactionRadius,6)/64.0;

	m_params.bpol = 0.007f / (powf(m_params.interactionRadius, 3.25));
	
	_initialize();
	m_numParticles = 0;
}
//==================================================================================================== 
//==================================================================================================== 
//==================================================================================================== 
PCISPH::~PCISPH()
{

}
//==================================================================================================== 
//==================================================================================================== 
//==================================================================================================== 
void PCISPH::_initialize()
{
	SPH::_initialize();

	unsigned int memSize = sizeof(SReal) * 4 * MAX_PARTICLE_NUMBER;
	unsigned int memSizeFloat = sizeof(SReal) * MAX_PARTICLE_NUMBER;

	allocateArray((void**)&m_dSortedRhoAdv, memSizeFloat);
	allocateArray((void**)&m_dSortedVelAdv, memSize);

	cudaMemset(m_dSortedRhoAdv, 0, memSizeFloat);
	cudaMemset(m_dSortedVelAdv, 0, memSize);

	setParameters(&m_params);
}
//==================================================================================================== 
//==================================================================================================== 
//==================================================================================================== 
void PCISPH::_finalize()
{
	SPH::_finalize();
}
//==================================================================================================== 
//==================================================================================================== 
//==================================================================================================== 
void PCISPH::update()
{
	std::cout << RED << "IMPLEMENTATION INCOMING" << std::endl;
}
//==================================================================================================== 
//==================================================================================================== 
//==================================================================================================== 
}
