#include "sph.h"
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
SPH::SPH ():
	m_gridSortBits(18)
{
	std::cout << "construction of sph based system" << std::endl;

	/********************
	*  SPH PARAMETERS  *
	********************/
	m_params.gasStiffness = 400.f;
	m_params.restDensity = 998.29;
	m_params.particleRadius = 0.02;
	m_params.timestep = 1E-3f;
	m_params.viscosity = 0.01f;
	m_params.surfaceTension = 0.005; //0.698f; 

	m_params.gravity.x = 0.f;
	//m_params.gravity.y = 0.f;
	m_params.gravity.y = -9.81f;
	m_params.gravity.z = 0.f;

	m_params.interactionRadius = 0.0457f;
	m_params.particleMass = powf(m_params.interactionRadius, 3)*m_params.restDensity;

	m_params.beta = 600.f;

	/*********************
	*  GRID PARAMETERS  *
	*********************/
	m_params.worldOrigin = make_float3(-1.1,-1.1,-1.1); //slight offset to avoid particles off the domain
	m_params.gridSize = make_uint3(64,64,64); // power of 2
	m_params.cellSize = make_float3(m_params.interactionRadius, m_params.interactionRadius, m_params.interactionRadius);
	m_params.numCells = m_params.gridSize.x * m_params.gridSize.y * m_params.gridSize.z;

	/****************************************
	*  SMOOTHING KERNELS PRE-COMPUTATIONS  *
	****************************************/
	m_params.kpoly = 315.f / (64.f * M_PI * powf(m_params.interactionRadius, 9.f));
	
	m_params.kpoly_grad = -945.f/(32.f*M_PI*powf(m_params.interactionRadius, 9.f));
	m_params.kpress_grad = -45.f/(M_PI*powf(m_params.interactionRadius, 6.f));

	m_params.kvisc_grad = 15.f / (2*M_PI*powf(m_params.interactionRadius, 3.f));
	m_params.kvisc_denum = 2.f*powf(m_params.interactionRadius, 3.f);

	m_params.ksurf1 = 32.f/(M_PI * powf(m_params.interactionRadius,9));
	m_params.ksurf2 = powf(m_params.interactionRadius,6)/64.f;

	m_params.bpol = 0.007f / (powf(m_params.interactionRadius, 3.25));
	
	_intialize();
	m_numParticles = 0;
}

//==================================================================================================== 
//==================================================================================================== 
//==================================================================================================== 
SPH::~SPH ()
{

}

//==================================================================================================== 
//==================================================================================================== 
//==================================================================================================== 
void SPH::_intialize()
{
	unsigned int memSize = sizeof(float) * 4 * MAX_PARTICLE_NUMBER;
	unsigned int memSizeFloat = sizeof(float) * MAX_PARTICLE_NUMBER;
	unsigned int memSizeUint = sizeof(unsigned int) * m_params.numCells;

	/*******************
	 *  HOST MEM INIT  *
	 *******************/
	cudaMallocHost((void**)&m_pos, memSize);
	cudaMallocHost((void**)&m_vel, memSize);
	cudaMallocHost((void**)&m_density, memSizeFloat);
	cudaMallocHost((void**)&m_pressure, memSizeFloat);
	cudaMallocHost((void**)&m_forces, memSize);
	cudaMallocHost((void**)&m_colors, memSize);

	cudaMallocHost((void**)&m_hParticleHash, sizeof(unsigned int)* MAX_PARTICLE_NUMBER);
	cudaMallocHost((void**)&m_hCellStart, memSizeUint);
	cudaMallocHost((void**)&m_hCellEnd, memSizeUint);

	memset(m_hCellStart, 0, m_params.numCells*sizeof(uint));
    memset(m_hCellEnd, 0, m_params.numCells*sizeof(uint));

	/******************
	 *  GPU MEM INIT  *
	 ******************/
	allocateArray((void **)&m_dpos, memSize);
	allocateArray((void **)&m_dvel, memSize);
	allocateArray((void **)&m_ddensity, memSizeFloat);
	allocateArray((void **)&m_dpressure, memSizeFloat);
	allocateArray((void **)&m_dforces, memSize);
	allocateArray((void **)&m_dcolors, memSize);

	allocateArray((void **)&m_dSortedPos, memSize);
	allocateArray((void **)&m_dSortedVel, memSize);
	allocateArray((void **)&m_dSortedDens, memSizeFloat);
	allocateArray((void **)&m_dSortedPress, memSizeFloat);
	allocateArray((void **)&m_dSortedForces, memSize);
	allocateArray((void **)&m_dSortedCol, memSize);

	allocateArray((void **)&m_dGridParticleHash, MAX_PARTICLE_NUMBER*sizeof(unsigned int));
	allocateArray((void **)&m_dGridParticleIndex, MAX_PARTICLE_NUMBER*sizeof(unsigned int));

	allocateArray((void **)&m_dCellStart, memSizeUint);
	allocateArray((void **)&m_dCellEnd, memSizeUint);

	cudaMemset(m_dpos, 0, memSize);
	cudaMemset(m_dvel, 0, memSize);
	cudaMemset(m_ddensity, 0, memSizeFloat);
	cudaMemset(m_dpressure, 0, memSizeFloat);
	cudaMemset(m_dforces, 0, memSize);
	cudaMemset(m_dcolors, 0, memSize);

	cudaMemset(m_dSortedPos, 0, memSize);
	cudaMemset(m_dSortedVel, 0, memSize);
	cudaMemset(m_dSortedDens, 0, memSizeFloat);
	cudaMemset(m_dSortedPress, 0, memSizeFloat);
	cudaMemset(m_dSortedForces, 0, memSize);
	cudaMemset(m_dSortedCol, 0, memSize);

	setParameters(&m_params);
}

//==================================================================================================== 
//==================================================================================================== 
//==================================================================================================== 
void SPH::_finalize()
{

}

//==================================================================================================== 
//==================================================================================================== 
//==================================================================================================== 
void SPH::update()
{
#if 0
	/*****************************************
	*  compute timestep with CFL condition  *
	*****************************************/
	
	float3 res = maxVelocity(m_dSortedVel, m_numParticles);
	float lambda = 0.4f;
	float ir = m_params.interactionRadius;

	if (length(res) > 0.f) 
	{
		float newDeltat = lambda * (ir / length(res));
		m_params.timestep = newDeltat;
		std::cout << "new timestep is " << newDeltat << std::endl;
	}
#endif	

	cudaMemcpy(m_dpos, m_pos, sizeof(float)*4*m_numParticles,cudaMemcpyHostToDevice);
	cudaMemcpy(m_dvel, m_vel, sizeof(float)*4*m_numParticles,cudaMemcpyHostToDevice);

	setParameters(&m_params);

	calcHash( m_dGridParticleHash, m_dGridParticleIndex, m_dpos, m_numParticles);

	sortParticles(m_dGridParticleHash, m_dGridParticleIndex, m_numParticles);

	reorderDataAndFindCellStart(
		m_dCellStart,
		m_dCellEnd,
		m_dSortedPos,
		m_dSortedVel,
		m_dSortedDens,
		m_dSortedPress,
		m_dSortedForces,
		m_dSortedCol,
		m_dGridParticleHash,
		m_dGridParticleIndex,
		m_dpos,
		m_dvel,
		m_ddensity,
		m_dpressure,
		m_dforces,
		m_dcolors,
		m_numParticles,
		m_params.numCells);

	computeDensityPressure(
			m_dSortedPos,
			m_dSortedVel,
			m_dSortedDens,
			m_dSortedPress,
			m_dSortedForces,
			m_dSortedCol,
			m_dSortedbi,
			m_dSortedVbi,
			m_dGridParticleIndex,
			m_dCellStart,
			m_dCellEnd,
			m_dGridBoundaryIndex,
			m_dBoundaryCellStart,
			m_dBoundaryCellEnd,
			m_numParticles,
			m_params.numCells,
			m_num_boundaries);

	integrateSystem( m_dSortedPos, m_dSortedVel, m_dSortedForces, m_params.timestep, m_numParticles);

	cudaMemcpy(m_pos, m_dSortedPos, sizeof(float)*4*m_numParticles,cudaMemcpyDeviceToHost);
	cudaMemcpy(m_vel, m_dSortedVel, sizeof(float)*4*m_numParticles,cudaMemcpyDeviceToHost);
}

//==================================================================================================== 
//==================================================================================================== 
//==================================================================================================== 
void SPH::addNewParticle(glm::vec4 p, glm::vec4 v)
{
	m_pos[m_numParticles*4+0] =  p.x;
	m_pos[m_numParticles*4+1] =  p.y;
	m_pos[m_numParticles*4+2] =  p.z;
	m_pos[m_numParticles*4+3] =  p.w;

	m_density[m_numParticles] = 0.f;

	m_pressure[m_numParticles] = 0.f;

	m_vel[m_numParticles*4+0] =  v.x;
	m_vel[m_numParticles*4+1] =  v.y;
	m_vel[m_numParticles*4+2] =  v.z;
	m_vel[m_numParticles*4+3] =  v.w;

	m_forces[m_numParticles*4+0] =  0.f;
	m_forces[m_numParticles*4+1] =  0.f;
	m_forces[m_numParticles*4+2] =  0.f;
	m_forces[m_numParticles*4+3] =  0.f;

	m_colors[m_numParticles*4+0] =  1.f;
	m_colors[m_numParticles*4+1] =  0.f;
	m_colors[m_numParticles*4+2] =  0.f;
	m_colors[m_numParticles*4+3] =  1.f;

	m_numParticles += 1;
}

//==================================================================================================== 
//==================================================================================================== 
//==================================================================================================== 
void SPH::generateParticleCube(glm::vec4 center, glm::vec4 size, glm::vec4 vel)
{
	for(float x = center.x-size.x/2.f; x <= center.x+size.x/2.f; x += m_params.particleRadius*2 )
	{
		for(float y = center.y-size.y/2.f; y <= center.y+size.y/2.f; y += m_params.particleRadius*2 )
		{
			for(float z = center.z-size.z/2.f; z <= center.z+size.z/2.f; z += m_params.particleRadius*2 )
			{
				addNewParticle(glm::vec4(x,y,z,1.f), vel);
			}
		}
	}
	std::cout << "Il y a eu " << m_numParticles << " particules generees." << std::endl;
}

//==================================================================================================== 
//==================================================================================================== 
//==================================================================================================== 
void SPH::updateGpuBoundaries(unsigned int nb_boundary_spheres)
{
	cudaFree(m_dbi);
	cudaFree(m_dvbi);

	cudaMalloc((void**)&m_dSortedbi, 4*sizeof(float)*nb_boundary_spheres);
	cudaMalloc((void**)&m_dSortedVbi, sizeof(float)*nb_boundary_spheres);

	cudaMalloc((void**)&m_dbi, 4*sizeof(float)*nb_boundary_spheres);
	cudaMalloc((void**)&m_dvbi, sizeof(float)*nb_boundary_spheres);

	cudaMalloc((void**)&m_dGridBoundaryIndex, sizeof(unsigned int)*nb_boundary_spheres);
	cudaMalloc((void**)&m_dGridBoundaryHash, sizeof(unsigned int)*nb_boundary_spheres);

	cudaMalloc((void**)&m_dBoundaryCellEnd, sizeof(unsigned int)*getNumCells());
	cudaMalloc((void**)&m_dBoundaryCellStart, sizeof(unsigned int)*getNumCells());

	cudaMemcpy(m_dbi, m_bi, 4*sizeof(float)*nb_boundary_spheres, cudaMemcpyHostToDevice);
	cudaMemcpy(m_dvbi, m_vbi, sizeof(float)*nb_boundary_spheres, cudaMemcpyHostToDevice);

	calcHash( m_dGridBoundaryHash, m_dGridBoundaryIndex, m_dbi, m_num_boundaries);
	sortParticles(m_dGridBoundaryHash, m_dGridBoundaryIndex, m_num_boundaries);

	//appeler le kernel de tri des particules de bord
	reorderDataAndFindCellStartDBoundary(
			m_dBoundaryCellStart,
			m_dBoundaryCellEnd,
			m_dSortedbi,
			m_dSortedVbi,
			m_dGridBoundaryHash,
			m_dGridBoundaryIndex,
			m_dbi,
			m_dvbi,
			m_num_boundaries,
			getNumCells()
			);

	std::cout << "boundaries updated !" << std::endl;
}

} /* CFD */ 
