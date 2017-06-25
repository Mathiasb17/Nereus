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
	m_gridSortBits(32)
{
	std::cout << "construction of sph based system" << std::endl;

	/********************
	*  SPH PARAMETERS  *
	********************/
	m_params.gasStiffness = 600;
	m_params.restDensity = 1000;
	m_params.particleRadius = 0.02;
	m_params.timestep = 1E-3;
	m_params.viscosity = 0.003f;
	m_params.surfaceTension = 0.02;

	m_params.gravity.x = 0.;
	m_params.gravity.y = 0.;
	m_params.gravity.y = -9.81;
	m_params.gravity.z = 0.;

	m_params.interactionRadius = 0.0547;//better !
	m_params.particleMass = powf(m_params.interactionRadius, 3)*m_params.restDensity;

	m_params.beta = 600.0;

	/*********************
	*  GRID PARAMETERS  *
	*********************/
	m_params.worldOrigin = make_SVec3(-1.1,-1.1,-1.1); //slight offset to avoid particles off the domain
	m_params.gridSize = make_uint3(128,128,128); // power of 2
	m_params.cellSize = make_SVec3(m_params.interactionRadius, m_params.interactionRadius, m_params.interactionRadius);
	m_params.numCells = m_params.gridSize.x * m_params.gridSize.y * m_params.gridSize.z;

	/****************************************
	*  SMOOTHING KERNELS PRE-COMPUTATIONS  *
	****************************************/
	m_params.kpoly = 315.0 / (64.0 * M_PI * powf(m_params.interactionRadius, 9.0));
	
	m_params.kpoly_grad = -945.0/(32.0*M_PI*powf(m_params.interactionRadius, 9.0));
	m_params.kpress_grad = -45.0/(M_PI*powf(m_params.interactionRadius, 6.0));

	m_params.kvisc_grad = 15.0 / (2*M_PI*powf(m_params.interactionRadius, 3.0));
	m_params.kvisc_denum = 2.0*powf(m_params.interactionRadius, 3.0);

	m_params.ksurf1 = 32.0/(M_PI * powf(m_params.interactionRadius,9));
	m_params.ksurf2 = powf(m_params.interactionRadius,6)/64.0;

	m_params.bpol = 0.007f / (powf(m_params.interactionRadius, 3.25));
	
	_intialize();
	m_numParticles = 0;
}

//==================================================================================================== 
//==================================================================================================== 
//==================================================================================================== 
SPH::SPH (SphSimParams params):
	m_gridSortBits(32),
	m_params(params)
{
	/****************************************
	*  SMOOTHING KERNELS PRE-COMPUTATIONS  *
	****************************************/
	m_params.kpoly = 315.0 / (64.0 * M_PI * powf(m_params.interactionRadius, 9.0));
	
	m_params.kpoly_grad = -945.0/(32.0*M_PI*powf(m_params.interactionRadius, 9.0));
	m_params.kpress_grad = -45.0/(M_PI*powf(m_params.interactionRadius, 6.0));

	m_params.kvisc_grad = 15.0 / (2*M_PI*powf(m_params.interactionRadius, 3.0));
	m_params.kvisc_denum = 2.0*powf(m_params.interactionRadius, 3.0);

	m_params.ksurf1 = 32.0/(M_PI * powf(m_params.interactionRadius,9));
	m_params.ksurf2 = powf(m_params.interactionRadius,6)/64.0;

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
	unsigned int memSize = sizeof(SReal) * 4 * MAX_PARTICLE_NUMBER;
	unsigned int memSizeFloat = sizeof(SReal) * MAX_PARTICLE_NUMBER;
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
	SReal lambda = 0.4f;
	SReal ir = m_params.interactionRadius;

	if (length(res) > 0.0) 
	{
		SReal newDeltat = lambda * (ir / length(res));
		m_params.timestep = newDeltat;
		std::cout << "new timestep is " << newDeltat << std::endl;
	}
#endif	

	cudaMemcpy(m_dpos, m_pos, sizeof(SReal)*4*m_numParticles,cudaMemcpyHostToDevice);
	cudaMemcpy(m_dvel, m_vel, sizeof(SReal)*4*m_numParticles,cudaMemcpyHostToDevice);

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

	cudaMemcpy(m_pos, m_dSortedPos, sizeof(SReal)*4*m_numParticles,cudaMemcpyDeviceToHost);
	cudaMemcpy(m_vel, m_dSortedVel, sizeof(SReal)*4*m_numParticles,cudaMemcpyDeviceToHost);
}

//==================================================================================================== 
//==================================================================================================== 
//==================================================================================================== 
void SPH::computeGridMinMax()
{

}

//==================================================================================================== 
//==================================================================================================== 
//==================================================================================================== 
void SPH::addNewParticle(SVec4 p, SVec4 v)
{
	m_pos[m_numParticles*4+0] =  p.x;
	m_pos[m_numParticles*4+1] =  p.y;
	m_pos[m_numParticles*4+2] =  p.z;
	m_pos[m_numParticles*4+3] =  p.w;

	m_density[m_numParticles] = 0.0;

	m_pressure[m_numParticles] = 0.0;

	m_vel[m_numParticles*4+0] =  v.x;
	m_vel[m_numParticles*4+1] =  v.y;
	m_vel[m_numParticles*4+2] =  v.z;
	m_vel[m_numParticles*4+3] =  v.w;

	m_forces[m_numParticles*4+0] =  0.0;
	m_forces[m_numParticles*4+1] =  0.0;
	m_forces[m_numParticles*4+2] =  0.0;
	m_forces[m_numParticles*4+3] =  0.0;

	m_colors[m_numParticles*4+0] =  1.0;
	m_colors[m_numParticles*4+1] =  0.0;
	m_colors[m_numParticles*4+2] =  0.0;
	m_colors[m_numParticles*4+3] =  1.0;

	m_numParticles += 1;
}

//==================================================================================================== 
//==================================================================================================== 
//==================================================================================================== 
void SPH::generateParticleCube(SVec4 center, SVec4 size, SVec4 vel)
{
	for(SReal x = center.x-size.x/2.0; x <= center.x+size.x/2.0; x += m_params.interactionRadius-0.005f )
	{
		for(SReal y = center.y-size.y/2.0; y <= center.y+size.y/2.0; y += m_params.interactionRadius-0.005f )
		{
			for(SReal z = center.z-size.z/2.0; z <= center.z+size.z/2.0; z += m_params.interactionRadius-0.005f )
			{
				addNewParticle(make_SVec4(x,y,z,1.0), vel);
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

	cudaMalloc((void**)&m_dSortedbi, 4*sizeof(SReal)*nb_boundary_spheres);
	cudaMalloc((void**)&m_dSortedVbi, sizeof(SReal)*nb_boundary_spheres);

	cudaMalloc((void**)&m_dbi, 4*sizeof(SReal)*nb_boundary_spheres);
	cudaMalloc((void**)&m_dvbi, sizeof(SReal)*nb_boundary_spheres);

	cudaMalloc((void**)&m_dGridBoundaryIndex, sizeof(unsigned int)*nb_boundary_spheres);
	cudaMalloc((void**)&m_dGridBoundaryHash, sizeof(unsigned int)*nb_boundary_spheres);

	cudaMalloc((void**)&m_dBoundaryCellEnd, sizeof(unsigned int)*getNumCells());
	cudaMalloc((void**)&m_dBoundaryCellStart, sizeof(unsigned int)*getNumCells());

	cudaMemcpy(m_dbi, m_bi, 4*sizeof(SReal)*nb_boundary_spheres, cudaMemcpyHostToDevice);
	cudaMemcpy(m_dvbi, m_vbi, sizeof(SReal)*nb_boundary_spheres, cudaMemcpyHostToDevice);

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
