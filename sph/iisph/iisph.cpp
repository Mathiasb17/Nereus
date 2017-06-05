#include "iisph.h"

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

IISPH::IISPH():
	SPH()
{

}

IISPH::~IISPH()
{

}

void IISPH::_intialize()
{
	SPH::_intialize();

	unsigned int memSize = sizeof(float) * 4 * MAX_PARTICLE_NUMBER;
	unsigned int memSizeFloat = sizeof(float) * MAX_PARTICLE_NUMBER;

	cudaMallocHost((void**)&m_dVel_adv, memSize);
	cudaMallocHost((void**)&m_dDensity_adv, memSizeFloat);
	cudaMallocHost((void**)&m_dDisplacement_factor, memSize);
	cudaMallocHost((void**)&m_dAdvection_factor, memSizeFloat);

	allocateArray((void **)&m_dVel_adv, memSize);
	allocateArray((void **)&m_dDensity_adv, memSizeFloat);
	allocateArray((void **)&m_dDisplacement_factor, memSize);
	allocateArray((void **)&m_dAdvection_factor, memSizeFloat);

	cudaMemset(m_dVel_adv, 0, memSize);
	cudaMemset(m_dDensity_adv, 0, memSizeFloat);
	cudaMemset(m_dDisplacement_factor, 0, memSize);
	cudaMemset(m_dAdvection_factor, 0, memSizeFloat);

	setParameters(&m_params);
}

void IISPH::_finalize()
{
	SPH::_finalize();
}

void IISPH::update()
{
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

	//TODO 
	predictAdvection(
		m_dCellStart,
		m_dCellEnd,
		m_dBoundaryCellStart,
		m_dBoundaryCellEnd,
		m_dbi,
		m_vbi,
		m_dSortedPos,
		m_dSortedVel,
		m_dVel_adv,
		m_dSortedDens,
		m_dDensity_adv,
		m_dSortedPress,
		m_dSortedForces,
		m_dDisplacement_factor,
		m_dAdvection_factor,
		m_dGridParticleHash,
		m_dGridParticleIndex,
		m_dGridBoundaryHash,
		m_dGridBoundaryIndex,
		m_numParticles,
		m_num_boundaries,
		getNumCells()
		);
	//predict Advection and pressure solve
	

}

} /* CFD */ 
