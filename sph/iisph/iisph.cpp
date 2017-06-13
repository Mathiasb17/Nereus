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
	std::cout << "construction of iisph based system" << std::endl;
	
}

IISPH::~IISPH()
{

}

void IISPH::_intialize()
{
	SPH::_intialize();

	unsigned int memSize = sizeof(float) * 4 * MAX_PARTICLE_NUMBER;
	unsigned int memSizeFloat = sizeof(float) * MAX_PARTICLE_NUMBER;

	allocateArray((void**)&m_dSortedDensAdv, memSizeFloat);
	allocateArray((void**)&m_dSortedDensCorr, memSizeFloat);
	allocateArray((void**)&m_dSortedP_l, memSizeFloat);
	allocateArray((void**)&m_dSortedPreviousP, memSizeFloat);
	allocateArray((void**)&m_dSortedAii, memSizeFloat);

	allocateArray((void**)&m_dSortedVelAdv, memSize);
	allocateArray((void**)&m_dSortedForcesAdv, memSize);
	allocateArray((void**)&m_dSortedForcesP, memSize);
	allocateArray((void**)&m_dSortedDiiFluid, memSize);
	allocateArray((void**)&m_dSortedDiiBoundary, memSize);
	allocateArray((void**)&m_dSortedSumDij, memSize);
	allocateArray((void**)&m_dSortedNormal, memSize);

	cudaMemset(m_dSortedDensAdv, 0, memSizeFloat);
	cudaMemset(m_dSortedDensCorr, 0, memSizeFloat);
	cudaMemset(m_dSortedP_l, 0, memSizeFloat);
	cudaMemset(m_dSortedPreviousP, 0, memSizeFloat);
	cudaMemset(m_dSortedAii, 0, memSizeFloat);

	cudaMemset(m_dSortedVelAdv, 0, memSize);
	cudaMemset(m_dSortedForcesAdv, 0, memSize);
	cudaMemset(m_dSortedForcesP, 0, memSize);
	cudaMemset(m_dSortedDiiFluid, 0, memSize);
	cudaMemset(m_dSortedDiiBoundary, 0, memSize);
	cudaMemset(m_dSortedSumDij, 0, memSize);
	cudaMemset(m_dSortedNormal, 0, memSize);

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
	cudaMemcpy(m_dpressure, m_pressure, sizeof(float)*m_numParticles,cudaMemcpyHostToDevice);

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


	predictAdvection(m_dSortedPos, m_dSortedVel, m_dSortedDens, m_dSortedPress, m_dSortedForces, m_dSortedCol, m_dCellStart, m_dCellEnd, m_dGridParticleIndex, m_dSortedbi, m_dSortedVbi,
					  m_dBoundaryCellStart, m_dBoundaryCellEnd, m_dGridBoundaryIndex, m_dSortedDensAdv, m_dSortedDensCorr, m_dSortedP_l,  m_dSortedPreviousP, 
					  m_dSortedAii, m_dSortedVelAdv, m_dSortedForcesAdv, m_dSortedForcesP, m_dSortedDiiFluid, m_dSortedDiiBoundary, m_dSortedSumDij, m_dSortedNormal,
					  m_numParticles, m_num_boundaries, getNumCells());


	pressureSolve(m_dSortedPos, m_dSortedVel, m_dSortedDens, m_dSortedPress, m_dSortedForces, m_dSortedCol, m_dCellStart, m_dCellEnd, m_dGridParticleIndex, m_dSortedbi, m_dSortedVbi,
					  m_dBoundaryCellStart, m_dBoundaryCellEnd, m_dGridBoundaryIndex, m_dSortedDensAdv, m_dSortedDensCorr, m_dSortedP_l,  m_dSortedPreviousP, 
					  m_dSortedAii, m_dSortedVelAdv, m_dSortedForcesAdv, m_dSortedForcesP, m_dSortedDiiFluid, m_dSortedDiiBoundary, m_dSortedSumDij, m_dSortedNormal,
					  m_numParticles, m_num_boundaries, getNumCells());

	cudaMemcpy(m_pos, m_dSortedPos, sizeof(float)*4*m_numParticles,cudaMemcpyDeviceToHost);
	cudaMemcpy(m_vel, m_dSortedVel, sizeof(float)*4*m_numParticles,cudaMemcpyDeviceToHost);
	cudaMemcpy(m_pressure, m_dSortedPress, sizeof(float)*m_numParticles,cudaMemcpyDeviceToHost);

}

} /* CFD */ 
