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

	//OVERIDING THE FOLLOWING AS FOR NOW FIXME
	/********************
	*  SPH PARAMETERS  *
	********************/
	m_params.gasStiffness = 100.f; //useless in iisph
	m_params.restDensity = 1000.f;
	m_params.particleRadius = 0.02;
	m_params.timestep = 1E-3f;
	m_params.viscosity = 0.01f;
	m_params.surfaceTension = 0.01f;

	m_params.gravity.x = 0.f;
	m_params.gravity.y = 0.f;
	m_params.gravity.y = -9.81f;
	m_params.gravity.z = 0.f;

	m_params.interactionRadius = 0.0527f;//better !
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

	std::cout << "kpoly = " << std::fixed << m_params.kpoly << std::endl;
	std::cout << "kpoly_grad = " <<  std::fixed << m_params.kpoly_grad << std::endl;

	m_params.kvisc_grad = 15.f / (2*M_PI*powf(m_params.interactionRadius, 3.f));
	m_params.kvisc_denum = 2.f*powf(m_params.interactionRadius, 3.f);

	m_params.ksurf1 = 32.f/(M_PI * powf(m_params.interactionRadius,9));
	m_params.ksurf2 = powf(m_params.interactionRadius,6)/64.f;

	m_params.bpol = 0.007f / (powf(m_params.interactionRadius, 3.25));
	
	_intialize();
	m_numParticles = 0;

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
	//exit(0);
}

} /* CFD */ 
