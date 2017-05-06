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

PCISPH::PCISPH():
	SPH()
{

}

PCISPH::~PCISPH()
{

}

void PCISPH::_intialize()
{
	SPH::_intialize();

	unsigned int memSize = sizeof(float) * 4 * MAX_PARTICLE_NUMBER;
	unsigned int memSizeFloat = sizeof(float) * MAX_PARTICLE_NUMBER;
	unsigned int memSizeUint = sizeof(unsigned int) * m_params.numCells;

	/*******************
	 *  HOST MEM INIT  *
	 *******************/
	cudaMallocHost((void**)&m_velstar, memSize);
	cudaMallocHost((void**)&m_posstar, memSize);
	cudaMallocHost((void**)&m_densstar, memSizeFloat);
	cudaMallocHost((void**)&m_denserror, memSizeFloat);

	/******************
	 *  GPY MEM INIT  *
	 ******************/
	allocateArray((void **)&m_dvelstar, memSize);
	allocateArray((void **)&m_dposstar, memSize);
	allocateArray((void **)&m_ddensstar, memSizeFloat);
	allocateArray((void **)&m_ddenserror, memSizeFloat);

	allocateArray((void **)&m_dSortedVelstar, memSize);
	allocateArray((void **)&m_dSortedPosstar, memSize);
	allocateArray((void **)&m_dSortedDensstar, memSizeFloat);
	allocateArray((void **)&m_dSortedDenserror, memSizeFloat);

	/************
	*  memset  *
	************/
	cudaMemset(m_dvelstar, 0, memSize);
	cudaMemset(m_dposstar, 0, memSize);
	cudaMemset(m_ddensstar, 0, memSizeFloat);
	cudaMemset(m_ddenserror, 0, memSizeFloat);

	cudaMemset(m_dSortedPosstar, 0, memSize);
	cudaMemset(m_dSortedVelstar, 0, memSize);
	cudaMemset(m_dSortedDensstar, 0, memSizeFloat);
	cudaMemset(m_dSortedDenserror, 0, memSizeFloat);
}

void PCISPH::_finalize()
{
	SPH::_finalize();
}

void PCISPH::addNewParticle(glm::vec4 p, glm::vec4 v)
{
	SPH::addNewParticle(p,v);
}

void PCISPH::update()
{
   /* cudaMemcpy(m_dpos, m_pos, sizeof(float)*4*m_numParticles,cudaMemcpyHostToDevice);*/
	//cudaMemcpy(m_dvel, m_vel, sizeof(float)*4*m_numParticles,cudaMemcpyHostToDevice);

	//std::cout << " update pcisph !" << std::endl;

	//setParameters(&m_params);

	//calcHash( m_dGridParticleHash, m_dGridParticleIndex, m_dpos, m_numParticles);

	//sortParticles(m_dGridParticleHash, m_dGridParticleIndex, m_numParticles);

	//reorderDataAndFindCellStart(
		//m_dCellStart,
		//m_dCellEnd,
		//m_dSortedPos,
		//m_dSortedVel,
		//m_dSortedDens,
		//m_dSortedPress,
		//m_dSortedForces,
		//m_dSortedCol,
		//m_dGridParticleHash,
		//m_dGridParticleIndex,
		//m_dpos,
		//m_dvel,
		//m_ddensity,
		//m_dpressure,
		//m_dforces,
		//m_dcolors,
		//m_numParticles,
		//m_params.numCells);

	//computePciDensityPressure(
			//&m_params,
			//m_dSortedPos,
			//m_dSortedVel,
			//m_dSortedDens,
			//m_dSortedPress,
			//m_dSortedForces,
			//m_dSortedCol,
			//m_dSortedPosstar,
			//m_dSortedVelstar,
			//m_dSortedDensstar,
			//m_dSortedDenserror,
			//m_dGridParticleIndex,
			//m_dCellStart,
			//m_dCellEnd,
			//m_numParticles,
			//m_params.numCells);

	//cudaMemcpy(m_pos, m_dSortedPos, sizeof(float)*4*m_numParticles,cudaMemcpyDeviceToHost);
	/*cudaMemcpy(m_vel, m_dSortedVel, sizeof(float)*4*m_numParticles,cudaMemcpyDeviceToHost);*/
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

	computePciDensityPressure(
			&m_params,
			m_dSortedPos,
			m_dSortedVel,
			m_dSortedDens,
			m_dSortedPress,
			m_dSortedForces,
			m_dSortedCol,
			m_dSortedPosstar,
			m_dSortedVelstar,
			m_dSortedDensstar,
			m_dSortedDenserror,
			m_dGridParticleIndex,
			m_dCellStart,
			m_dCellEnd,
			m_numParticles,
			m_params.numCells);

	integrateSystem( m_dSortedPos, m_dSortedVel, m_dSortedForces, m_params.timestep, m_numParticles);

	cudaMemcpy(m_pos, m_dSortedPos, sizeof(float)*4*m_numParticles,cudaMemcpyDeviceToHost);
	cudaMemcpy(m_vel, m_dSortedVel, sizeof(float)*4*m_numParticles,cudaMemcpyHostToDevice);

}
	
} /* CFD */ 
