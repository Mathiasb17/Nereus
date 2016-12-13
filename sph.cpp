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

typedef thrust::host_vector<int>::iterator   IntIterator;
typedef thrust::host_vector<float>::iterator FloatIterator;
typedef thrust::host_vector<char>::iterator  CharIterator;
typedef thrust::host_vector<double>::iterator  DoubleIterator;
typedef thrust::host_vector<glm::vec4>::iterator  Vec3Iterator;
typedef thrust::host_vector<unsigned int>::iterator  UIntIterator;

typedef thrust::tuple<Vec3Iterator, Vec3Iterator, FloatIterator, FloatIterator, Vec3Iterator, Vec3Iterator> IteratorTuple;

typedef thrust::zip_iterator<IteratorTuple> ZipIterator;


static bool compareVel(glm::vec3 v1, glm::vec3 v2)
{
	return glm::length(v1) < glm::length(v2);
}

SPH::SPH ():
	m_gridSortBits(18)
{
	m_params.gasStiffness = 300.f;
	m_params.restDensity = 998.29;
	m_params.particleRadius = 0.02;
	m_params.timestep = 1E-3f;
	m_params.viscosity = 0.013f;
	m_params.surfaceTension = 0.01f;
	m_params.interactionRadius = 0.0457f;

	m_params.worldOrigin = make_float3(-2,-2,-2);
	m_params.gridSize = make_uint3(100,100,100);
	m_params.cellSize = make_float3(m_params.interactionRadius, m_params.interactionRadius, m_params.interactionRadius);
	m_params.numCells = m_params.gridSize.x * m_params.gridSize.y * m_params.gridSize.z;

	m_params.particleMass = powf(m_params.interactionRadius, 3)*m_params.restDensity;

	_intialize();

	m_numParticles = 0;
}

SPH::~SPH ()
{

}

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

	setParameters(&m_params);
}

void SPH::_finalize()
{

}

void SPH::update()
{
	cudaMemcpy(m_dpos, m_pos, sizeof(float)*4*m_numParticles,cudaMemcpyHostToDevice);
	setParameters(&m_params);

	integrateSystem( m_dpos, m_dvel, m_params.timestep, m_numParticles);

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

	cudaMemcpy(m_pos, m_dpos, sizeof(float)*4*m_numParticles,cudaMemcpyDeviceToHost);
}

void SPH::initNeighbors()
{
}

void SPH::ComputeNeighbors()
{
}

void SPH::ComputeDensitiesAndPressure()
{
}

void SPH::ComputeInternalForces()
{
}

void SPH::ComputeExternalForces()
{

}

void SPH::CollisionDetectionsAndResponses()
{

}

void SPH::ComputeImplicitEulerScheme()
{

}

void SPH::addNewParticle(glm::vec4 p)
{
	m_pos[m_numParticles*4+0] =  p.x;
	m_pos[m_numParticles*4+1] =  p.y;
	m_pos[m_numParticles*4+2] =  p.z;
	m_pos[m_numParticles*4+3] =  p.w;

	m_density[m_numParticles] = 0.f;

	m_pressure[m_numParticles] = 0.f;

	m_vel[m_numParticles*4+0] =  0.f;
	m_vel[m_numParticles*4+1] =  0.f;
	m_vel[m_numParticles*4+2] =  0.f;
	m_vel[m_numParticles*4+3] =  0.f;

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

void SPH::generateParticleCube(glm::vec4 center, glm::vec4 size)
{
	for(float x = center.x-size.x/2.f; x <= center.x+size.x/2.f; x += m_params.particleRadius*2 )
	{
		for(float y = center.y-size.y/2.f; y <= center.y+size.y/2.f; y += m_params.particleRadius*2 )
		{
			for(float z = center.z-size.z/2.f; z <= center.z+size.z/2.f; z += m_params.particleRadius*2 )
			{
				addNewParticle(glm::vec4(x,y,z,1.f));
			}
		}
	}
	//std::cout << "Il y a eu " << m_pos.size() << " particules generees." << std::endl;
}

} /* CFD */ 
