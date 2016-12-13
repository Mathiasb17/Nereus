#ifndef _PARTICLES_KERNEL_IMPL_CUH
#define _PARTICLES_KERNEL_IMPL_CUH

#include "sph_kernel.cuh"

#include "kernels_impl.cuh"

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <helper_cuda.h>
#include <helper_cuda_gl.h>

#include <helper_functions.h>

#include <stdio.h>
#include <math.h>

#include <helper_math.h>
#include <math_constants.h>


#if USE_TEX
// textures for particle position and velocity
texture<float4, 1, cudaReadModeElementType> oldPosTex;
texture<float4, 1, cudaReadModeElementType> oldVelTex;
texture<float, 1, cudaReadModeElementType> oldDensTex;
texture<float, 1, cudaReadModeElementType> oldPresTex;
texture<float4, 1, cudaReadModeElementType> oldForcesTex;
texture<float4, 1, cudaReadModeElementType> oldColTex;

texture<unsigned int, 1, cudaReadModeElementType> gridParticleHashTex;
texture<unsigned int, 1, cudaReadModeElementType> cellStartTex;
texture<unsigned int, 1, cudaReadModeElementType> cellEndTex;
#endif

__constant__ SphSimParams sph_params;


struct integrate_functor
{
    float deltaTime;

    __host__ __device__
    integrate_functor(float delta_time) : deltaTime(delta_time) {}

    template <typename Tuple>
    __device__
    void operator()(Tuple t)
    {
        volatile float4 posData = thrust::get<0>(t);
        volatile float4 velData = thrust::get<1>(t);
        float3 pos = make_float3(posData.x, posData.y, posData.z);
        float3 vel = make_float3(velData.x, velData.y, velData.z);

        //vel += sph_params.gravity * deltaTime;
        //vel *= 0.45f;

        // new position = old position + velocity * deltaTime
        //pos += vel * deltaTime;
		
		pos = pos + make_float3(0.0001f, 0, 0);

		//printf("params %f \n", sph_params.viscosity);

        // set this to zero to disable collisions with cube sides
#if 1

        if (pos.x > 1.0f - sph_params.particleRadius)
        {
            pos.x = 1.0f - sph_params.particleRadius;
            vel.x *= 0.45f;
        }

        if (pos.x < -1.0f + sph_params.particleRadius)
        {
            pos.x = -1.0f + sph_params.particleRadius;
            vel.x *= 0.45f;
        }

        if (pos.y > 1.0f - sph_params.particleRadius)
        {
            pos.y = 1.0f - sph_params.particleRadius;
            vel.y *= 0.45f;
        }

        if (pos.z > 1.0f - sph_params.particleRadius)
        {
            pos.z = 1.0f - sph_params.particleRadius;
            vel.z *= 0.45f;
        }

        if (pos.z < -1.0f + sph_params.particleRadius)
        {
            pos.z = -1.0f + sph_params.particleRadius;
            vel.z *= 0.45f;
        }

#endif

        if (pos.y < -1.0f + sph_params.particleRadius)
        {
            pos.y = -1.0f + sph_params.particleRadius;
            vel.y *= 0.45f;
        }

        // store new position and velocity
        thrust::get<0>(t) = make_float4(pos, posData.w);
        thrust::get<1>(t) = make_float4(vel, velData.w);
    }
};


__device__ int3 calcGridPos(float3 p)
{
    int3 gridPos;
    gridPos.x = floor((p.x - sph_params.worldOrigin.x) / sph_params.cellSize.x);
    gridPos.y = floor((p.y - sph_params.worldOrigin.y) / sph_params.cellSize.y);
    gridPos.z = floor((p.z - sph_params.worldOrigin.z) / sph_params.cellSize.z);
    return gridPos;
}

__device__ unsigned int calcGridHash(int3 gridPos)
{
    gridPos.x = gridPos.x & (sph_params.gridSize.x-1);  // wrap grid, assumes size is power of 2
    gridPos.y = gridPos.y & (sph_params.gridSize.y-1);
    gridPos.z = gridPos.z & (sph_params.gridSize.z-1);
    return __umul24(__umul24(gridPos.z, sph_params.gridSize.y), sph_params.gridSize.x) + __umul24(gridPos.y, sph_params.gridSize.x) + gridPos.x;
}

__global__ void calcHashD(unsigned int   *gridParticleHash,  // output
               unsigned int   *gridParticleIndex, // output
               float4 *pos,               // input: positions
               unsigned int    numParticles)
{
    unsigned int index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;

    if (index >= numParticles) return;

    volatile float4 p = pos[index];

    // get address in grid
    int3 gridPos = calcGridPos(make_float3(p.x, p.y, p.z));
    unsigned int hash = calcGridHash(gridPos);

    // store grid hash and particle index
    gridParticleHash[index] = hash;
    gridParticleIndex[index] = index;
}

__global__ void reorderDataAndFindCellStartD(unsigned int   *cellStart,        // output: cell start index
                                  unsigned int   *cellEnd,          // output: cell end index
                                  float4 *sortedPos,        // output: sorted positions
                                  float4 *sortedVel,        // output: sorted velocities
                                  float *sortedDens,       // output: sorted densities
                                  float *sortedPres,       // output: sorted pressures
                                  float4 *sortedForces,     // output: sorted forces
                                  float4 *sortedCol,        // output: sorted colors
                                  unsigned int   *gridParticleHash, // input: sorted grid hashes
                                  unsigned int   *gridParticleIndex,// input: sorted particle indices
                                  float4 *oldPos,           // input: sorted position array
                                  float4 *oldVel,           // input: sorted velocity array
                                  float *oldDens,          // input: sorted density  array
                                  float *oldPres,          // input: sorted pressure array
                                  float4 *oldForces,        // input: sorted forces   array
                                  float4 *oldCol,           // input: sorted color    array
                                  unsigned int    numParticles)
{
    extern __shared__ unsigned int sharedHash[];    // blockSize + 1 elements
    unsigned int index = __umul24(blockIdx.x,blockDim.x) + threadIdx.x;

    unsigned int hash;

	// handle case when no. of particles not multiple of block size
	if (index < numParticles)
	{
		hash = gridParticleHash[index];

		// Load hash data into shared memory so that we can look
		// at neighboring particle's hash value without loading
		// two hash values per thread
		sharedHash[threadIdx.x+1] = hash;

		if (index > 0 && threadIdx.x == 0)
		{
			// first thread in block must load neighbor particle hash
			sharedHash[0] = gridParticleHash[index-1];
		}
	}

	__syncthreads();

	if (index < numParticles)
	{
        //// If this particle has a different cell index to the previous
        //// particle then it must be the first particle in the cell,
        //// so store the index of this particle in the cell.
        //// As it isn't the first particle, it must also be the cell end of
        //// the previous particle's cell
		
		//printf("hash %d\n",hash );

		if (index == 0 || hash != sharedHash[threadIdx.x])
		{
			cellStart[hash] = index;

			if (index > 0)
				cellEnd[sharedHash[threadIdx.x]] = index;
		}

		if (index == numParticles - 1)
		{
			cellEnd[hash] = index + 1;
		}

		// Now use the sorted index to reorder the pos and vel data
		unsigned int sortedIndex = gridParticleIndex[index];
		float4 pos = FETCH(oldPos, sortedIndex);       // macro does either global read or texture fetch
		float4 vel = FETCH(oldVel, sortedIndex);       // see particles_kernel.cuh
		float dens = FETCH(oldDens, sortedIndex);
		float pres = FETCH(oldPres, sortedIndex);
		float4 forces = FETCH(oldForces, sortedIndex);
		float4 col = FETCH(oldCol, sortedIndex);

		sortedPos[index] = pos;
		sortedVel[index] = vel;
		sortedDens[index] = dens;
		sortedPres[index] = pres;
		sortedForces[index] = forces;
		sortedCol[index] = col;
	}
}

__device__ float computeCellDensity(int3 gridPos, unsigned int index, float3 pos, float4 *oldPos, unsigned int *cellStart, unsigned int *cellEnd)
{
    unsigned int gridHash = calcGridHash(gridPos);
	unsigned int startIndex = FETCH(cellStart, gridHash);

	float dens = 0.f;
	float3 pos1 = make_float3(pos.x, pos.y, pos.z);

	if (startIndex != 0xffffffff)
	{ 
		unsigned int endIndex = FETCH(cellEnd, gridHash);

        for (unsigned int j=startIndex; j<endIndex; j++)
		{
			if(j != index)
			{
				float3 pos2 = make_float3(FETCH(oldPos, j));
				dens += sph_params.particleMass * Wdefault(pos2-pos1, sph_params.interactionRadius);
			}
		}
	}

	return dens;
}

__global__
void computeDensityPressure(float *newDens,               // output: new velocity
			  float* newPres,
              float4 *oldPos,               // input: sorted positions
              float4 *oldVel,               // input: sorted velocities
              float *oldDens,               // input: sorted velocities
              float *oldPres,               // input: sorted velocities
              float4 *oldForces,            // input: sorted velocities
              float4 *oldCol,               // input: sorted velocities
              unsigned int   *gridParticleIndex,    // input: sorted particle indices
              unsigned int   *cellStart,
              unsigned int   *cellEnd,
              unsigned int    numParticles)
{
    unsigned int index = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;

    if (index >= numParticles) return;

    // read particle data from sorted arrays
    float3 pos = make_float3(FETCH(oldPos, index));
    float3 vel = make_float3(FETCH(oldVel, index));

    // get address in grid
    int3 gridPos = calcGridPos(pos);

    // examine neighbouring cells
    float dens = 0.f;

	//compute pressure
    for (int z=-1; z<=1; z++)
    {
        for (int y=-1; y<=1; y++)
        {
            for (int x=-1; x<=1; x++)
            {
                int3 neighbourPos = gridPos + make_int3(x, y, z);
                dens += computeCellDensity(neighbourPos, index, pos, oldPos, cellStart, cellEnd);
            }
        }
    }

	//printf("density %5f \n", dens );

	//compute Pressure
	float pressure = sph_params.gasStiffness * (powf(dens/sph_params.restDensity, 7) -1);

    // write new velocity back to original unsorted location
    unsigned int originalIndex = gridParticleIndex[index];
    newDens[originalIndex] = dens;
    newPres[originalIndex] = pressure;
}

#endif /* ifndef _PARTICLES_KERNEL_IMPL_CUH */
