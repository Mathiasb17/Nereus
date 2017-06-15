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
#include <float.h>

#if USE_TEX
// textures for particle position and velocity
texture<float4, 1, cudaReadModeElementType> oldPosTex;
texture<float4, 1, cudaReadModeElementType> oldVelTex;
texture<float, 1, cudaReadModeElementType> oldDensTex;
texture<float, 1, cudaReadModeElementType> oldPresTex;
texture<float4, 1, cudaReadModeElementType> oldForcesTex;
texture<float4, 1, cudaReadModeElementType> oldColTex;

//texture for iisph
texture<float, 1, cudaReadModeElementType> oldDensAdvTex;
texture<float, 1, cudaReadModeElementType> oldDensCorrTex;
texture<float, 1, cudaReadModeElementType> oldP_lTex;
texture<float, 1, cudaReadModeElementType> oldPreviousPTex;
texture<float, 1, cudaReadModeElementType> oldAiiTex;

texture<float4, 1, cudaReadModeElementType> oldVelAdvTex;
texture<float4, 1, cudaReadModeElementType> oldForcesAdvTex;
texture<float4, 1, cudaReadModeElementType> oldForcesPTex;
texture<float4, 1, cudaReadModeElementType> oldDiiFluidTex;
texture<float4, 1, cudaReadModeElementType> oldDiiBoundaryTex;
texture<float4, 1, cudaReadModeElementType> oldSumDijTex;
texture<float4, 1, cudaReadModeElementType> oldNormalTex;

//grid textures
texture<unsigned int, 1, cudaReadModeElementType> gridParticleHashTex;
texture<unsigned int, 1, cudaReadModeElementType> cellStartTex;
texture<unsigned int, 1, cudaReadModeElementType> cellEndTex;

//boundaries
texture<float4, 1, cudaReadModeElementType> oldBoundaryPosTex;
texture<float, 1, cudaReadModeElementType> oldBoundaryVbiTex;
texture<unsigned int, 1, cudaReadModeElementType> gridBoundaryHashTex;
texture<unsigned int, 1, cudaReadModeElementType> cellBoundaryStartTex;
texture<unsigned int, 1, cudaReadModeElementType> cellBoundaryEndTex;

#endif

__constant__ SphSimParams sph_params;

//====================================================================================================  
//====================================================================================================  
//====================================================================================================  
struct integrate_functor
{
    float deltaTime;

    __host__ __device__ integrate_functor(float delta_time) : deltaTime(delta_time) {}

    template <typename Tuple>
    __device__ void operator()(Tuple t)
    {
		float dt = sph_params.timestep;
		float m1 = sph_params.particleMass;

        volatile float4 posData = thrust::get<0>(t);
        volatile float4 velData = thrust::get<1>(t);
        volatile float4 forData = thrust::get<2>(t);

        float3 pos = make_float3(posData.x, posData.y, posData.z);
        float3 vel = make_float3(velData.x, velData.y, velData.z);
        float3 frc = make_float3(forData.x, forData.y, forData.z);

		float3 accel = dt*frc/m1;

		vel = vel+accel;
		pos = pos + dt*vel;

        // store new position and velocity
        thrust::get<0>(t) = make_float4(pos, posData.w);
        thrust::get<1>(t) = make_float4(vel, velData.w);
    }
};

//====================================================================================================  
//====================================================================================================  
//====================================================================================================  
__device__ int3 calcGridPos(float3 p)
{
    int3 gridPos;
    gridPos.x = floor((p.x - sph_params.worldOrigin.x) / sph_params.cellSize.x);
    gridPos.y = floor((p.y - sph_params.worldOrigin.y) / sph_params.cellSize.y);
    gridPos.z = floor((p.z - sph_params.worldOrigin.z) / sph_params.cellSize.z);

    return gridPos;
}

//====================================================================================================  
//====================================================================================================  
//====================================================================================================  
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
    unsigned int index = blockIdx.x*blockDim.x + threadIdx.x;

    if (index >= numParticles) return;

    volatile float4 p = pos[index];

    // get address in grid
    int3 gridPos = calcGridPos(make_float3(p.x, p.y, p.z));
    unsigned int hash = calcGridHash(gridPos);

    // store grid hash and particle index
    gridParticleHash[index] = hash;
    gridParticleIndex[index] = index;
}

//====================================================================================================  
//====================================================================================================  
//====================================================================================================  
__global__ void reorderDataAndFindCellStartDBoundary(unsigned int *cellBoundaryStart,
												unsigned int * cellBoundaryEnd,
												float4* sortedBoundaryPos,
												float* sortedBoundaryVbi,
												unsigned int *gridBoundaryHash,
												unsigned int *gridBoundaryIndex,
												float4* oldBoundaryPos,
												float*  oldBoundaryVbi,
												unsigned int numBoundaries)
{
	extern __shared__ unsigned int sharedHash[];
	unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

	unsigned int hash;

	if (index < numBoundaries) 
	{
		hash = gridBoundaryHash[index];

		sharedHash[threadIdx.x+1] = hash;

		if (index > 0 && threadIdx.x == 0)
		{
			// first thread in block must load neighbor particle hash
			sharedHash[0] = gridBoundaryHash[index-1];
		}
	}

	__syncthreads();


	if (index < numBoundaries)
	{
		if (index == 0 || hash != sharedHash[threadIdx.x])
		{
			cellBoundaryStart[hash] = index;

			if (index > 0)
				cellBoundaryEnd[sharedHash[threadIdx.x]] = index;
		}

		if (index == numBoundaries - 1)
		{
			cellBoundaryEnd[hash] = index + 1;
		}

		// Now use the sorted index to reorder the pos and vel data
		unsigned int sortedIndex = gridBoundaryIndex[index];

		float4 pos = FETCH(oldBoundaryPos, sortedIndex);       // macro does either global read or texture fetch
		float vbi = FETCH(oldBoundaryVbi, sortedIndex);       // see particles_kernel.cuh
		
		oldBoundaryPos[index] = pos;
		oldBoundaryVbi[index] = vbi;
	}
}

//====================================================================================================  
//====================================================================================================  
//====================================================================================================  
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
    unsigned int index = blockIdx.x*blockDim.x + threadIdx.x;

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
		float pressure = FETCH(oldPres, sortedIndex);       // see particles_kernel.cuh
		sortedVel[index] = vel;
		sortedPos[index] = pos;
		sortedPres[index] = pressure;
	}
}

/**********************************************************************
*                      COMPUTE DENSITY PRESSURE                      *
**********************************************************************/

//====================================================================================================  
//====================================================================================================  
//====================================================================================================  
__device__ float computeCellDensity(int *nb, int3 gridPos, unsigned int index, float3 pos, float4 *oldPos, unsigned int *cellStart, unsigned int *cellEnd,
		float ir, float kp, float rd, float pm)
{
    const unsigned int gridHash = calcGridHash(gridPos);
	const unsigned int startIndex = FETCH(cellStart, gridHash);

	float dens = 0.f;
	const float3 pos1 = make_float3(pos.x, pos.y, pos.z);

	if (startIndex != 0xffffffff)
	{ 
		const unsigned int endIndex = FETCH(cellEnd, gridHash);

        for (unsigned int j=startIndex; j<endIndex; j++)
		{
			if(j != index)
			{
				const float3 pos2 = make_float3(FETCH(oldPos, j));
				const float3 p1p2 = pos1 - pos2;
				if(length(p1p2) < ir)
				{
					dens += pm * Wdefault(p1p2, ir, kp);
				}
			}
		}
	}
	return dens;
}

//====================================================================================================  
//====================================================================================================  
//====================================================================================================  
__device__ float computeBoundaryCellDensity(int3 gridPos, float3 pos, unsigned int* gridBoundaryIndex, float4* oldBoundaryPos, float* oldBoundaryVbi, unsigned int* cellBoundaryStart, unsigned int* cellBoundaryEnd,
		float ir, float kp, float rd, float pm)
{
	const unsigned int gridHash = calcGridHash(gridPos);
	const unsigned int startIndex = FETCH(cellBoundaryStart, gridHash);

	float dens = 0.f;
	const float3 pos1 = pos;

	if (startIndex != 0xffffffff) 
	{
		const unsigned int endIndex = FETCH(cellBoundaryEnd, gridHash);

		for (unsigned  int j = startIndex; j < endIndex; ++j)
		{
			const unsigned int originalIndex = gridBoundaryIndex[j];

			const float3 pos2 = make_float3(FETCH(oldBoundaryPos, originalIndex));
			const float  vbi  = FETCH(oldBoundaryVbi, originalIndex);
			const float3 p1p2 = pos1 - pos2;

			if (length(p1p2) < ir) 
			{
				dens += (rd* vbi) * Wdefault(p1p2, ir, kp);
			}
		}
	}
	return dens;
}

//====================================================================================================  
//====================================================================================================  
//====================================================================================================  
__global__ void computeDensityPressure(
              float4 *oldPos,               // input: sorted positions
              float4 *oldVel,               // input: sorted velocities
              float *oldDens,               // input: sorted velocities
              float *oldPres,               // input: sorted velocities
              float4 *oldForces,            // input: sorted velocities
              float4 *oldCol,               // input: sorted velocities
			  float4 *oldBoundaryPos,
			  float  *oldBoundaryVbi,
              unsigned int   *gridParticleIndex,    // input: sorted particle indices
              unsigned int   *cellStart,
              unsigned int   *cellEnd,
			  unsigned int   *gridBoundaryIndex,
			  unsigned int   *cellBoundaryStart,
			  unsigned int   *cellBoundaryEnd,
              unsigned int    numParticles)
{
    const unsigned int index = blockIdx.x*blockDim.x + threadIdx.x;

    if (index >= numParticles) return;

	const unsigned int originalIndex = gridParticleIndex[index];

    // read particle data from sorted arrays
    const float3 pos = make_float3(FETCH(oldPos, originalIndex));
    const float3 vel = make_float3(FETCH(oldVel, originalIndex));

    // get address in grid
    const int3 gridPos = calcGridPos(pos);

    // examine neighbouring cells
    float dens = 0.f;
	int nbVois = 0;

	//const memory access
	const float ir = sph_params.interactionRadius;
	const float kp = sph_params.kpoly;
	const float rd = sph_params.restDensity;
	const float pm = sph_params.particleMass;

	//compute pressure
    for (int z=-1; z<=1; z++)
    {
        for (int y=-1; y<=1; y++)
        {
            for (int x=-1; x<=1; x++)
            {
                const int3 neighbourPos = gridPos + make_int3(x, y, z);
				dens += computeCellDensity(&nbVois, neighbourPos, originalIndex, pos, oldPos, cellStart, cellEnd, ir, kp, rd, pm);
				dens += computeBoundaryCellDensity(neighbourPos, pos, gridBoundaryIndex, oldBoundaryPos, oldBoundaryVbi, cellBoundaryStart, cellBoundaryEnd, ir, kp, rd, pm);
            }
        }
    } 
	
	//compute Pressure
	const float pressure = sph_params.gasStiffness * (powf(dens/sph_params.restDensity, 7) -1);

    // write new velocity back to original unsorted location
    oldDens[originalIndex] = dens;
    oldPres[originalIndex] = pressure;
}

/**********************************************************************
*                           COMPUTE FORCES                           *
**********************************************************************/

//====================================================================================================  
//====================================================================================================  
//====================================================================================================  
__device__ void computeCellForces(
		float3     *fpres,
		float3     *fvisc,
		float3     *fsurf,
		float3     *fbound,
		int3       gridPos,
		unsigned   int index,
		float3     pos,
		float3     vel,
		float      dens,
		float      pres,
		float4*    oldPos,
		float      *oldDens,
		float*     oldPres,
		float4*    oldVel,
		unsigned int* gridBoundaryIndex,
		float4*    oldBoundaryPos,
		float*     oldBoundaryVbi,
		unsigned int *cellStart,
		unsigned int *cellEnd,
		unsigned int *cellBoundaryStart,
		unsigned int *cellBoundaryEnd)
{
    const unsigned int gridHash = calcGridHash(gridPos);
	unsigned int startIndex = FETCH(cellStart, gridHash);

	const float3 pos1 = make_float3(pos.x, pos.y, pos.z);
	const float3 vel1 = make_float3(vel.x, vel.y, vel.z);

	float3 forces = make_float3(0.f, 0.f, 0.f);
	float3 forces_pres = make_float3(0.f, 0.f, 0.f);
	float3 forces_visc = make_float3(0.f, 0.f, 0.f);

	const float m2 = sph_params.particleMass;
	const float ir = sph_params.interactionRadius;
	const float kp = sph_params.kpoly;
	const float kpg= sph_params.kpoly_grad;

	const float kprg = sph_params.kpress_grad;
	const float kvg  = sph_params.kvisc_grad;
	const float kvd  = sph_params.kvisc_denum;

	/*const float ksurf1 = sph_params.ksurf1;*/
	/*const float ksurf2 = sph_params.ksurf2;*/

	if (startIndex != 0xffffffff)
	{ 
		const unsigned int endIndex = FETCH(cellEnd, gridHash);

        for (unsigned int j=startIndex; j<endIndex; j++)
		{
			if(j != index)
			{
				const float3 pos2 = make_float3(FETCH(oldPos, j));
				const float dens2 = FETCH(oldDens, j);
				const float pres2 = FETCH(oldPres, j);
				const float3 vel2 = make_float3(FETCH(oldVel, j));

				const float3 p1p2 = pos1-pos2;
				const float3 v1v2 = vel1-vel2;

				const float d1sq = dens*dens;
				const float d2sq = dens2*dens2;

				const float kdefault = Wdefault(p1p2, ir, kp);
				const float3 kdefault_grad = Wdefault_grad(p1p2, ir, kpg);
				const float3 kpressure_grad = Wpressure_grad(p1p2, ir, kprg);
				const float3 kvisco_grad = Wviscosity_grad(p1p2, ir, kvg, kvd);

				if (length(p1p2) < ir)
				{
					*fpres = *fpres + (m2 * ( pres/d1sq + pres2/d2sq ) *kpressure_grad);

					const float a = dot(p1p2, kvisco_grad);
					const float b = dot(p1p2,p1p2) + 0.01f*ir*ir;
					*fvisc = *fvisc + (m2/dens2  * v1v2 * (a/b));

					*fsurf = *fsurf + m2 * p1p2 * Wdefault(p1p2, ir, sph_params.kpoly) ;
					/*float gamma = sph_params.surfaceTension;*/
					/**fsurf = *fsurf + (-gamma * m2*m2 * Cakinci(p1p2, ir, ksurf1, ksurf2)*(p1p2/length(p1p2)));*/
				}
			}
		}
	}

	//start again with boundaries
	startIndex = FETCH(cellBoundaryStart, gridHash);
	float3 forces_boundaries = make_float3(0.f, 0.f, 0.f);

	if (startIndex != 0xffffffff)
	{
		const float beta = sph_params.beta;
		const unsigned int endIndex = FETCH(cellBoundaryEnd, gridHash);
		const float rd = sph_params.restDensity;

		//loop over rigid boundary particles
        for (unsigned int j=startIndex; j<endIndex; j++)
		{
			const unsigned int originalIndex = gridBoundaryIndex[j];

			const float vbi  = FETCH(oldBoundaryVbi, originalIndex);
			const float3 vpos= make_float3(FETCH(oldBoundaryPos, originalIndex));

			const float psi = (rd*vbi);
			const float3 p1p2 = pos1 - vpos;

			const float kdefault = Wdefault(p1p2, ir, sph_params.kpoly);

			float3 contrib = (beta * psi * p1p2 * kdefault);
			*fbound = *fbound + contrib;
		}
	}
}

//====================================================================================================  
//====================================================================================================  
//====================================================================================================  
__global__
void computeForces(
              float4       * oldPos,               
              float4       * oldVel,              
              float        * oldDens,            
              float        * oldPres,           
              float4       * oldForces,        
              float4       * oldCol,          
			  unsigned int * gridBoundaryIndex,
			  float4       * oldBoundaryPos,
			  float        * oldBoundaryVbi,
              unsigned int * gridParticleIndex, 
              unsigned int * cellStart,
              unsigned int * cellEnd,
			  unsigned int * cellBoundaryStart,
			  unsigned int * cellBoundaryEnd,
              unsigned int    numParticles)
{
    const unsigned int index = blockIdx.x*blockDim.x + threadIdx.x;

    if (index >= numParticles) return;

	const unsigned int originalIndex = gridParticleIndex[index];

    // read particle data from sorted arrays
    const float3 pos = make_float3(FETCH(oldPos, originalIndex));
    const float3 vel = make_float3(FETCH(oldVel, originalIndex));
	const float dens = FETCH(oldDens, originalIndex);
	const float pres = FETCH(oldPres, originalIndex);

	const float m1 = sph_params.particleMass;

	//grid address
    const int3 gridPos = calcGridPos(pos);

	//accumulators
	float3 fpres = make_float3(0.f, 0.f, 0.f);
	float3 fvisc = make_float3(0.f, 0.f, 0.f);
	float3 fsurf = make_float3(0.f, 0.f, 0.f);
	float3 fbound= make_float3(0.f, 0.f, 0.f);

	for (int z=-1; z<=1; z++)
	{
		for (int y=-1; y<=1; y++)
		{
			for (int x=-1; x<=1; x++)
			{
				const int3 neighbourPos = gridPos + make_int3(x, y, z);
				//a optimiser !!!
				computeCellForces(&fpres, &fvisc, &fsurf, &fbound, neighbourPos, originalIndex, pos, vel, dens, pres, oldPos, oldDens, oldPres, oldVel, gridBoundaryIndex, oldBoundaryPos, oldBoundaryVbi, cellStart, cellEnd, cellBoundaryStart, cellBoundaryEnd);
			}
		}
	}

	//finishing gradient and laplacian computations
	fpres = dens * fpres;
	fvisc = 2.f * fvisc;
	fsurf = -(sph_params.surfaceTension/m1) * fsurf;

	//computing forces
	fpres = -(m1 / dens) * fpres;
	fvisc = (m1*sph_params.viscosity) * fvisc;

	float3 f = fpres + fvisc + (sph_params.gravity*m1) + fsurf + fbound;
	float4 res = make_float4(f.x, f.y, f.z, 0);

	oldForces[originalIndex] = res;
}

//====================================================================================================  
//====================================================================================================  
//====================================================================================================  

/***********
*  IISPH  *
***********/
__device__ float3 computeDisplacementFactorCell(float dens, float mj, int3 gridPos, float3 pos, float4* oldPos, unsigned int index, unsigned int* cellStart, unsigned int* cellEnd,
		float ir, float kp, float rd, float pm)
{
	const unsigned int gridHash = calcGridHash(gridPos);
	const unsigned int startIndex = FETCH(cellStart, gridHash);

	float3 res  = make_float3(0.f, 0.f, 0.f);
	const float3 pos1 = make_float3(pos.x, pos.y, pos.z);

	if (startIndex != 0xffffffff)
	{ 
		const unsigned int endIndex = FETCH(cellEnd, gridHash);

        for (unsigned int j=startIndex; j<endIndex; j++)
		{
			if(j != index)
			{
				const float3 pos2 = make_float3(FETCH(oldPos, j));
				const float3 p1p2 = pos1 - pos2;

				if(length(p1p2) < ir)
				{
					res = res + ( - ( pm/(dens*dens) )) * Wdefault_grad(p1p2, ir, sph_params.kpoly_grad);
				}
			}
		}
	}
	return res;
}

//====================================================================================================  
//====================================================================================================  
//====================================================================================================  
__device__ float3 computeDisplacementFactorBoundaryCell(float dens, float mj, int3 gridPos, float3 pos, float4* oldBoundaryPos, float* oldBoundaryVbi, unsigned int* cellBoundaryStart, unsigned int* cellBoundaryEnd,
		float ir, float kpg, float rd, float pm, float dt)
{
	const unsigned int gridHash = calcGridHash(gridPos);
	const unsigned int startIndex = FETCH(cellBoundaryStart, gridHash);

	float3 res  = make_float3(0.f, 0.f, 0.f);
	const float3 pos1 = make_float3(pos.x, pos.y, pos.z);

	if (startIndex != 0xffffffff)
	{ 
		const unsigned int endIndex = FETCH(cellBoundaryEnd, gridHash);

        for (unsigned int j=startIndex; j<endIndex; j++)
		{
			const float3 pos2 = make_float3(FETCH(oldBoundaryPos, j));
			const float  vbi  = FETCH(oldBoundaryVbi, j);
			const float3 p1p2 = pos1 - pos2;

			const float psi  = rd*vbi;
			if(length(p1p2) < ir)
			{
				res = res + (-dt*dt*psi/(dens*dens))*Wdefault_grad(p1p2, ir, kpg);
			}
		}
	}
	return res;
}

//====================================================================================================  
//====================================================================================================  
//====================================================================================================  
__global__ void computeIisphDensity(
	float4                      * oldPos,
	float4                      * oldVel,
	float                       * oldDens,
	float                       * oldPres,
	float4                      * oldForces,
	float4                      * oldCol,
	unsigned int                * cellStart,
	unsigned int                * cellEnd,
	unsigned int                * gridParticleIndex,
	float4                      * oldBoundaryPos,
	float                       * oldBoundaryVbi,
	unsigned int                * cellBoundaryStart,
	unsigned int                * cellBoundaryEnd,
	unsigned int                * gridBoundaryIndex,
	float                       * oldDensAdv,
	float                       * oldDensCorr,
	float                       * oldP_l,
	float                       * oldPreviousP,
	float                       * oldAii,
	float4                      * oldVelAdv,
	float4                      * oldForcesAdv,
	float4                      * oldForcesP,
	float4                      * oldDiiFluid,
	float4                      * oldDiiBoundary,
	float4                      * oldSumDij,
	float4                      * oldNormal,
	unsigned int numParticles,
	unsigned int numBoundaries,
	unsigned int numCells
)
{
	const unsigned int index = blockIdx.x*blockDim.x + threadIdx.x;
    if (index >= numParticles) return;
	const unsigned int originalIndex = gridParticleIndex[index];
	
	//global memory reads
	const float3 pos1 = make_float3(FETCH(oldPos, originalIndex));
	const float3 vel1 = make_float3(FETCH(oldVel, originalIndex));
	
	//const memory reads
	const float kp = sph_params.kpoly;
	const float pm = sph_params.particleMass;
	const float ir = sph_params.interactionRadius;
	const float rd = sph_params.restDensity;

	//grid computations
    const int3 gridPos = calcGridPos(pos1);
	
	/*********************
	*  COMPUTE DENSITY  *
	*********************/
	float dens = 0.f;
	int nb = 0;

	//loop over each neighbor cell
	for (int z=-1; z<=1; z++)
	{
		for (int y=-1; y<=1; y++)
		{
			for (int x=-1; x<=1; x++)
			{
				const int3 neighbourPos = gridPos + make_int3(x, y, z);
				dens += computeCellDensity(&nb, neighbourPos, originalIndex, pos1, oldPos, cellStart, cellEnd, ir, kp, rd, pm);
				dens += computeBoundaryCellDensity(neighbourPos, pos1, gridBoundaryIndex, oldBoundaryPos, oldBoundaryVbi, cellBoundaryStart, cellBoundaryEnd, ir, kp, rd, pm);
			}
		}
	}
	oldDens[originalIndex] = dens;

}

//====================================================================================================  
//====================================================================================================  
//====================================================================================================  
__global__ void computeDisplacementFactor(
	float4                      * oldPos,
	float4                      * oldVel,
	float                       * oldDens,
	float                       * oldPres,
	float4                      * oldForces,
	float4                      * oldCol,
	unsigned int                * cellStart,
	unsigned int                * cellEnd,
	unsigned int                * gridParticleIndex,
	float4                      * oldBoundaryPos,
	float                       * oldBoundaryVbi,
	unsigned int                * cellBoundaryStart,
	unsigned int                * cellBoundaryEnd,
	unsigned int                * gridBoundaryIndex,
	float                       * oldDensAdv,
	float                       * oldDensCorr,
	float                       * oldP_l,
	float                       * oldPreviousP,
	float                       * oldAii,
	float4                      * oldVelAdv,
	float4                      * oldForcesAdv,
	float4                      * oldForcesP,
	float4                      * oldDiiFluid,
	float4                      * oldDiiBoundary,
	float4                      * oldSumDij,
	float4                      * oldNormal,
	unsigned int numParticles,
	unsigned int numBoundaries,
	unsigned int numCells)
{
    const unsigned int index = blockIdx.x*blockDim.x + threadIdx.x;
    if (index >= numParticles) return;
	const unsigned int originalIndex = gridParticleIndex[index];
	
	//global memory reads
	const float3 pos1 = make_float3(FETCH(oldPos, originalIndex));
	const float3 vel1 = make_float3(FETCH(oldVel, originalIndex));
	const float pres = 0.f; //useless, just to reuse computeCellForces below
	const float dens = FETCH(oldDens, originalIndex);

	//const memory reads
	const float kpg= sph_params.kpoly_grad;
	const float pm = sph_params.particleMass;
	const float ir = sph_params.interactionRadius;
	const float rd = sph_params.restDensity;
	const float dt = sph_params.timestep;

	//grid computations
    const int3 gridPos = calcGridPos(pos1);
	
	/***********************
	*  PREDICT ADVECTION  *
	***********************/
	float3 fvisc = make_float3(0.f, 0.f, 0.f);
	float3 fsurf = make_float3(0.f, 0.f, 0.f);
	float3 fgrav = make_float3(0.f, 0.f, 0.f);
	float3 fbound= make_float3(0.f, 0.f, 0.f);
	float3 fpres = make_float3(0.f, 0.f, 0.f); //ignored here, just to reuse computeCellForces

	for (int z=-1; z<=1; z++)
	{
		for (int y=-1; y<=1; y++)
		{
			for (int x=-1; x<=1; x++)
			{
				const int3 neighbourPos = gridPos + make_int3(x, y, z);
				//a optimiser !!!
				computeCellForces(&fpres, &fvisc, &fsurf, &fbound, neighbourPos, originalIndex, pos1, vel1, dens, pres, oldPos, oldDens, oldPres, oldVel, gridBoundaryIndex, oldBoundaryPos, oldBoundaryVbi, cellStart, cellEnd, cellBoundaryStart, cellBoundaryEnd);
			}
		}
	}

	//finishing gradient and laplacian computations
	fvisc = 2.f * fvisc;
	fsurf = -(sph_params.surfaceTension/pm) * fsurf;

	//computing forces
	fvisc = (pm*sph_params.viscosity) * fvisc;
	fgrav =  pm*sph_params.gravity;

	float3 force_adv = fvisc + fsurf + fbound + fgrav;
	oldForcesAdv[originalIndex] = make_float4(force_adv.x, force_adv.y, force_adv.z, 0.f);
		
	float3 vel_adv = vel1 + dt*(force_adv/pm);
	oldVelAdv[originalIndex] = make_float4(vel_adv.x, vel_adv.y, vel_adv.z, 0.f);

	__syncthreads();

	/*****************
	*  COMPUTE Dii  *
	*****************/
	float3 displacement_factor_fluid = make_float3(0.f, 0.f, 0.f);
	float3 displacement_factor_boundary = make_float3(0.f, 0.f, 0.f);
	for (int z=-1; z<=1; z++)
    {
        for (int y=-1; y<=1; y++)
        {
            for (int x=-1; x<=1; x++)
            {
                const int3 neighbourPos = gridPos + make_int3(x, y, z);
				displacement_factor_fluid = displacement_factor_fluid + computeDisplacementFactorCell(dens, pm, neighbourPos, pos1, oldPos, originalIndex, cellStart, cellEnd, ir, kpg, rd, pm);
				displacement_factor_boundary = displacement_factor_boundary + computeDisplacementFactorBoundaryCell(dens, pm, gridPos, pos1, oldBoundaryPos, oldBoundaryVbi, cellBoundaryStart, cellBoundaryEnd, ir, kpg, rd, pm, dt);
			}
        }
    }

	displacement_factor_fluid = displacement_factor_fluid * (dt*dt);

	oldDiiFluid[originalIndex] = make_float4(displacement_factor_fluid.x, displacement_factor_fluid.y, displacement_factor_fluid.z, 0.f);
	oldDiiBoundary[originalIndex] = make_float4(displacement_factor_boundary.x, displacement_factor_boundary.y, displacement_factor_boundary.z, 0.f);
}

//====================================================================================================  
//====================================================================================================  
//====================================================================================================  

__device__ float rho_adv_fluid(float ir, float pm, unsigned int index, float3 pos1, float3 velAdv1, float kpg, float4* oldPos, float4* oldVelAdv, int3 neighbourPos, unsigned int* cellStart, unsigned int* cellEnd)
{
	const unsigned int gridHash = calcGridHash(neighbourPos);
	const unsigned int startIndex = FETCH(cellStart, gridHash);

	float res  = 0.f;
	if (startIndex != 0xffffffff)
	{ 
		const unsigned int endIndex = FETCH(cellEnd, gridHash);
        for (unsigned int j=startIndex; j<endIndex; j++)
		{
			if(j != index)
			{
				const float3 pos2 = make_float3(FETCH(oldPos, j));
				const float3 velAdv2 = make_float3(FETCH(oldVelAdv, j));
				const float3 v1v2 = velAdv1 - velAdv2;

				const float3 p1p2 = pos1 - pos2;

				if(length(p1p2) < ir)
				{
					res += pm  * dot(v1v2, Wdefault_grad(p1p2, ir, kpg));
				}
			}
		}
	}
	return res;
}

//====================================================================================================  
//====================================================================================================  
//====================================================================================================  
__device__ float rho_adv_boundary(float3 pos1, float3 vel1, float rd, float pm, float ir, float kpg, int3 neighbourPos, float4* oldBoundaryPos, float* oldBoundaryVbi, unsigned int* cellBoundaryStart, unsigned int* cellBoundaryEnd)
{
	const unsigned int gridHash = calcGridHash(neighbourPos);
	const unsigned int startIndex = FETCH(cellBoundaryStart, gridHash);
	float res = 0.f;

	if (startIndex != 0xffffffff)
	{
		const unsigned int endIndex = FETCH(cellBoundaryEnd, gridHash);
		for (unsigned int j=startIndex; j<endIndex; j++)
		{
			const float3 vb = make_float3(0.1f, 0.1f, 0.1f);
			const float3 bpos = make_float3(FETCH(oldBoundaryPos, j));
			const float  vbi  = FETCH(oldBoundaryVbi, j);

			const float3 p1p2 = pos1 - bpos;
			const float3 v1v2 = vel1 - vb;


			const float psi = (rd * vbi);
			res += (psi* dot(v1v2, Wdefault_grad(p1p2, ir, kpg)));
		}
	}
	return res;
}
//====================================================================================================  
//====================================================================================================  
//====================================================================================================  
__device__ float compute_aii_cell(float ir, float dt, float pm, float kpg, float dens, float3 pos1, float3 diif, float3 diib, float4* oldPos, unsigned int* cellStart, unsigned int* cellEnd, int3 neighbourPos, unsigned int index)
{
	const unsigned int gridHash = calcGridHash(neighbourPos);
	const unsigned int startIndex = FETCH(cellStart, gridHash);

	float res  = 0.f;
	if (startIndex != 0xffffffff)
	{ 
		const unsigned int endIndex = FETCH(cellEnd, gridHash);
        for (unsigned int j=startIndex; j<endIndex; j++)
		{
			if(j != index)
			{
				const float3 pos2 = make_float3(FETCH(oldPos, j));
				const float3 p1p2 = pos1 - pos2;

				float3 dji = ( -(dt*dt*pm)/(dens*dens) )*(-1.f * Wdefault_grad(p1p2, ir, kpg));
				res += pm * dot((diif+diib)-dji, Wdefault_grad(p1p2, ir, kpg));

				float3 grad = Wdefault_grad(p1p2, ir, kpg);

				/*if (index == 821) */
				/*{*/
					/*printf("grad = %8f %8f %8f\n", grad.x, grad.y, grad.z);*/
					/*printf("dji = %8f %8f %8f\n", (-1.f * Wdefault_grad(p1p2, ir, kpg)).x, (-1.f * Wdefault_grad(p1p2, ir, kpg)).y, (-1.f * Wdefault_grad(p1p2, ir, kpg)).z);*/
				/*}*/
			}
		}
	}
	return res;
}

//====================================================================================================  
//====================================================================================================  
//====================================================================================================  
__device__ float compute_aii_cell_boundary(float rd, float ir, float kpg, float3 diif, float3 diib, float3 pos1, float4* oldBoundaryPos, float* oldBoundaryVbi, unsigned int* cellBoundaryStart, unsigned int* cellBoundaryEnd, int3 neighbourPos)
{
	const unsigned int gridHash = calcGridHash(neighbourPos);
	const unsigned int startIndex = FETCH(cellBoundaryStart, gridHash);
	float res = 0.f;

	if (startIndex != 0xffffffff)
	{
		const unsigned int endIndex = FETCH(cellBoundaryEnd, gridHash);
		for (unsigned int j=startIndex; j<endIndex; j++)
		{
			const float3 pos2 = make_float3(FETCH(oldBoundaryPos, j));
			const float3 p1p2 = pos1 - pos2;
			const float  vbi  = FETCH(oldBoundaryVbi, j);

			const float psi = rd*vbi;
			res += psi * dot(diif + diib, Wdefault_grad(p1p2, ir, kpg));
		}
	}
	return res;
}

//====================================================================================================  
//====================================================================================================  
//====================================================================================================  

__global__ void computeAdvectionFactor(
	float4                      * oldPos,
	float4                      * oldVel,
	float                       * oldDens,
	float                       * oldPres,
	float4                      * oldForces,
	float4                      * oldCol,
	unsigned int                * cellStart,
	unsigned int                * cellEnd,
	unsigned int                * gridParticleIndex,
	float4					    * oldBoundaryPos,
	float                       * oldBoundaryVbi,
	unsigned int                * cellBoundaryStart,
	unsigned int                * cellBoundaryEnd,
	unsigned int                * gridBoundaryIndex,
	float                       * oldDensAdv,
	float                       * oldDensCorr,
	float                       * oldP_l,
	float                       * oldPreviousP,
	float                       * oldAii,
	float4                      * oldVelAdv,
	float4                      * oldForcesAdv,
	float4                      * oldForcesP,
	float4                      * oldDiiFluid,
	float4                      * oldDiiBoundary,
	float4                      * oldSumDij,
	float4                      * oldNormal,
	unsigned int numParticles,
	unsigned int numBoundaries,
	unsigned int numCells)
{
    const unsigned int index = blockIdx.x*blockDim.x + threadIdx.x;
    if (index >= numParticles) return;
	const unsigned int originalIndex = gridParticleIndex[index];

	//global memory reads
	const float3 pos1 = make_float3(FETCH(oldPos, originalIndex));
	const float3 vel1 = make_float3(FETCH(oldVel, originalIndex));
	const float3 velAdv1 = make_float3(FETCH(oldVelAdv, originalIndex));
	const float  dens = FETCH(oldDens, originalIndex);
	const float3 diif = make_float3(FETCH(oldDiiFluid, originalIndex));
	const float3 diib = make_float3(FETCH(oldDiiBoundary, originalIndex));

	//grid computation
    const int3 gridPos = calcGridPos(pos1);

	//const memory reads
	const float kpg= sph_params.kpoly_grad;
	const float pm = sph_params.particleMass;
	const float ir = sph_params.interactionRadius;
	const float rd = sph_params.restDensity;
	const float dt = sph_params.timestep;

	/*********************
	*  COMPUTE RHO_ADV  *
	*********************/
	float rho_advf= 0.f;
	float rho_advb= 0.f;

	//loop over fluid particles and boundary particles
	for (int z=-1; z<=1; z++)
	{
		for (int y=-1; y<=1; y++)
		{
			for (int x=-1; x<=1; x++)
			{
				const int3 neighbourPos = gridPos + make_int3(x, y, z);

				rho_advf = rho_advf + rho_adv_fluid(ir, pm, originalIndex, pos1, velAdv1, kpg, oldPos, oldVelAdv, neighbourPos, cellStart, cellEnd);
				rho_advb += rho_adv_boundary(pos1, vel1, rd, pm, ir, kpg, neighbourPos, oldBoundaryPos, oldBoundaryVbi, cellBoundaryStart, cellBoundaryEnd);
			}
		}
	}

	float rho_adv = dens + dt*(rho_advf + rho_advb);
	oldDensAdv[originalIndex] = rho_adv; 

	/*******************
	*  COMPUTE P_i^0  *
	*******************/
	//FIXME
	oldP_l[originalIndex] = 0.5f * oldPres[originalIndex]; 

	/*****************
	*  COMPUTE AII  *
	*****************/
	float aii = 0.f;
	for (int z=-1; z<=1; z++)
	{
		for (int y=-1; y<=1; y++)
		{
			for (int x=-1; x<=1; x++)
			{
				const int3 neighbourPos = gridPos + make_int3(x, y, z);

				aii += compute_aii_cell(ir, dt, pm, kpg, dens, pos1, diif, diib, oldPos, cellStart, cellEnd, neighbourPos, originalIndex);
				aii += compute_aii_cell_boundary(rd, ir, kpg, diif, diib, pos1, oldBoundaryPos, oldBoundaryVbi, cellBoundaryStart, cellBoundaryEnd, neighbourPos);
			}
		}
	}
	oldAii[originalIndex] = aii;
}

//====================================================================================================  
//====================================================================================================  
//====================================================================================================  

__device__ float3 dijpjcell(float ir, float pm, float kpg, float3 pos1, float4* oldPos, float* oldDens, float* oldP_l, unsigned int index, unsigned int *cellStart, unsigned int *cellEnd, int3 neighbourPos)
{
	float3 res = make_float3(0.f, 0.f, 0.f);
	const unsigned int gridHash = calcGridHash(neighbourPos);
	const unsigned int startIndex = FETCH(cellStart, gridHash);

	if (startIndex != 0xffffffff)
	{ 
		const unsigned int endIndex = FETCH(cellEnd, gridHash);
        for (unsigned int j=startIndex; j<endIndex; j++)
		{
			if(j != index)
			{
				//todo
				const float3 pos2 = make_float3(FETCH(oldPos, j));
				const float3 p1p2 = pos1 - pos2;
				const float p_lj = FETCH(oldP_l, j);
				const float densj = FETCH(oldDens, j);

				res = res + ((-pm/(densj*densj))*p_lj*Wdefault_grad(p1p2, ir, kpg));
			}
		}
	}
	return res;
}

__global__ void computeSumDijPj(
	float4                      * oldPos,
	float4                      * oldVel,
	float                       * oldDens,
	float                       * oldPres,
	float4                      * oldForces,
	float4                      * oldCol,
	unsigned int                * cellStart,
	unsigned int                * cellEnd,
	unsigned int                * gridParticleIndex,
	float4					    * oldBoundaryPos,
	float                       * oldBoundaryVbi,
	unsigned int                * cellBoundaryStart,
	unsigned int                * cellBoundaryEnd,
	unsigned int                * gridBoundaryIndex,
	float                       * oldDensAdv,
	float                       * oldDensCorr,
	float                       * oldP_l,
	float                       * oldPreviousP,
	float                       * oldAii,
	float4                      * oldVelAdv,
	float4                      * oldForcesAdv,
	float4                      * oldForcesP,
	float4                      * oldDiiFluid,
	float4                      * oldDiiBoundary,
	float4                      * oldSumDij,
	float4                      * oldNormal,
	unsigned int numParticles,
	unsigned int numBoundaries,
	unsigned int numCells
		)
{
    const unsigned int index = blockIdx.x*blockDim.x + threadIdx.x;
    if (index >= numParticles) return;
	const unsigned int originalIndex = gridParticleIndex[index];

	//global reads
	const float3 pos1 = make_float3(FETCH(oldPos, originalIndex));

	//grid compute
    const int3 gridPos = calcGridPos(pos1);

	//const reads
	const float ir = sph_params.interactionRadius;
	const float pm = sph_params.particleMass;
	const float kpg= sph_params.kpoly_grad;
	const float dt = sph_params.timestep;

	float3 dijpj = make_float3(0.f, 0.f, 0.f);
	for (int z=-1; z<=1; z++)
	{
		for (int y=-1; y<=1; y++)
		{
			for (int x=-1; x<=1; x++)
			{
				const int3 neighbourPos = gridPos + make_int3(x, y, z);

				dijpj = dijpj + dijpjcell(ir, pm, kpg, pos1, oldPos, oldDens, oldP_l, originalIndex, cellStart, cellEnd, neighbourPos);
			}
		}
	}
	dijpj = dijpj * (dt*dt);

	/*if (length(dijpj) != length(dijpj)) dijpj = make_float3(0.f, 0.f, 0.f);//FIXME nan issues*/

	oldSumDij[originalIndex] = make_float4(dijpj.x, dijpj.y, dijpj.z, 0.f);
}

//====================================================================================================  
//====================================================================================================  
//====================================================================================================  
__global__ void computePressure(
	float4                      * oldPos,
	float4                      * oldVel,
	float                       * oldDens,
	float                       * oldPres,
	float4                      * oldForces,
	float4                      * oldCol,
	unsigned int                * cellStart,
	unsigned int                * cellEnd,
	unsigned int                * gridParticleIndex,
	float4					    * oldBoundaryPos,
	float                       * oldBoundaryVbi,
	unsigned int                * cellBoundaryStart,
	unsigned int                * cellBoundaryEnd,
	unsigned int                * gridBoundaryIndex,
	float                       * oldDensAdv,
	float                       * oldDensCorr,
	float                       * oldP_l,
	float                       * oldPreviousP,
	float                       * oldAii,
	float4                      * oldVelAdv,
	float4                      * oldForcesAdv,
	float4                      * oldForcesP,
	float4                      * oldDiiFluid,
	float4                      * oldDiiBoundary,
	float4                      * oldSumDij,
	float4                      * oldNormal,
	unsigned int numParticles,
	unsigned int numBoundaries,
	unsigned int numCells
		)
{
	const unsigned int index = blockIdx.x*blockDim.x + threadIdx.x;
    if (index >= numParticles) return;
	const unsigned int originalIndex = gridParticleIndex[index];

	//global reads
	const float3 pos1 = make_float3(FETCH(oldPos, originalIndex));
	const float dens = FETCH(oldDens, originalIndex);
	float  p_l = FETCH(oldP_l, originalIndex);
	const float previous_p_l = p_l;
	const float rho_adv = FETCH(oldDensAdv, originalIndex);
	const float3 sum_dij = make_float3(FETCH(oldSumDij, originalIndex));
	const float3 diif = make_float3(FETCH(oldDiiFluid, originalIndex));
	const float3 diib = make_float3(FETCH(oldDiiBoundary, originalIndex));
	const float aii = FETCH(oldAii, originalIndex);

	//grid compute
    const int3 gridPos = calcGridPos(pos1);

	//const reads
	const float ir = sph_params.interactionRadius;
	const float pm = sph_params.particleMass;
	const float kpg= sph_params.kpoly_grad;
	const float dt = sph_params.timestep;
	const float rd = sph_params.restDensity;

	float3 dijpj = make_float3(0.f, 0.f, 0.f);

	float fsum = 0.f;
	float bsum = 0.f;
		
	for (int z=-1; z<=1; z++)
	{
		for (int y=-1; y<=1; y++)
		{
			for (int x=-1; x<=1; x++)
			{
				const int3 neighbourPos = gridPos + make_int3(x, y, z);
				const unsigned int gridHash = calcGridHash(neighbourPos);
				const unsigned int startIndex = FETCH(cellStart, gridHash);

				if (startIndex != 0xffffffff)
				{ 
					const unsigned int endIndex = FETCH(cellEnd, gridHash);
					for (unsigned int j=startIndex; j<endIndex; j++)
					{
						if(j != index)
						{
							//todo
							const float3 pos2 = make_float3(FETCH(oldPos, j));
							const float3 p1p2 = pos1 - pos2;
							const float p_lj = FETCH(oldP_l, j);

							float3 dji = -(dt*dt*pm)/(dens*dens)*(-1.f * Wdefault_grad(p1p2, ir, kpg));//FIXME nan issues
							/*if (length(dji) != length(dji)) dji = make_float3(0.f, 0.f ,0.f) ;*/

							const float3 diifj = make_float3(FETCH(oldDiiFluid, j));
							const float3 diibj = make_float3(FETCH(oldDiiBoundary, j));
							const float3 sum_dijj = make_float3(FETCH(oldSumDij, j));
							float3 aux = sum_dij - (diifj+diibj)*p_lj - (sum_dijj - dji*p_l);

							fsum += pm*dot(aux, Wdefault_grad(p1p2, ir, kpg));
						}
					}
				}

				const unsigned int startIndexB = FETCH(cellBoundaryStart, gridHash);
				if (startIndexB != 0xffffffff) 
				{
					const unsigned int endIndexB = FETCH(cellBoundaryEnd, gridHash);
					for (unsigned int j=startIndex; j<endIndexB; j++)
					{
						const float3 posb = make_float3(FETCH(oldBoundaryPos, j));
						const float3 p1p2 = pos1 - posb;
						const float vbi = FETCH(oldBoundaryVbi, j);
						const float psi = rd * vbi;
						bsum += psi * dot(sum_dij, Wdefault_grad(p1p2, ir, kpg)); 
					}
				}
			}
		}
	}

	if (fsum != fsum) fsum = 0.f;
	if (bsum != bsum) bsum = 0.f;

	float omega = 0.5f;
	float rho_corr = rho_adv + fsum + bsum;

    if(fabs(aii)>FLT_EPSILON)
    {
        p_l = (1.f-omega)*previous_p_l + (omega/aii)*(rd - rho_corr);
    }
    else
    {
        p_l = 0.0;
    }

    float p = fmax(p_l, 0.f);
    p_l = p;
    rho_corr += aii*previous_p_l;


	//global writes
	oldP_l[originalIndex] = p_l;
	oldPres[originalIndex] = p_l;
	oldDensCorr[originalIndex] = rho_corr;

}

//====================================================================================================  
//====================================================================================================  
//====================================================================================================  
__global__ void computePressureForce(
		float4                      * oldPos,
		float4                      * oldVel,
		float                       * oldDens,
		float                       * oldPres,
		float4                      * oldForces,
		float4                      * oldCol,
		unsigned int                * cellStart,
		unsigned int                * cellEnd,
		unsigned int                * gridParticleIndex,
		float4					    * oldBoundaryPos,
		float                       * oldBoundaryVbi,
		unsigned int                * cellBoundaryStart,
		unsigned int                * cellBoundaryEnd,
		unsigned int                * gridBoundaryIndex,
		float                       * oldDensAdv,
		float                       * oldDensCorr,
		float                       * oldP_l,
		float                       * oldPreviousP,
		float                       * oldAii,
		float4                      * oldVelAdv,
		float4                      * oldForcesAdv,
		float4                      * oldForcesP,
		float4                      * oldDiiFluid,
		float4                      * oldDiiBoundary,
		float4                      * oldSumDij,
		float4                      * oldNormal,
		unsigned int numParticles,
		unsigned int numBoundaries,
		unsigned int numCells
		)
{
	const unsigned int index = blockIdx.x*blockDim.x + threadIdx.x;
    if (index >= numParticles) return;
	const unsigned int originalIndex = gridParticleIndex[index];

	//global reads
	const float3 pos1 = make_float3(FETCH(oldPos, originalIndex));
	const float3 velAdv1 = make_float3(FETCH(oldVelAdv, originalIndex));
	const float p = FETCH(oldPres, originalIndex);
	const float dens = FETCH(oldDens, originalIndex);

	//grid compute
    const int3 gridPos = calcGridPos(pos1);

	//const reads
	const float ir = sph_params.interactionRadius;
	const float pm = sph_params.particleMass;
	const float kpg= sph_params.kpoly_grad;
	const float rd = sph_params.restDensity;
	
	/***************************
	*  UPDATE PRESSURE FORCE  *
	***************************/
	float3 fpres_res = make_float3(0.f, 0.f, 0.f);

	for (int z=-1; z<=1; z++)
	{
		for (int y=-1; y<=1; y++)
		{
			for (int x=-1; x<=1; x++)
			{
				const int3 neighbourPos = gridPos + make_int3(x, y, z);
				const unsigned int gridHash = calcGridHash(neighbourPos);
				const unsigned int startIndex = FETCH(cellStart, gridHash);

				if (startIndex != 0xffffffff)
				{ 
					const unsigned int endIndex = FETCH(cellEnd, gridHash);
					for (unsigned int j=startIndex; j<endIndex; j++)
					{
						if(j != index)
						{
							//todo
							const float3 pos2 = make_float3(FETCH(oldPos, j));
							const float3 p1p2 = pos1 - pos2;
							const float pj = FETCH(oldPres, j);
							const float densj = FETCH(oldDens, j);

							fpres_res += -pm*pm*( p/(dens*dens) + pj/(densj*densj) ) * Wdefault_grad(p1p2, ir, kpg);
						}
					}
				}

				const unsigned int startIndexB = FETCH(cellBoundaryStart, gridHash);
				if (startIndexB != 0xffffffff) 
				{
					const unsigned int endIndexB = FETCH(cellBoundaryEnd, gridHash);
					for (unsigned int j=startIndex; j<endIndexB; j++)
					{
						const float3 posb = make_float3(FETCH(oldBoundaryPos, j));
						const float3 p1p2 = pos1 - posb;
						const float vbi = FETCH(oldBoundaryVbi, j);
						const float psi = rd * vbi;

						fpres_res += pm*psi*( p/(dens*dens) ) * Wdefault_grad(p1p2, ir, kpg);
					}
				}
			}
		}
	}
	if (length(fpres_res) != length(fpres_res)) fpres_res = make_float3(0.f, 0.f, 0.f);
	oldForcesP[originalIndex] = make_float4(fpres_res.x, fpres_res.y, fpres_res.z, 0.f);
	__syncthreads();
}

__global__ void iisph_integrate(
			float4* oldPos,
			float4* oldVel,
			float4* oldVelAdv,
			float4* oldForcesP,
			unsigned int* gridParticleIndex,
			unsigned int numParticles
			)
{
	const unsigned int index = blockIdx.x*blockDim.x + threadIdx.x;
    if (index >= numParticles) return;

	const unsigned int originalIndex = gridParticleIndex[index];

	const float3 pos1 = make_float3(FETCH(oldPos, originalIndex));
	const float3 velAdv1 = make_float3(FETCH(oldVelAdv, originalIndex));
	const float3 fpres1 = make_float3(FETCH(oldForcesP,originalIndex));

	const float dt = sph_params.timestep;
	const float pm = sph_params.particleMass;

	float3 newVel = velAdv1 + (dt*fpres1/pm);
	float3 newPos = pos1+ (dt*newVel);

	oldPos[originalIndex] = make_float4(newPos.x, newPos.y, newPos.z, 1.f); // FUCK !
	oldVel[originalIndex] = make_float4(newVel.x, newVel.y, newVel.z, 0.f);
}


#endif//_PARTICLES_KERNEL_IMPL_CUH

