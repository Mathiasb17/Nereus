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

//texture for pcisph
texture<float4, 1, cudaReadModeElementType> oldPosStarTex;
texture<float4, 1, cudaReadModeElementType> oldVelStarTex;
texture<float, 1, cudaReadModeElementType> oldDensStarTex;
texture<float, 1, cudaReadModeElementType> oldPresStarTex;
texture<float, 1, cudaReadModeElementType> oldDensErrorTex;

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

#if 0

	   /* if (pos.x > 1.0f - sph_params.particleRadius)*/
		/*{*/
			/*pos.x = 1.0f - sph_params.particleRadius;*/
			/*vel.x = 0.45f * -vel.x;*/
		/*}*/

		/*if (pos.x < -1.0f + sph_params.particleRadius)*/
		/*{*/
			/*pos.x = -1.0f + sph_params.particleRadius;*/
			/*vel.x = 0.45f * -vel.x;*/
		/*}*/

		/*if (pos.y > 1.0f - sph_params.particleRadius)*/
		/*{*/
			/*pos.y = 1.0f - sph_params.particleRadius;*/
			/*vel.y = 0.45f * -vel.y;*/
		/*}*/

		/*if (pos.z > 1.0f - sph_params.particleRadius)*/
		/*{*/
			/*pos.z = 1.0f - sph_params.particleRadius;*/
			/*vel.z = 0.45f * -vel.z;*/
		/*}*/

		/*if (pos.z < -1.0f + sph_params.particleRadius)*/
		/*{*/
			/*pos.z = -1.0f + sph_params.particleRadius;*/
			/*vel.z = 0.45f * -vel.z;*/
		/*}*/


		/*if (pos.y < -1.0f + sph_params.particleRadius)*/
		/*{*/
			/*pos.y = -1.0f + sph_params.particleRadius;*/
			/*vel.y = 0.45f * -vel.y;*/
		/*}*/

#endif

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
		//TODO
		float4 pos = FETCH(oldBoundaryPos, sortedIndex);       // macro does either global read or texture fetch
		float vbi = FETCH(oldBoundaryVbi, sortedIndex);       // see particles_kernel.cuh
		
		/*sortedPos[index] = pos;*/
		/*sortedVbi[index] = vbi;*/
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
		sortedVel[index] = vel;
		sortedPos[index] = pos;
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
	
	/*printf("dens = %f\n", dens);*/
	//if (nbVois > 40) printf("nbVois too large : %5d\n", nbVois ); ;

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

	const float ksurf1 = sph_params.ksurf1;
	const float ksurf2 = sph_params.ksurf2;

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
					*fpres = *fpres + m2 * ( pres/d1sq + pres2/d2sq ) *kpressure_grad;

					const float a = dot(p1p2, kvisco_grad);
					const float b = dot(p1p2,p1p2) + 0.01f*ir*ir;
					*fvisc = *fvisc + m2/dens2  * v1v2 * (a/b);

					/**fsurf = *fsurf + m2 * p1p2 * Wdefault(p1p2, ir, sph_params.kpoly) ;*/
					float gamma = sph_params.surfaceTension;
					*fsurf = *fsurf + (-gamma * m2*m2 * Cakinci(p1p2, ir, ksurf1, ksurf2)*(p1p2/length(p1p2)));
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

	float3 f = fpres + fvisc + m1*sph_params.gravity + fsurf + fbound;
	float4 res = make_float4(f.x, f.y, f.z, 0);

	oldForces[originalIndex] = res;
}

//====================================================================================================  
//====================================================================================================  
//====================================================================================================  

__global__ void iisph_compute_displacement_factor()
{

}

__global__ void iisph_compute_advection_factor()
{

}

__global__ void compute_sum_pressure_movement()
{

}

__global__ void compute_pressure()
{

}

__global__ void compute_predicted_density()
{

}

__global__ void compute_density_error()
{

}

#endif /* ifndef _PARTICLES_KERNEL_IMPL_CUH */

