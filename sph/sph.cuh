#ifndef SPH_CUH
#define SPH_CUH 

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <helper_cuda.h>
#include <helper_cuda_gl.h>

#include <helper_functions.h>

#include <stdio.h>
#include <math.h>

#include <helper_math.h>
#include <math_constants.h>

#include "sph_kernel.cuh"

EXTERN_C_BEGIN
//==================================================================================================== 
//==================================================================================================== 
//==================================================================================================== 
void cudaInit(int argc, char **argv);

void allocateArray(void **devPtr, int size);
void freeArray(void *devPtr);

void threadSync();

void copyArrayFromDevice(void *host, const void *device, struct cudaGraphicsResource **cuda_vbo_resource, int size);
void copyArrayToDevice(void *device, const void *host, int offset, int size);
void registerGLBufferObject(unsigned int vbo, struct cudaGraphicsResource **cuda_vbo_resource);
void unregisterGLBufferObject(struct cudaGraphicsResource *cuda_vbo_resource);
void *mapGLBufferObject(struct cudaGraphicsResource **cuda_vbo_resource);
void unmapGLBufferObject(struct cudaGraphicsResource *cuda_vbo_resource);

//==================================================================================================== 
//==================================================================================================== 
//==================================================================================================== 
void setParameters(SphSimParams *hostParams);

//==================================================================================================== 
//==================================================================================================== 
//==================================================================================================== 
void computeGridSize(unsigned int n, unsigned int blockSize, unsigned int &numBlocks, unsigned int &numThreads);

//==================================================================================================== 
//==================================================================================================== 
//==================================================================================================== 
void integrateSystem(SReal *pos,
					 SReal *vel,
					 SReal *forces,
					 SReal deltaTime,
					 unsigned int numParticles);

//==================================================================================================== 
//==================================================================================================== 
//==================================================================================================== 
void calcHash(unsigned int  *gridParticleHash,
			  unsigned int  *gridParticleIndex,
			  SReal *pos,
			  int    numParticles);

//==================================================================================================== 
//==================================================================================================== 
//==================================================================================================== 
void reorderDataAndFindCellStartDBoundary(unsigned int *cellStart,
										unsigned int *cellEnd,
										SReal *sortedPos,
										SReal *sortedVbi,
										unsigned int *gridParticleHash,
										unsigned int *gridParticleIndex,
										SReal *oldPos,
										SReal *oldVbi,
										unsigned int numBoundaries,
										unsigned int numCells
										);

//==================================================================================================== 
//==================================================================================================== 
//==================================================================================================== 
void reorderDataAndFindCellStart(unsigned int  *cellStart,
								 unsigned int  *cellEnd,
								 SReal *sortedPos,
								 SReal *sortedVel,
								 SReal *sortedDens,
								 SReal *sortedPres,
								 SReal *sortedForces,
								 SReal *sortedCol,
								 unsigned int  *gridParticleHash,
								 unsigned int  *gridParticleIndex,
								 SReal *oldPos,
								 SReal *oldVel,
								 SReal *oldDens,
								 SReal *oldPres,
								 SReal *oldForces,
								 SReal *oldCol,
								 unsigned int   numParticles,
								 unsigned int   numCells);


//==================================================================================================== 
//==================================================================================================== 
//==================================================================================================== 
void computeDensityPressure(
		SReal *sortedPos,
		SReal *sortedVel,
		SReal *sortedDens,
		SReal *sortedPres,
		SReal *sortedForces,
		SReal *sortedCol,
		SReal *sortedBoundaryPos,
		SReal *sortedBoundaryVbi,
		unsigned int  *gridParticleIndex,
		unsigned int  *cellStart,
		unsigned int  *cellEnd,
		unsigned int *gridBoundaryIndex,
		unsigned int *cellBoundaryStart,
		unsigned int *cellBoundaryEnd,
		unsigned int   numParticles,
		unsigned int   numCells,
		unsigned int   numBoundaries);

//==================================================================================================== 
//==================================================================================================== 
//==================================================================================================== 
void computePciDensityPressure(
		SphSimParams *hostParams,
		SReal* sortedPos,
		SReal* sortedVel,
		SReal* sortedDens,
		SReal* sortedPres,
		SReal* sortedForces,
		SReal* sortedCol,
		SReal* sortedPosStar,
		SReal* sortedVelStar,
		SReal* sortedDensStar,
		SReal* sortedDensError,
		unsigned int  *gridParticleIndex,
		unsigned int  *cellStart,
		unsigned int  *cellEnd,
		unsigned int   numParticles,
		unsigned int   numCells);

//==================================================================================================== 
//==================================================================================================== 
//==================================================================================================== 
SReal maxDensity(SReal* dDensities, unsigned int numParticles);

//==================================================================================================== 
//==================================================================================================== 
//==================================================================================================== 
SVec3 maxVelocity(SReal* dVelocities, unsigned int numParticles);

//==================================================================================================== 
//==================================================================================================== 
//==================================================================================================== 
void sortParticles(unsigned int *dGridParticleHash, unsigned int *dGridParticleIndex, unsigned int numParticles);

/***********
*  IISPH  *
***********/
//==================================================================================================== 
//==================================================================================================== 
//==================================================================================================== 
void predictAdvection(SReal* sortedPos,
	SReal                       * sortedVel,
	SReal                       * sortedDens,
	SReal                       * sortedPres,
	SReal                       * sortedForces,
	SReal                       * sortedCol,
	unsigned int                * cellStart,
	unsigned int                * cellEnd,
	unsigned int                * gridParticleIndex,
	SReal						* sortedBoundaryPos,
	SReal						* sortedBoundaryVbi,
	unsigned int                * cellBoundaryStart,
	unsigned int                * cellBoundaryEnd,
	unsigned int                * gridBoundaryIndex,
	SReal                       * sortedDensAdv,
	SReal                       * sortedDensCorr,
	SReal                       * sortedP_l,
	SReal                       * sortedPreviousP,
	SReal                       * sortedAii,
	SReal                       * sortedVelAdv,
	SReal                       * sortedForcesAdv,
	SReal                       * sortedForcesP,
	SReal                       * sortedDiiFluid,
	SReal                       * sortedDiiBoundary,
	SReal                       * sortedSumDij,
	SReal                       * sortedNormal,
	unsigned int numParticles,
	unsigned int numBoundaries,
	unsigned int numCells);

//==================================================================================================== 
//==================================================================================================== 
//==================================================================================================== 
void pressureSolve(SReal* sortedPos, SReal* sortedVel, SReal* sortedDens, SReal* sortedPres, SReal* sortedForces, SReal* sortedCol, unsigned int* cellStart, unsigned int* cellEnd, unsigned int* gridParticleIndex,
				  SReal* sortedBoundaryPos, SReal* sortedBoundaryVbi,
				  unsigned int* cellBoundaryStart, unsigned int* cellBoundaryEnd, unsigned int* gridBoundaryIndex, SReal* sortedDensAdv, SReal* sortedDensCorr, SReal* sortedP_l,  SReal* sortedPreviousP, 
				  SReal* sortedAii, SReal* sortedVelAdv, SReal* sortedForcesAdv, SReal* sortedForcesP, SReal* sortedDiiFluid, SReal* sortedDiiBoundary, SReal* sortedSumDij, SReal* sortedNormal,
				  unsigned int numParticles, unsigned int numBoundaries, unsigned int numCells);
/************
*  PCISPH  *
************/
//==================================================================================================== 
//==================================================================================================== 
//==================================================================================================== 
void pcisph_internalForces(SReal* sortedPos, SReal* sortedVel, SReal* sortedDens, SReal* sortedPres, SReal* sortedForces, SReal* sortedCol, unsigned int* cellStart, unsigned int* cellEnd, unsigned int* gridParticleIndex,
				SReal* sortedBoundaryPos, SReal* sortedBoundaryVbi, unsigned int* cellBoundaryStart, unsigned int* cellBoundaryEnd, unsigned int* gridBoundaryIndex, SReal* sortedRhoAdv, SReal* sortedVelAdv, 
				SReal* sortedForcesAdv, SReal* sortedForcesP, SReal* sortedNormal, unsigned int numParticles, unsigned int numBoundaries, unsigned int numCells);
//==================================================================================================== 
//==================================================================================================== 
//==================================================================================================== 
void pcisph_pressureSolve(SReal* sortedPos, SReal* sortedVel, SReal* sortedDens, SReal* sortedPres, SReal* sortedForces, SReal* sortedCol, unsigned int* cellStart, unsigned int* cellEnd, unsigned int* gridParticleIndex,
				SReal* sortedBoundaryPos, SReal* sortedBoundaryVbi, unsigned int* cellBoundaryStart, unsigned int* cellBoundaryEnd, unsigned int* gridBoundaryIndex, SReal* sortedRhoAdv, SReal* sortedVelAdv, 
				SReal* sortedForcesAdv, SReal* sortedForcesP, SReal* sortedNormal, unsigned int numParticles, unsigned int numBoundaries, unsigned int numCells);
//==================================================================================================== 
//==================================================================================================== 
//==================================================================================================== 
EXTERN_C_END
#endif /* ifndef SPH_CUH */
