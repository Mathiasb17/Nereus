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
void registerGLBufferObject(SUint vbo, struct cudaGraphicsResource **cuda_vbo_resource);
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
void computeGridSize(SUint n, SUint blockSize, SUint &numBlocks, SUint &numThreads);

//==================================================================================================== 
//==================================================================================================== 
//==================================================================================================== 
void integrateSystem(SReal *pos,
					 SReal *vel,
					 SReal *forces,
					 SReal deltaTime,
					 SUint numParticles);

//==================================================================================================== 
//==================================================================================================== 
//==================================================================================================== 
void calcHash(SUint  *gridParticleHash,
			  SUint  *gridParticleIndex,
			  SReal *pos,
			  int    numParticles);

//==================================================================================================== 
//==================================================================================================== 
//==================================================================================================== 
void reorderDataAndFindCellStartDBoundary(SUint *cellStart,
										SUint *cellEnd,
										SReal *sortedPos,
										SReal *sortedVbi,
										SUint *gridParticleHash,
										SUint *gridParticleIndex,
										SReal *oldPos,
										SReal *oldVbi,
										SUint numBoundaries,
										SUint numCells
										);

//==================================================================================================== 
//==================================================================================================== 
//==================================================================================================== 
void reorderDataAndFindCellStart(SUint  *cellStart,
								 SUint  *cellEnd,
								 SReal *sortedPos,
								 SReal *sortedVel,
								 SReal *sortedDens,
								 SReal *sortedPres,
								 SReal *sortedForces,
								 SReal *sortedCol,
								 SUint  *gridParticleHash,
								 SUint  *gridParticleIndex,
								 SReal *oldPos,
								 SReal *oldVel,
								 SReal *oldDens,
								 SReal *oldPres,
								 SReal *oldForces,
								 SReal *oldCol,
								 SUint   numParticles,
								 SUint   numCells);

//==================================================================================================== 
//==================================================================================================== 
//==================================================================================================== 
SVec3 BBMin(SReal *sortedBoundaryPos, SUint numBoundaries);
//==================================================================================================== 
//==================================================================================================== 
//==================================================================================================== 
SVec3 BBMax(SReal *sortedBoundaryPos, SUint numBoundaries);

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
		SUint  *gridParticleIndex,
		SUint  *cellStart,
		SUint  *cellEnd,
		SUint *gridBoundaryIndex,
		SUint *cellBoundaryStart,
		SUint *cellBoundaryEnd,
		SUint   numParticles,
		SUint   numCells,
		SUint   numBoundaries);

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
		SUint  *gridParticleIndex,
		SUint  *cellStart,
		SUint  *cellEnd,
		SUint   numParticles,
		SUint   numCells);

//==================================================================================================== 
//==================================================================================================== 
//==================================================================================================== 
SReal maxDensity(SReal* dDensities, SUint numParticles);

//==================================================================================================== 
//==================================================================================================== 
//==================================================================================================== 
SVec4 maxVelocity(SReal* dVelocities, SUint numParticles);

//==================================================================================================== 
//==================================================================================================== 
//==================================================================================================== 
void sortParticles(SUint *dGridParticleHash, SUint *dGridParticleIndex, SUint numParticles);

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
	SUint                * cellStart,
	SUint                * cellEnd,
	SUint                * gridParticleIndex,
	SReal						* sortedBoundaryPos,
	SReal						* sortedBoundaryVbi,
	SUint                * cellBoundaryStart,
	SUint                * cellBoundaryEnd,
	SUint                * gridBoundaryIndex,
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
	SUint numParticles,
	SUint numBoundaries,
	SUint numCells);

//==================================================================================================== 
//==================================================================================================== 
//==================================================================================================== 
void pressureSolve(SReal* sortedPos, SReal* sortedVel, SReal* sortedDens, SReal* sortedPres, SReal* sortedForces, SReal* sortedCol, SUint* cellStart, SUint* cellEnd, SUint* gridParticleIndex,
				  SReal* sortedBoundaryPos, SReal* sortedBoundaryVbi,
				  SUint* cellBoundaryStart, SUint* cellBoundaryEnd, SUint* gridBoundaryIndex, SReal* sortedDensAdv, SReal* sortedDensCorr, SReal* sortedP_l,  SReal* sortedPreviousP, 
				  SReal* sortedAii, SReal* sortedVelAdv, SReal* sortedForcesAdv, SReal* sortedForcesP, SReal* sortedDiiFluid, SReal* sortedDiiBoundary, SReal* sortedSumDij, SReal* sortedNormal,
				  SUint numParticles, SUint numBoundaries, SUint numCells);
/************
*  PCISPH  *
************/
//==================================================================================================== 
//==================================================================================================== 
//==================================================================================================== 
void pcisph_internalForces(SReal* sortedPos, SReal* sortedVel, SReal* sortedDens, SReal* sortedPres, SReal* sortedForces, SReal* sortedCol, SUint* cellStart, SUint* cellEnd, SUint* gridParticleIndex,
				SReal* sortedBoundaryPos, SReal* sortedBoundaryVbi, SUint* cellBoundaryStart, SUint* cellBoundaryEnd, SUint* gridBoundaryIndex, SReal* sortedRhoAdv, SReal* sortedPosAdv, SReal* sortedVelAdv, 
				SReal* sortedForcesAdv, SReal* sortedForcesP, SReal* sortedNormal, SUint numParticles, SUint numBoundaries, SUint numCells);
//==================================================================================================== 
//==================================================================================================== 
//==================================================================================================== 
void pcisph_pressureSolve(SReal* sortedPos, SReal* sortedVel, SReal* sortedDens, SReal* sortedPres, SReal* sortedForces, SReal* sortedCol, SUint* cellStart, SUint* cellEnd, SUint* gridParticleIndex,
				SReal* sortedBoundaryPos, SReal* sortedBoundaryVbi, SUint* cellBoundaryStart, SUint* cellBoundaryEnd, SUint* gridBoundaryIndex, SReal* sortedRhoAdv, SReal* sortedPosAdv, SReal* sortedVelAdv, 
				SReal* sortedForcesAdv, SReal* sortedForcesP, SReal* sortedNormal, SUint numParticles, SUint numBoundaries, SUint numCells);
//==================================================================================================== 
//==================================================================================================== 
//==================================================================================================== 
EXTERN_C_END
#endif /* ifndef SPH_CUH */
