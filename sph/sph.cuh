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

extern "C"
{
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

    void setParameters(SphSimParams *hostParams);

	void computeGridSize(unsigned int n, unsigned int blockSize, unsigned int &numBlocks, unsigned int &numThreads);

    void integrateSystem(float *pos,
                         float *vel,
                         float *forces,
                         float deltaTime,
                         unsigned int numParticles);

    void calcHash(unsigned int  *gridParticleHash,
                  unsigned int  *gridParticleIndex,
                  float *pos,
                  int    numParticles);

	void reorderDataAndFindCellStartDBoundary(unsigned int *cellStart,
											unsigned int *cellEnd,
											float *sortedPos,
											float *sortedVbi,
											unsigned int *gridParticleHash,
											unsigned int *gridParticleIndex,
											float *oldPos,
											float *oldVbi,
											unsigned int numBoundaries,
											unsigned int numCells
											);

    void reorderDataAndFindCellStart(unsigned int  *cellStart,
                                     unsigned int  *cellEnd,
                                     float *sortedPos,
                                     float *sortedVel,
                                     float *sortedDens,
                                     float *sortedPres,
                                     float *sortedForces,
                                     float *sortedCol,
                                     unsigned int  *gridParticleHash,
                                     unsigned int  *gridParticleIndex,
                                     float *oldPos,
                                     float *oldVel,
                                     float *oldDens,
                                     float *oldPres,
                                     float *oldForces,
                                     float *oldCol,
                                     unsigned int   numParticles,
                                     unsigned int   numCells);


void computeDensityPressure(
			float *sortedPos,
			float *sortedVel,
			float *sortedDens,
			float *sortedPres,
			float *sortedForces,
			float *sortedCol,
			float *sortedBoundaryPos,
			float *sortedBoundaryVbi,
			unsigned int  *gridParticleIndex,
			unsigned int  *cellStart,
			unsigned int  *cellEnd,
			unsigned int *gridBoundaryIndex,
			unsigned int *cellBoundaryStart,
			unsigned int *cellBoundaryEnd,
			unsigned int   numParticles,
			unsigned int   numCells,
			unsigned int   numBoundaries);

	void computePciDensityPressure(
			SphSimParams *hostParams,
			float* sortedPos,
			float* sortedVel,
			float* sortedDens,
			float* sortedPres,
			float* sortedForces,
			float* sortedCol,
			float* sortedPosStar,
			float* sortedVelStar,
			float* sortedDensStar,
			float* sortedDensError,
			unsigned int  *gridParticleIndex,
			unsigned int  *cellStart,
			unsigned int  *cellEnd,
			unsigned int   numParticles,
			unsigned int   numCells);

	float maxDensity(float* dDensities, unsigned int numParticles);
	float3 maxVelocity(float* dVelocities, unsigned int numParticles);

    void sortParticles(unsigned int *dGridParticleHash, unsigned int *dGridParticleIndex, unsigned int numParticles);

	/***********
	*  IISPH  *
	***********/
	void predictAdvection(float* sortedPos,
		float                       * sortedVel,
		float                       * sortedDens,
		float                       * sortedPres,
		float                       * sortedForces,
		float                       * sortedCol,
		unsigned int                * cellStart,
		unsigned int                * cellEnd,
		unsigned int                * gridParticleIndex,
		float						* sortedBoundaryPos,
		float						* sortedBoundaryVbi,
		unsigned int                * cellBoundaryStart,
		unsigned int                * cellBoundaryEnd,
		unsigned int                * gridBoundaryIndex,
		float                       * sortedDensAdv,
		float                       * sortedDensCorr,
		float                       * sortedP_l,
		float                       * sortedPreviousP,
		float                       * sortedAii,
		float                       * sortedVelAdv,
		float                       * sortedForcesAdv,
		float                       * sortedForcesP,
		float                       * sortedDiiFluid,
		float                       * sortedDiiBoundary,
		float                       * sortedSumDij,
		float                       * sortedNormal,
		unsigned int numParticles,
		unsigned int numBoundaries,
		unsigned int numCells);


	void pressureSolve(float* sortedPos, float* sortedVel, float* sortedDens, float* sortedPres, float* sortedForces, float* sortedCol, unsigned int* cellStart, unsigned int* cellEnd, unsigned int* gridParticleIndex,
					  unsigned int* cellBoundaryStart, unsigned int* cellBoundaryEnd, unsigned int* gridBoundaryIndex, float* sortedDensAdv, float* sortedDensCorr, float* sortedP_l,  float* sortedPreviousP, 
					  float* sortedAii, float* sortedVelAdv, float* sortedForcesAdv, float* sortedForcesP, float* sortedDiiFluid, float* sortedDiiBoundary, float* sortedSumDij, float* sortedNormal,
					  unsigned int numParticles, unsigned int numBoundaries, unsigned int numCells);
}
#endif /* ifndef SPH_CUH */
