#include <cstdlib>
#include <cstdio>
#include <string.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <helper_cuda.h>
#include <helper_cuda_gl.h>
#include <helper_math.h>

#include <helper_functions.h>
#include <thrust/device_ptr.h>
#include <thrust/for_each.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sort.h>
#include <thrust/extrema.h>
#include <thrust/execution_policy.h>

#include "sph_kernel.cuh"
#include "sph_kernel_impl.cuh"

EXTERN_C_BEGIN

	/********************************
	*  SORT AND THRUST REDUCTIONS  *
	********************************/

//==================================================================================================== 
//==================================================================================================== 
//==================================================================================================== 
SReal maxDensity(SReal* dDensities, unsigned int numParticles)
{
	
	SReal res = *thrust::max_element(thrust::device, 
			thrust::device_ptr<SReal>(dDensities),
			thrust::device_ptr<SReal>(dDensities+numParticles)
			);

	return res;
}

//==================================================================================================== 
//==================================================================================================== 
//==================================================================================================== 
SVec3 maxVelocity(SReal* dVelocities, unsigned int numParticles)
{
	SVec3 res = *thrust::max_element(thrust::device,
			thrust::device_ptr<SVec3>((SVec3*)dVelocities),
			thrust::device_ptr<SVec3>((SVec3*)dVelocities+numParticles),
			comp());
	return res;
}
	
//==================================================================================================== 
//==================================================================================================== 
//==================================================================================================== 
void sortParticles(unsigned int *dGridParticleHash, unsigned int *dGridParticleIndex, unsigned int numParticles)
{
	thrust::sort_by_key(thrust::device_ptr<unsigned int>(dGridParticleHash),
			thrust::device_ptr<unsigned int>(dGridParticleHash + numParticles),
			thrust::device_ptr<unsigned int>(dGridParticleIndex));
}

//==================================================================================================== 
//==================================================================================================== 
//==================================================================================================== 
void cudaInit(int argc, char **argv)
{
	int devID;

	// use command-line specified CUDA device, otherwise use device with highest Gflops/s
	devID = findCudaDevice(argc, (const char **)argv);

	if (devID < 0)
	{
		printf("No CUDA Capable devices found, exiting...\n");
		exit(EXIT_SUCCESS);
	}
}

//==================================================================================================== 
//==================================================================================================== 
//==================================================================================================== 
void cudaGLInit(int argc, char **argv)
{
	// use command-line specified CUDA device, otherwise use device with highest Gflops/s
	findCudaGLDevice(argc, (const char **)argv);
}

//==================================================================================================== 
//==================================================================================================== 
//==================================================================================================== 
void allocateArray(void **devPtr, size_t size)
{
	checkCudaErrors(cudaMalloc(devPtr, size));
}

//==================================================================================================== 
//==================================================================================================== 
//==================================================================================================== 
void freeArray(void *devPtr)
{
	checkCudaErrors(cudaFree(devPtr));
}

//==================================================================================================== 
//==================================================================================================== 
//==================================================================================================== 
void threadSync()
{
	checkCudaErrors(cudaDeviceSynchronize());
}

//==================================================================================================== 
//==================================================================================================== 
//==================================================================================================== 
void copyArrayToDevice(void *device, const void *host, int offset, int size)
{
	checkCudaErrors(cudaMemcpy((char *) device + offset, host, size, cudaMemcpyHostToDevice));
}

//==================================================================================================== 
//==================================================================================================== 
//==================================================================================================== 
void registerGLBufferObject(unsigned int vbo, struct cudaGraphicsResource **cuda_vbo_resource)
{
	checkCudaErrors(cudaGraphicsGLRegisterBuffer(cuda_vbo_resource, vbo,
				cudaGraphicsMapFlagsNone));
}

//==================================================================================================== 
//==================================================================================================== 
//==================================================================================================== 
void unregisterGLBufferObject(struct cudaGraphicsResource *cuda_vbo_resource)
{
	checkCudaErrors(cudaGraphicsUnregisterResource(cuda_vbo_resource));
}

//==================================================================================================== 
//==================================================================================================== 
//==================================================================================================== 
void *mapGLBufferObject(struct cudaGraphicsResource **cuda_vbo_resource)
{
	void *ptr;
	checkCudaErrors(cudaGraphicsMapResources(1, cuda_vbo_resource, 0));
	size_t num_bytes;
	checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&ptr, &num_bytes,
				*cuda_vbo_resource));
	return ptr;
}

//==================================================================================================== 
//==================================================================================================== 
//==================================================================================================== 
void unmapGLBufferObject(struct cudaGraphicsResource *cuda_vbo_resource)
{
	checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_vbo_resource, 0));
}

//==================================================================================================== 
//==================================================================================================== 
//==================================================================================================== 
void copyArrayFromDevice(void *host, const void *device,
		struct cudaGraphicsResource **cuda_vbo_resource, int size)
{
	if (cuda_vbo_resource)
	{
		device = mapGLBufferObject(cuda_vbo_resource);
	}

	checkCudaErrors(cudaMemcpy(host, device, size, cudaMemcpyDeviceToHost));

	if (cuda_vbo_resource)
	{
		unmapGLBufferObject(*cuda_vbo_resource);
	}
}

//==================================================================================================== 
//==================================================================================================== 
//==================================================================================================== 
void setParameters(SphSimParams *hostParams)
{
	// copy parameters to constant memory
	checkCudaErrors(cudaMemcpyToSymbol(sph_params, hostParams, sizeof(SphSimParams)));
}

//==================================================================================================== 
//==================================================================================================== 
//==================================================================================================== 
//Round a / b to nearest higher integer value
unsigned int iDivUp(unsigned int a, unsigned int b)
{
	return (a % b != 0) ? (a / b + 1) : (a / b);
}

//==================================================================================================== 
//==================================================================================================== 
//==================================================================================================== 
// compute grid and thread block size for a given number of elements
void computeGridSize(unsigned int n, unsigned int blockSize, unsigned int &numBlocks, unsigned int &numThreads)
{
	numThreads = min(blockSize, n);
	numBlocks = iDivUp(n, numThreads);
}

//==================================================================================================== 
//==================================================================================================== 
//==================================================================================================== 
void integrateSystem(SReal *pos,
		SReal *vel,
		SReal *forces,
		SReal deltaTime,
		unsigned int numParticles)
{
	thrust::device_ptr<SVec4> d_pos4((SVec4 *)pos);
	thrust::device_ptr<SVec4> d_vel4((SVec4 *)vel);
	thrust::device_ptr<SVec4> d_forces4((SVec4 *)forces);

	thrust::for_each(
			thrust::make_zip_iterator(thrust::make_tuple(d_pos4, d_vel4, d_forces4)),
			thrust::make_zip_iterator(thrust::make_tuple(d_pos4+numParticles, d_vel4+numParticles, d_forces4+numParticles)),
			integrate_functor(deltaTime));
}

//==================================================================================================== 
//==================================================================================================== 
//==================================================================================================== 
void calcHash(unsigned int  *gridParticleHash,
		unsigned int  *gridParticleIndex,
		SReal *pos,
		int    numParticles)
{
	unsigned int numThreads, numBlocks;
	computeGridSize(numParticles, 256, numBlocks, numThreads);

	// execute the kernel
	calcHashD<<< numBlocks, numThreads >>>(gridParticleHash,
			gridParticleIndex,
			(SVec4 *) pos,
			numParticles);

	// check if kernel invocation generated an error
	getLastCudaError("calcHash Failed");
}

//==================================================================================================== 
//==================================================================================================== 
//==================================================================================================== 
/*********************
 *  REORDERING CALL  *
 *********************/
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
										)
{
	unsigned int numThreads, numBlocks;
	computeGridSize(numBoundaries, 64, numBlocks, numThreads);

	checkCudaErrors(cudaMemset(cellStart, 0xffffffff, numCells*sizeof(unsigned int)));

#if USE_TEX
	checkCudaErrors(cudaBindTexture(0, oldBoundaryPosTex, oldPos, numBoundaries*sizeof(SVec4)));
	checkCudaErrors(cudaBindTexture(0, oldBoundaryVbiTex, oldVbi, numBoundaries*sizeof(SReal)));
#endif
	unsigned int smemSize = sizeof(unsigned int)*(numThreads+1);

	reorderDataAndFindCellStartDBoundary<<<numBlocks, numThreads, smemSize>>>(
			cellStart,
			cellEnd,
			(SVec4*) sortedPos,
			(SReal*)  sortedVbi,
			(unsigned int *)gridParticleHash,
			(unsigned int *)gridParticleIndex,
			(SVec4*) oldPos,
			(SReal*)  oldVbi,
			numBoundaries);

	getLastCudaError("Kernel execution failed: reorderDataAndFindCellStartD");
#if USE_TEX
	checkCudaErrors(cudaBindTexture(0, cellBoundaryStartTex, cellStart, numCells*sizeof(unsigned int)));
	checkCudaErrors(cudaBindTexture(0, cellBoundaryEndTex, cellEnd, numCells*sizeof(unsigned int)));
#endif
}

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
		unsigned int   numCells)
{
	unsigned int numThreads, numBlocks;
	computeGridSize(numParticles, 64, numBlocks, numThreads);

	// set all cells to empty
	checkCudaErrors(cudaMemset(cellStart, 0xffffffff, numCells*sizeof(unsigned int)));

#if USE_TEX
	checkCudaErrors(cudaBindTexture(0, oldPosTex, oldPos, numParticles*sizeof(SVec4)));
	checkCudaErrors(cudaBindTexture(0, oldVelTex, oldVel, numParticles*sizeof(SVec4)));
	checkCudaErrors(cudaBindTexture(0, oldDensTex, oldDens, numParticles*sizeof(SReal)));
	checkCudaErrors(cudaBindTexture(0, oldPresTex, oldPres, numParticles*sizeof(SReal)));
	checkCudaErrors(cudaBindTexture(0, oldForcesTex, oldForces, numParticles*sizeof(SVec4)));
	checkCudaErrors(cudaBindTexture(0, oldColTex, oldCol, numParticles*sizeof(SVec4)));
#endif

	unsigned int smemSize = sizeof(unsigned int)*(numThreads+1);
	reorderDataAndFindCellStartD<<< numBlocks, numThreads, smemSize>>>(
			cellStart,
			cellEnd,
			(SVec4 *) sortedPos,
			(SVec4 *) sortedVel,
			NULL,
			sortedPres,
			NULL,
			NULL,
			gridParticleHash,
			gridParticleIndex,
			(SVec4 *) oldPos,
			(SVec4 *) oldVel,
			NULL,
			oldPres,
			NULL,
			NULL,
			numParticles);
	getLastCudaError("Kernel execution failed: reorderDataAndFindCellStartD");

#if USE_TEX
	checkCudaErrors(cudaUnbindTexture(oldPosTex));
	checkCudaErrors(cudaUnbindTexture(oldVelTex));
	checkCudaErrors(cudaUnbindTexture(oldDensTex));
	checkCudaErrors(cudaUnbindTexture(oldPresTex));
	checkCudaErrors(cudaUnbindTexture(oldForcesTex));
	checkCudaErrors(cudaUnbindTexture(oldColTex));
#endif
}

//==================================================================================================== 
//==================================================================================================== 
//==================================================================================================== 
	/******************************
	*  SPH COMPUTATION WITH EOS  *
	******************************/
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
		unsigned int   numBoundaries)
{
#if USE_TEX
	checkCudaErrors(cudaBindTexture(0, oldPosTex, sortedPos, numParticles*sizeof(SVec4)));
	checkCudaErrors(cudaBindTexture(0, oldVelTex, sortedVel, numParticles*sizeof(SVec4)));
	checkCudaErrors(cudaBindTexture(0, oldDensTex, sortedDens, numParticles*sizeof(SReal)));
	checkCudaErrors(cudaBindTexture(0, oldPresTex, sortedPres, numParticles*sizeof(SReal)));
	checkCudaErrors(cudaBindTexture(0, oldForcesTex, sortedForces, numParticles*sizeof(SVec4)));
	checkCudaErrors(cudaBindTexture(0, oldColTex, sortedCol, numParticles*sizeof(SVec4)));

	checkCudaErrors(cudaBindTexture(0, cellStartTex, cellStart, numCells*sizeof(unsigned int)));
	checkCudaErrors(cudaBindTexture(0, cellEndTex, cellEnd, numCells*sizeof(unsigned int)));
#endif

	// thread per particle
	unsigned int numThreads, numBlocks;
	computeGridSize(numParticles, 64, numBlocks, numThreads);

	// execute the kernel
	computeDensityPressure<<<numBlocks, numThreads>>>(
			(SVec4 *)sortedPos,
			(SVec4 *)sortedVel,
			(SReal  *)sortedDens,
			(SReal  *)sortedPres,
			(SVec4 *)sortedForces,
			(SVec4 *)sortedCol,
			(SVec4 *)sortedBoundaryPos,
			(SReal  *)sortedBoundaryVbi,
			gridParticleIndex,    // input: sorted particle indices
			cellStart,
			cellEnd,
			gridBoundaryIndex,
			cellBoundaryStart,
			cellBoundaryEnd,
			numParticles
	);

	/*SReal maxd =  maxDensity(sortedDens, numParticles);*/
	/*printf("maxd = %f\n", maxd);*/
	
	/*cudaDeviceSynchronize();*/

	computeForces<<< numBlocks, numThreads >>>(
		  (SVec4*) sortedPos,               // input: sorted positions
		  (SVec4*) sortedVel,               // input: sorted velocities
		  (SReal*) sortedDens,               // input: sorted velocities
		  (SReal*) sortedPres,               // input: sorted velocities
		  (SVec4*) sortedForces,            // input: sorted velocities
		  (SVec4*) sortedCol,               // input: sorted velocities
		  gridBoundaryIndex,
		  (SVec4*) sortedBoundaryPos,
		  (SReal*) sortedBoundaryVbi,
		  gridParticleIndex,    // input: sorted particle indices
		  cellStart,
		  cellEnd,
		  cellBoundaryStart,
		  cellBoundaryEnd,
		  numParticles);


	// check if kernel invocation generated an error
	getLastCudaError("Kernel execution failed");

#if USE_TEX
	checkCudaErrors(cudaUnbindTexture(oldPosTex));
	checkCudaErrors(cudaUnbindTexture(oldVelTex));
	checkCudaErrors(cudaUnbindTexture(oldDensTex));
	checkCudaErrors(cudaUnbindTexture(oldPresTex));
	checkCudaErrors(cudaUnbindTexture(oldForcesTex));
	checkCudaErrors(cudaUnbindTexture(oldColTex));
	checkCudaErrors(cudaUnbindTexture(cellStartTex));
	checkCudaErrors(cudaUnbindTexture(cellEndTex));
#endif
}

//==================================================================================================== 
//==================================================================================================== 
//==================================================================================================== 
/***********
*  IISPH  *
***********/
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
	unsigned int numCells)
{
#if USE_TEX
	checkCudaErrors(cudaBindTexture(0, oldPosTex, sortedPos, numParticles*sizeof(SVec4)));
	checkCudaErrors(cudaBindTexture(0, oldVelTex, sortedVel, numParticles*sizeof(SVec4)));
	checkCudaErrors(cudaBindTexture(0, oldDensTex, sortedDens, numParticles*sizeof(SReal)));
	checkCudaErrors(cudaBindTexture(0, oldPresTex, sortedPres, numParticles*sizeof(SReal)));
	checkCudaErrors(cudaBindTexture(0, oldForcesTex, sortedForces, numParticles*sizeof(SVec4)));
	checkCudaErrors(cudaBindTexture(0, oldColTex, sortedCol, numParticles*sizeof(SVec4)));

	checkCudaErrors(cudaBindTexture(0, cellStartTex, cellStart, numCells*sizeof(unsigned int)));
	checkCudaErrors(cudaBindTexture(0, cellEndTex, cellEnd, numCells*sizeof(unsigned int)));

	checkCudaErrors(cudaBindTexture(0, oldDensAdvTex, sortedDensAdv, numParticles*sizeof(SReal)));
	checkCudaErrors(cudaBindTexture(0, oldDensCorrTex, sortedDensCorr, numParticles*sizeof(SReal)));
	checkCudaErrors(cudaBindTexture(0, oldP_lTex, sortedP_l, numParticles*sizeof(SReal)));
	checkCudaErrors(cudaBindTexture(0, oldPreviousPTex, sortedPreviousP, numParticles*sizeof(SReal)));
	checkCudaErrors(cudaBindTexture(0, oldAiiTex, sortedAii, numParticles*sizeof(SReal)));

	checkCudaErrors(cudaBindTexture(0, oldVelAdvTex, sortedVelAdv, numParticles*sizeof(SVec4)));
	checkCudaErrors(cudaBindTexture(0, oldForcesAdvTex, sortedForcesAdv, numParticles*sizeof(SVec4)));
	checkCudaErrors(cudaBindTexture(0, oldForcesPTex, sortedForcesP, numParticles*sizeof(SVec4)));
	checkCudaErrors(cudaBindTexture(0, oldDiiFluidTex, sortedDiiFluid, numParticles*sizeof(SVec4)));
	checkCudaErrors(cudaBindTexture(0, oldDiiBoundaryTex, sortedDiiBoundary, numParticles*sizeof(SVec4)));
	checkCudaErrors(cudaBindTexture(0, oldSumDijTex, sortedSumDij, numParticles*sizeof(SVec4)));
	checkCudaErrors(cudaBindTexture(0, oldNormalTex, sortedNormal, numParticles*sizeof(SVec4)));
#endif

	unsigned int numThreads, numBlocks;
	computeGridSize(numParticles, 64, numBlocks, numThreads);

	computeIisphDensity<<<numBlocks, numThreads>>>(
			(SVec4*) sortedPos,
			(SVec4*) sortedVel,
			sortedDens,
			sortedPres,
			(SVec4*) sortedForces,
			(SVec4*) sortedCol,
			cellStart,
			cellEnd,
			gridParticleIndex,
			(SVec4*) sortedBoundaryPos,
			sortedBoundaryVbi,
			cellBoundaryStart,
			cellBoundaryEnd,
			gridBoundaryIndex,
			sortedDensAdv,
			sortedDensCorr,
			sortedP_l,
			sortedPreviousP,
			sortedAii,
			(SVec4*) sortedVelAdv,
			(SVec4*) sortedForcesAdv,
			(SVec4*) sortedForcesP,
			(SVec4*) sortedDiiFluid,
			(SVec4*) sortedDiiBoundary,
			(SVec4*) sortedSumDij,
			(SVec4*) sortedNormal,
			numParticles,
			numBoundaries,
			numCells);


	/*cudaDeviceSynchronize();*/

	computeDisplacementFactor<<<numBlocks, numThreads>>>(
			(SVec4*) sortedPos,
			(SVec4*) sortedVel,
			sortedDens,
			sortedPres,
			(SVec4*) sortedForces,
			(SVec4*) sortedCol,
			cellStart,
			cellEnd,
			gridParticleIndex,
			(SVec4*) sortedBoundaryPos,
			sortedBoundaryVbi,
			cellBoundaryStart,
			cellBoundaryEnd,
			gridBoundaryIndex,
			sortedDensAdv,
			sortedDensCorr,
			sortedP_l,
			sortedPreviousP,
			sortedAii,
			(SVec4*) sortedVelAdv,
			(SVec4*) sortedForcesAdv,
			(SVec4*) sortedForcesP,
			(SVec4*) sortedDiiFluid,
			(SVec4*) sortedDiiBoundary,
			(SVec4*) sortedSumDij,
			(SVec4*) sortedNormal,
			numParticles,
			numBoundaries,
			numCells);

	/*cudaDeviceSynchronize();*/

	computeAdvectionFactor<<<numBlocks, numThreads>>>(
			(SVec4*) sortedPos,
			(SVec4*) sortedVel,
			sortedDens,
			sortedPres,
			(SVec4*) sortedForces,
			(SVec4*) sortedCol,
			cellStart,
			cellEnd,
			gridParticleIndex,
			(SVec4*) sortedBoundaryPos,
			sortedBoundaryVbi,
			cellBoundaryStart,
			cellBoundaryEnd,
			gridBoundaryIndex,
			sortedDensAdv,
			sortedDensCorr,
			sortedP_l,
			sortedPreviousP,
			sortedAii,
			(SVec4*) sortedVelAdv,
			(SVec4*) sortedForcesAdv,
			(SVec4*) sortedForcesP,
			(SVec4*) sortedDiiFluid,
			(SVec4*) sortedDiiBoundary,
			(SVec4*) sortedSumDij,
			(SVec4*) sortedNormal,
			numParticles,
			numBoundaries,
			numCells);

	/*cudaDeviceSynchronize();*/
#if USE_TEX
	checkCudaErrors(cudaUnbindTexture(oldPosTex));
	checkCudaErrors(cudaUnbindTexture(oldVelTex));
	checkCudaErrors(cudaUnbindTexture(oldDensTex));
	checkCudaErrors(cudaUnbindTexture(oldPresTex));
	checkCudaErrors(cudaUnbindTexture(oldForcesTex));
	checkCudaErrors(cudaUnbindTexture(oldColTex));

	checkCudaErrors(cudaUnbindTexture(cellStartTex));
	checkCudaErrors(cudaUnbindTexture(cellEndTex));

	checkCudaErrors(cudaUnbindTexture(oldDensAdvTex));
	checkCudaErrors(cudaUnbindTexture(oldDensCorrTex));
	checkCudaErrors(cudaUnbindTexture(oldP_lTex));
	checkCudaErrors(cudaUnbindTexture(oldPreviousPTex));
	checkCudaErrors(cudaUnbindTexture(oldAiiTex));

	checkCudaErrors(cudaUnbindTexture(oldVelAdvTex));
	checkCudaErrors(cudaUnbindTexture(oldForcesAdvTex));
	checkCudaErrors(cudaUnbindTexture(oldForcesPTex));
	checkCudaErrors(cudaUnbindTexture(oldDiiFluidTex));
	checkCudaErrors(cudaUnbindTexture(oldDiiBoundaryTex));
	checkCudaErrors(cudaUnbindTexture(oldSumDijTex));
	checkCudaErrors(cudaUnbindTexture(oldNormalTex));
#endif

}

//==================================================================================================== 
//==================================================================================================== 
//==================================================================================================== 
void pressureSolve(SReal* sortedPos, SReal* sortedVel, SReal* sortedDens, SReal* sortedPres, SReal* sortedForces, SReal* sortedCol, unsigned int* cellStart, unsigned int* cellEnd, unsigned int* gridParticleIndex,
					  SReal* sortedBoundaryPos, SReal* sortedBoundaryVbi,
					  unsigned int* cellBoundaryStart, unsigned int* cellBoundaryEnd, unsigned int* gridBoundaryIndex, SReal* sortedDensAdv, SReal* sortedDensCorr, SReal* sortedP_l,  SReal* sortedPreviousP, 
					  SReal* sortedAii, SReal* sortedVelAdv, SReal* sortedForcesAdv, SReal* sortedForcesP, SReal* sortedDiiFluid, SReal* sortedDiiBoundary, SReal* sortedSumDij, SReal* sortedNormal,
					  unsigned int numParticles, unsigned int numBoundaries, unsigned int numCells)
{
#if USE_TEX
	checkCudaErrors(cudaBindTexture(0, oldPosTex, sortedPos, numParticles*sizeof(SVec4)));
	checkCudaErrors(cudaBindTexture(0, oldVelTex, sortedVel, numParticles*sizeof(SVec4)));
	checkCudaErrors(cudaBindTexture(0, oldDensTex, sortedDens, numParticles*sizeof(SReal)));
	checkCudaErrors(cudaBindTexture(0, oldPresTex, sortedPres, numParticles*sizeof(SReal)));
	checkCudaErrors(cudaBindTexture(0, oldForcesTex, sortedForces, numParticles*sizeof(SVec4)));
	checkCudaErrors(cudaBindTexture(0, oldColTex, sortedCol, numParticles*sizeof(SVec4)));

	checkCudaErrors(cudaBindTexture(0, cellStartTex, cellStart, numCells*sizeof(unsigned int)));
	checkCudaErrors(cudaBindTexture(0, cellEndTex, cellEnd, numCells*sizeof(unsigned int)));

	checkCudaErrors(cudaBindTexture(0, oldDensAdvTex, sortedDensAdv, numParticles*sizeof(SReal)));
	checkCudaErrors(cudaBindTexture(0, oldDensCorrTex, sortedDensCorr, numParticles*sizeof(SReal)));
	checkCudaErrors(cudaBindTexture(0, oldP_lTex, sortedP_l, numParticles*sizeof(SReal)));
	checkCudaErrors(cudaBindTexture(0, oldPreviousPTex, sortedPreviousP, numParticles*sizeof(SReal)));
	checkCudaErrors(cudaBindTexture(0, oldAiiTex, sortedAii, numParticles*sizeof(SReal)));

	checkCudaErrors(cudaBindTexture(0, oldVelAdvTex, sortedVelAdv, numParticles*sizeof(SVec4)));
	checkCudaErrors(cudaBindTexture(0, oldForcesAdvTex, sortedForcesAdv, numParticles*sizeof(SVec4)));
	checkCudaErrors(cudaBindTexture(0, oldForcesPTex, sortedForcesP, numParticles*sizeof(SVec4)));
	checkCudaErrors(cudaBindTexture(0, oldDiiFluidTex, sortedDiiFluid, numParticles*sizeof(SVec4)));
	checkCudaErrors(cudaBindTexture(0, oldDiiBoundaryTex, sortedDiiBoundary, numParticles*sizeof(SVec4)));
	checkCudaErrors(cudaBindTexture(0, oldSumDijTex, sortedSumDij, numParticles*sizeof(SVec4)));
	checkCudaErrors(cudaBindTexture(0, oldNormalTex, sortedNormal, numParticles*sizeof(SVec4)));
#endif
	unsigned int numThreads, numBlocks;
	computeGridSize(numParticles, 64, numBlocks, numThreads);

	unsigned int l=0; 
	SReal rho_avg = 0.f;
	const SReal rd = 1000.f;
	const SReal max_rho_err = 1.f;

	while( ((rho_avg - rd) > max_rho_err) || (l<2))
	{
		//compute sumdijpj
		computeSumDijPj<<<numBlocks, numThreads>>>(
				(SVec4                      *) sortedPos,
				(SVec4                      *) sortedVel,
				sortedDens,
				sortedPres,
				(SVec4                      *) sortedForces,
				(SVec4                      *) sortedCol,
				cellStart,
				cellEnd,
				gridParticleIndex,
				(SVec4					    *)sortedBoundaryPos,
				sortedBoundaryVbi,
				cellBoundaryStart,
				cellBoundaryEnd,
				gridBoundaryIndex,
				sortedDensAdv,
				sortedDensCorr,
				sortedP_l,
				sortedPreviousP,
				sortedAii,
				(SVec4                      *) sortedVelAdv,
				(SVec4                      *) sortedForcesAdv,
				(SVec4                      *) sortedForcesP,
				(SVec4                      *) sortedDiiFluid,
				(SVec4                      *) sortedDiiBoundary,
				(SVec4                      *) sortedSumDij,
				(SVec4                      *) sortedNormal,
				numParticles,
				numBoundaries,
				numCells
		);

		/*SReal maxd =  maxDensity(sortedDensCorr, numParticles);*/
		/*printf("maxd = %f\n", maxd);*/


		/*cudaDeviceSynchronize();*/
		//compute pressure
		computePressure<<<numBlocks, numThreads>>>(
				(SVec4                      *) sortedPos,
				(SVec4                      *) sortedVel,
				sortedDens,
				sortedPres,
				(SVec4                      *) sortedForces,
				(SVec4                      *) sortedCol,
				cellStart,
				cellEnd,
				gridParticleIndex,
				(SVec4					    *)sortedBoundaryPos,
				sortedBoundaryVbi,
				cellBoundaryStart,
				cellBoundaryEnd,
				gridBoundaryIndex,
				sortedDensAdv,
				sortedDensCorr,
				sortedP_l,
				sortedPreviousP,
				sortedAii,
				(SVec4                      *) sortedVelAdv,
				(SVec4                      *) sortedForcesAdv,
				(SVec4                      *) sortedForcesP,
				(SVec4                      *) sortedDiiFluid,
				(SVec4                      *) sortedDiiBoundary,
				(SVec4                      *) sortedSumDij,
				(SVec4                      *) sortedNormal,
				numParticles,
				numBoundaries,
				numCells
		);

		/*cudaDeviceSynchronize();*/

		//reduce rho_error buffers
		rho_avg = 0.f;
		rho_avg = thrust::reduce(thrust::device_ptr<SReal>(sortedDensCorr),thrust::device_ptr<SReal>(sortedDensCorr+numParticles));
		rho_avg /= numParticles;


		l++;
	}

	/*printf("l = %d\n", l );*/

	computePressureForce<<<numBlocks, numThreads>>>(
				(SVec4                      *) sortedPos,
				(SVec4                      *) sortedVel,
				sortedDens,
				sortedPres,
				(SVec4                      *) sortedForces,
				(SVec4                      *) sortedCol,
				cellStart,
				cellEnd,
				gridParticleIndex,
				(SVec4					    *)sortedBoundaryPos,
				sortedBoundaryVbi,
				cellBoundaryStart,
				cellBoundaryEnd,
				gridBoundaryIndex,
				sortedDensAdv,
				sortedDensCorr,
				sortedP_l,
				sortedPreviousP,
				sortedAii,
				(SVec4                      *) sortedVelAdv,
				(SVec4                      *) sortedForcesAdv,
				(SVec4                      *) sortedForcesP,
				(SVec4                      *) sortedDiiFluid,
				(SVec4                      *) sortedDiiBoundary,
				(SVec4                      *) sortedSumDij,
				(SVec4                      *) sortedNormal,
				numParticles,
				numBoundaries,
				numCells
		);

	/*cudaDeviceSynchronize();*/
	iisph_integrate<<<numBlocks, numThreads>>>(
			(SVec4*) sortedPos,
			(SVec4*) sortedVel,
			(SVec4*) sortedVelAdv,
			(SVec4*) sortedForcesP,
			gridParticleIndex,
			numParticles
			);

	/*cudaDeviceSynchronize();*/

	/*exit(0);*/

#if USE_TEX
	checkCudaErrors(cudaUnbindTexture(oldPosTex));
	checkCudaErrors(cudaUnbindTexture(oldVelTex));
	checkCudaErrors(cudaUnbindTexture(oldDensTex));
	checkCudaErrors(cudaUnbindTexture(oldPresTex));
	checkCudaErrors(cudaUnbindTexture(oldForcesTex));
	checkCudaErrors(cudaUnbindTexture(oldColTex));

	checkCudaErrors(cudaUnbindTexture(cellStartTex));
	checkCudaErrors(cudaUnbindTexture(cellEndTex));

	checkCudaErrors(cudaUnbindTexture(oldDensAdvTex));
	checkCudaErrors(cudaUnbindTexture(oldDensCorrTex));
	checkCudaErrors(cudaUnbindTexture(oldP_lTex));
	checkCudaErrors(cudaUnbindTexture(oldPreviousPTex));
	checkCudaErrors(cudaUnbindTexture(oldAiiTex));

	checkCudaErrors(cudaUnbindTexture(oldVelAdvTex));
	checkCudaErrors(cudaUnbindTexture(oldForcesAdvTex));
	checkCudaErrors(cudaUnbindTexture(oldForcesPTex));
	checkCudaErrors(cudaUnbindTexture(oldDiiFluidTex));
	checkCudaErrors(cudaUnbindTexture(oldDiiBoundaryTex));
	checkCudaErrors(cudaUnbindTexture(oldSumDijTex));
	checkCudaErrors(cudaUnbindTexture(oldNormalTex));
#endif

}
/************
*  PCISPH  *
************/
//==================================================================================================== 
//==================================================================================================== 
//==================================================================================================== 
void pcisph_internalForces(SReal* sortedPos, SReal* sortedVel, SReal* sortedDens, SReal* sortedPres, SReal* sortedForces, SReal* sortedCol, unsigned int* cellStart, unsigned int* cellEnd, unsigned int* gridParticleIndex,
				SReal* sortedBoundaryPos, SReal* sortedBoundaryVbi, unsigned int* cellBoundaryStart, unsigned int* cellBoundaryEnd, unsigned int* gridBoundaryIndex, SReal* sortedRhoAdv, SReal* sortedPosAdv, SReal* sortedVelAdv, 
				SReal* sortedForcesAdv, SReal* sortedForcesP, SReal* sortedNormal, unsigned int numParticles, unsigned int numBoundaries, unsigned int numCells)
{
#if USE_TEX
	//add texture management someday
#endif

	// thread per particle
	unsigned int numThreads, numBlocks;
	computeGridSize(numParticles, 64, numBlocks, numThreads);

	computeDensityPressure<<<numBlocks, numThreads>>>(
			(SVec4 *)sortedPos,
			(SVec4 *)sortedVel,
			(SReal *)sortedDens,
			(SReal *)sortedPres,
			(SVec4 *)sortedForces,
			(SVec4 *)sortedCol,
			(SVec4 *)sortedBoundaryPos,
			(SReal *)sortedBoundaryVbi,
			gridParticleIndex,    // input: sorted particle indices
			cellStart,
			cellEnd,
			gridBoundaryIndex,
			cellBoundaryStart,
			cellBoundaryEnd,
			numParticles
	);


#if USE_TEX
	//add texture management someday
#endif
}
//==================================================================================================== 
//==================================================================================================== 
//==================================================================================================== 
void pcisph_pressureSolve(SReal* sortedPos, SReal* sortedVel, SReal* sortedDens, SReal* sortedPres, SReal* sortedForces, SReal* sortedCol, unsigned int* cellStart, unsigned int* cellEnd, unsigned int* gridParticleIndex,
				SReal* sortedBoundaryPos, SReal* sortedBoundaryVbi, unsigned int* cellBoundaryStart, unsigned int* cellBoundaryEnd, unsigned int* gridBoundaryIndex, SReal* sortedRhoAdv, SReal* sortedPosAdv, SReal* sortedVelAdv, 
				SReal* sortedForcesAdv, SReal* sortedForcesP, SReal* sortedNormal, unsigned int numParticles, unsigned int numBoundaries, unsigned int numCells)
{
#if USE_TEX
	//add texture management someday
#endif
	/*printf("pcisph pressure solve\n");*/
}
//==================================================================================================== 
//==================================================================================================== 
//==================================================================================================== 
EXTERN_C_END
