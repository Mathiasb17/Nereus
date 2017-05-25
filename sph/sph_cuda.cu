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

extern "C"
{

	/********************************
	*  SORT AND THRUST REDUCTIONS  *
	********************************/

//==================================================================================================== 
//==================================================================================================== 
//==================================================================================================== 
	float maxDensity(float* dDensities, unsigned int numParticles)
	{
		
		float res = *thrust::max_element(thrust::device, 
				thrust::device_ptr<float>(dDensities),
				thrust::device_ptr<float>(dDensities+numParticles)
				);

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
	void integrateSystem(float *pos,
			float *vel,
			float *forces,
			float deltaTime,
			unsigned int numParticles)
	{
		thrust::device_ptr<float4> d_pos4((float4 *)pos);
		thrust::device_ptr<float4> d_vel4((float4 *)vel);
		thrust::device_ptr<float4> d_forces4((float4 *)forces);

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
			float *pos,
			int    numParticles)
	{
		unsigned int numThreads, numBlocks;
		computeGridSize(numParticles, 256, numBlocks, numThreads);

		// execute the kernel
		calcHashD<<< numBlocks, numThreads >>>(gridParticleHash,
				gridParticleIndex,
				(float4 *) pos,
				numParticles);

		// check if kernel invocation generated an error
		getLastCudaError("Kernel execution failed");
	}

//==================================================================================================== 
//==================================================================================================== 
//==================================================================================================== 

	/*********************
	*  REORDERING CALL  *
	*********************/

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
											)
	{
		unsigned int numThreads, numBlocks;
		computeGridSize(numBoundaries, 64, numBlocks, numThreads);

		checkCudaErrors(cudaMemset(cellStart, 0xffffffff, numCells*sizeof(unsigned int)));

#if USE_TEX
		checkCudaErrors(cudaBindTexture(0, oldBoundaryPosTex, oldPos, numBoundaries*sizeof(float4)));
		checkCudaErrors(cudaBindTexture(0, oldBoundaryVbiTex, oldVbi, numBoundaries*sizeof(float)));

#endif
		unsigned int smemSize = sizeof(unsigned int)*(numThreads+1);

		reorderDataAndFindCellStartDBoundary<<<numBlocks, numThreads, smemSize>>>(
				cellStart,
				cellEnd,
				(float4*) sortedPos,
				(float*)  sortedVbi,
				(unsigned int *)gridParticleHash,
				(unsigned int *)gridParticleIndex,
				(float4*) oldPos,
				(float*)  oldVbi,
				numBoundaries);

		getLastCudaError("Kernel execution failed: reorderDataAndFindCellStartD");


#if USE_TEX
		checkCudaErrors(cudaBindTexture(0, cellBoundaryStartTex, cellStart, numCells*sizeof(unsigned int)));
		checkCudaErrors(cudaBindTexture(0, cellBoundaryEndTex, cellEnd, numCells*sizeof(unsigned int)));
#endif
	}

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
			unsigned int   numCells)
	{
		unsigned int numThreads, numBlocks;
		computeGridSize(numParticles, 64, numBlocks, numThreads);

		// set all cells to empty
		checkCudaErrors(cudaMemset(cellStart, 0xffffffff, numCells*sizeof(unsigned int)));

#if USE_TEX
		checkCudaErrors(cudaBindTexture(0, oldPosTex, oldPos, numParticles*sizeof(float4)));
		checkCudaErrors(cudaBindTexture(0, oldVelTex, oldVel, numParticles*sizeof(float4)));
		checkCudaErrors(cudaBindTexture(0, oldDensTex, oldDens, numParticles*sizeof(float)));
		checkCudaErrors(cudaBindTexture(0, oldPresTex, oldPres, numParticles*sizeof(float)));
		checkCudaErrors(cudaBindTexture(0, oldForcesTex, oldForces, numParticles*sizeof(float4)));
		checkCudaErrors(cudaBindTexture(0, oldColTex, oldCol, numParticles*sizeof(float4)));
#endif

		unsigned int smemSize = sizeof(unsigned int)*(numThreads+1);
		reorderDataAndFindCellStartD<<< numBlocks, numThreads, smemSize>>>(
				cellStart,
				cellEnd,
				(float4 *) sortedPos,
				(float4 *) sortedVel,
				NULL,
				NULL,
				NULL,
				NULL,
				gridParticleHash,
				gridParticleIndex,
				(float4 *) oldPos,
				(float4 *) oldVel,
				NULL,
				NULL,
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
			unsigned int   numBoundaries)
	{
#if USE_TEX
		checkCudaErrors(cudaBindTexture(0, oldPosTex, sortedPos, numParticles*sizeof(float4)));
		checkCudaErrors(cudaBindTexture(0, oldVelTex, sortedVel, numParticles*sizeof(float4)));
		checkCudaErrors(cudaBindTexture(0, oldDensTex, sortedDens, numParticles*sizeof(float)));
		checkCudaErrors(cudaBindTexture(0, oldPresTex, sortedPres, numParticles*sizeof(float)));
		checkCudaErrors(cudaBindTexture(0, oldForcesTex, sortedForces, numParticles*sizeof(float4)));
		checkCudaErrors(cudaBindTexture(0, oldColTex, sortedCol, numParticles*sizeof(float4)));

		checkCudaErrors(cudaBindTexture(0, cellStartTex, cellStart, numCells*sizeof(unsigned int)));
		checkCudaErrors(cudaBindTexture(0, cellEndTex, cellEnd, numCells*sizeof(unsigned int)));

		/*checkCudaErrors(cudaBindTexture(0, oldBoundaryPosTex, sortedBoundaryPos, numBoundaries*sizeof(float4)));*/
		/*checkCudaErrors(cudaBindTexture(0, oldBoundaryVbiTex, sortedBoundaryVbi, numBoundaries*sizeof(float)));*/

		//TODO missing cudamalloc in updateBoundary
		/*checkCudaErrors(cudaBindTexture(0, cellBoundaryStartTex, cellBoundaryStart, numCells*sizeof(unsigned int)));*/
		/*checkCudaErrors(cudaBindTexture(0, cellBoundaryEndTex, cellBoundaryEnd, numCells*sizeof(unsigned int)));*/
#endif

		// thread per particle
		unsigned int numThreads, numBlocks;
		computeGridSize(numParticles, 64, numBlocks, numThreads);

		// execute the kernel
		computeDensityPressure<<<numBlocks, numThreads>>>(
				(float4 *)sortedPos,
				(float4 *)sortedVel,
				(float  *)sortedDens,
				(float  *)sortedPres,
				(float4 *)sortedForces,
				(float4 *)sortedCol,
				(float4 *)sortedBoundaryPos,
				(float  *)sortedBoundaryVbi,
				gridParticleIndex,    // input: sorted particle indices
				cellStart,
				cellEnd,
				gridBoundaryIndex,
				cellBoundaryStart,
				cellBoundaryEnd,
				numParticles
		);
		
		cudaDeviceSynchronize();

		computeForces<<< numBlocks, numThreads >>>(
			  (float4*) sortedPos,               // input: sorted positions
			  (float4*) sortedVel,               // input: sorted velocities
			  (float*) sortedDens,               // input: sorted velocities
			  (float*) sortedPres,               // input: sorted velocities
			  (float4*) sortedForces,            // input: sorted velocities
			  (float4*) sortedCol,               // input: sorted velocities
			  gridParticleIndex,    // input: sorted particle indices
			  cellStart,
			  cellEnd,
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

		/*checkCudaErrors(cudaUnbindTexture(oldBoundaryPosTex));*/
		/*checkCudaErrors(cudaUnbindTexture(oldBoundaryVbiTex));*/

		//TODO
		/*checkCudaErrors(cudaUnbindTexture(cellBoundaryStartTex));*/
		/*checkCudaErrors(cudaUnbindTexture(cellBoundaryEndTex));*/
#endif
	}
}
