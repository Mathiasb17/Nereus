
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

#include "sph_kernel.cuh"
#include "sph_kernel_impl.cuh"

extern "C"
{
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

	void cudaGLInit(int argc, char **argv)
	{
		// use command-line specified CUDA device, otherwise use device with highest Gflops/s
		findCudaGLDevice(argc, (const char **)argv);
	}

	void allocateArray(void **devPtr, size_t size)
	{
		checkCudaErrors(cudaMalloc(devPtr, size));
	}

	void freeArray(void *devPtr)
	{
		checkCudaErrors(cudaFree(devPtr));
	}

	void threadSync()
	{
		checkCudaErrors(cudaDeviceSynchronize());
	}

	void copyArrayToDevice(void *device, const void *host, int offset, int size)
	{
		checkCudaErrors(cudaMemcpy((char *) device + offset, host, size, cudaMemcpyHostToDevice));
	}

	void registerGLBufferObject(uint vbo, struct cudaGraphicsResource **cuda_vbo_resource)
	{
		checkCudaErrors(cudaGraphicsGLRegisterBuffer(cuda_vbo_resource, vbo,
					cudaGraphicsMapFlagsNone));
	}

	void unregisterGLBufferObject(struct cudaGraphicsResource *cuda_vbo_resource)
	{
		checkCudaErrors(cudaGraphicsUnregisterResource(cuda_vbo_resource));
	}

	void *mapGLBufferObject(struct cudaGraphicsResource **cuda_vbo_resource)
	{
		void *ptr;
		checkCudaErrors(cudaGraphicsMapResources(1, cuda_vbo_resource, 0));
		size_t num_bytes;
		checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&ptr, &num_bytes,
					*cuda_vbo_resource));
		return ptr;
	}

	void unmapGLBufferObject(struct cudaGraphicsResource *cuda_vbo_resource)
	{
		checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_vbo_resource, 0));
	}

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

	void setParameters(SphSimParams *hostParams)
	{
		// copy parameters to constant memory
		checkCudaErrors(cudaMemcpyToSymbol(sph_params, hostParams, sizeof(SphSimParams)));
	}

	//Round a / b to nearest higher integer value
	uint iDivUp(uint a, uint b)
	{
		return (a % b != 0) ? (a / b + 1) : (a / b);
	}

	// compute grid and thread block size for a given number of elements
	void computeGridSize(uint n, uint blockSize, uint &numBlocks, uint &numThreads)
	{
		numThreads = min(blockSize, n);
		numBlocks = iDivUp(n, numThreads);
	}

	void integrateSystem(float *pos,
			float *vel,
			float deltaTime,
			uint numParticles)
	{
		thrust::device_ptr<float4> d_pos4((float4 *)pos);
		thrust::device_ptr<float4> d_vel4((float4 *)vel);

		thrust::for_each(
				thrust::make_zip_iterator(thrust::make_tuple(d_pos4, d_vel4)),
				thrust::make_zip_iterator(thrust::make_tuple(d_pos4+numParticles, d_vel4+numParticles)),
				integrate_functor(deltaTime));
	}

	void calcHash(uint  *gridParticleHash,
			uint  *gridParticleIndex,
			float *pos,
			int    numParticles)
	{
		uint numThreads, numBlocks;
		computeGridSize(numParticles, 256, numBlocks, numThreads);

		// execute the kernel
		calcHashD<<< numBlocks, numThreads >>>(gridParticleHash,
				gridParticleIndex,
				(float4 *) pos,
				numParticles);

		// check if kernel invocation generated an error
		getLastCudaError("Kernel execution failed");
	}

	void reorderDataAndFindCellStart(uint  *cellStart,
			uint  *cellEnd,
			float *sortedPos,
			float *sortedVel,
			float *sortedDens,
			float *sortedPres,
			float *sortedForces,
			float *sortedCol,
			uint  *gridParticleHash,
			uint  *gridParticleIndex,
			float *oldPos,
			float *oldVel,
			float *oldDens,
			float *oldPres,
			float *oldForces,
			float *oldCol,
			uint   numParticles,
			uint   numCells)
	{
		uint numThreads, numBlocks;
		computeGridSize(numParticles, 256, numBlocks, numThreads);

		// set all cells to empty
		checkCudaErrors(cudaMemset(cellStart, 0xffffffff, numCells*sizeof(uint)));

#if USE_TEX
		checkCudaErrors(cudaBindTexture(0, oldPosTex, oldPos, numParticles*sizeof(float4)));
		checkCudaErrors(cudaBindTexture(0, oldVelTex, oldVel, numParticles*sizeof(float4)));
		checkCudaErrors(cudaBindTexture(0, oldDensTex, oldDens, numParticles*sizeof(float)));
		checkCudaErrors(cudaBindTexture(0, oldPresTex, oldPres, numParticles*sizeof(float)));
		checkCudaErrors(cudaBindTexture(0, oldForcesTex, oldForces, numParticles*sizeof(float4)));
		checkCudaErrors(cudaBindTexture(0, oldColTex, oldCol, numParticles*sizeof(float4)));
#endif

		uint smemSize = sizeof(uint)*(numThreads+1);
		reorderDataAndFindCellStartD<<< numBlocks, numThreads, smemSize>>>(
				cellStart,
				cellEnd,
				(float4 *) sortedPos,
				(float4 *) sortedVel,
				(float *) sortedDens,
				(float *) sortedPres,
				(float4 *) sortedForces,
				(float4 *) sortedCol,
				gridParticleHash,
				gridParticleIndex,
				(float4 *) oldPos,
				(float4 *) oldVel,
				(float *) oldDens,
				(float *) oldPres,
				(float4 *) oldForces,
				(float4 *) oldCol,
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

	void computeDensityPressure(float *newDens, float* newPres,
			float *sortedPos,
			float *sortedVel,
			float *sortedDens,
			float *sortedPres,
			float *sortedForces,
			float *sortedCol,
			uint  *gridParticleIndex,
			uint  *cellStart,
			uint  *cellEnd,
			uint   numParticles,
			uint   numCells)
	{
#if USE_TEX
		checkCudaErrors(cudaBindTexture(0, oldPosTex, sortedPos, numParticles*sizeof(float4)));
		checkCudaErrors(cudaBindTexture(0, oldVelTex, sortedVel, numParticles*sizeof(float4)));
		checkCudaErrors(cudaBindTexture(0, oldDensTex, sortedVel, numParticles*sizeof(float)));
		checkCudaErrors(cudaBindTexture(0, oldPresTex, sortedVel, numParticles*sizeof(float)));
		checkCudaErrors(cudaBindTexture(0, oldForcesTex, sortedVel, numParticles*sizeof(float4)));
		checkCudaErrors(cudaBindTexture(0, oldColTex, sortedVel, numParticles*sizeof(float4)));
		checkCudaErrors(cudaBindTexture(0, cellStartTex, cellStart, numCells*sizeof(uint)));
		checkCudaErrors(cudaBindTexture(0, cellEndTex, cellEnd, numCells*sizeof(uint)));
#endif

		// thread per particle
		uint numThreads, numBlocks;
		computeGridSize(numParticles, 64, numBlocks, numThreads);

		// execute the kernel
		
		computeDensityPressure<<< numBlocks, numThreads >>>((float*)newDens, (float*)newPres,
				(float4 *)sortedPos,
				(float4 *)sortedVel,
				(float *)sortedDens,
				(float *)sortedPres,
				(float4 *)sortedForces,
				(float4 *)sortedCol,
				gridParticleIndex,
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
#endif
	}


	void sortParticles(uint *dGridParticleHash, uint *dGridParticleIndex, uint numParticles)
	{
		thrust::sort_by_key(thrust::device_ptr<uint>(dGridParticleHash),
				thrust::device_ptr<uint>(dGridParticleHash + numParticles),
				thrust::device_ptr<uint>(dGridParticleIndex));
	}

}
