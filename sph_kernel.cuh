#ifndef SPH_KERNEL_H
#define SPH_KERNEL_H

#include <cuda_runtime.h>

#ifndef GLM_SWIZZLE
#define GLM_SWIZZLE
#endif /* ifndef GLM_SWIZZLE */
#define USE_TEX 1
#include <glm/glm.hpp>

#if USE_TEX
#define FETCH(t, i) tex1Dfetch(t##Tex, i)
#else
#define FETCH(t, i) t[i]
#endif

struct SphSimParams
{
    uint3 gridSize;
    unsigned int numCells;
    float3 worldOrigin;
    float3 cellSize;

    unsigned int numBodies;
    unsigned int maxParticlesPerCell;

	float gasStiffness;
    float viscosity;
    float surfaceTension;
    float restDensity;
	float particleMass;
    float interactionRadius;
	float timestep;
    float particleRadius;
    float3 gravity;
};

#endif /* ifndef SPH_KERNEL_H */
