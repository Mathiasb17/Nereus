#ifndef SPH_KERNEL_H
#define SPH_KERNEL_H

#define USE_TEX 0
#if USE_TEX
#define FETCH(t, i) tex1Dfetch(t##Tex, i)
#else
#define FETCH(t, i) t[i]
#endif

#include <vector_types.h>

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
