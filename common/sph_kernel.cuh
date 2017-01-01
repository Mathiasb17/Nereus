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
	/*********************
	*  GRID PARAMETERS  *
	*********************/
    uint3 gridSize;
    unsigned int numCells;
    float3 worldOrigin;
    float3 cellSize;

    unsigned int numBodies;
    unsigned int maxParticlesPerCell;
	
	/************************
	*  PHYSICS PARAMETERS  *
	************************/
	float gasStiffness;
    float viscosity;
    float surfaceTension;
    float restDensity;
	float particleMass;
    float interactionRadius;
	float timestep;
    float particleRadius;
    float3 gravity;

	/***********************************
	*  SPH KERNELS PRE-COMPUTED PART  *
	***********************************/
	float kpoly;
	float kpoly_grad;
	float kpress_grad;

	float kvisc_grad;
	float kvisc_denum;
};

#endif /* ifndef SPH_KERNEL_H */
