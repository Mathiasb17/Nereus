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

	/*********************
	*  BOUNDARY PARAMS  *
	*********************/
	float beta;

	/***********************************
	*  SPH KERNELS PRE-COMPUTED PART  *
	***********************************/
	float kpoly;      //muller03
	float kpoly_grad; //muller03
	float kpress_grad;//muller03

	float kvisc_grad; //muller03
	float kvisc_denum;//muller03

	float ksurf1; //akinci surface tension kernel
	float ksurf2; //akinci surface tension kernel

	float bpol;//akinci boundary kernel
};

#endif /* ifndef SPH_KERNEL_H */
