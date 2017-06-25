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
    SVec3 worldOrigin;
    SVec3 cellSize;

    unsigned int numBodies;
    unsigned int maxParticlesPerCell;
	
	/************************
	*  PHYSICS PARAMETERS  *
	************************/
	SReal gasStiffness;
    SReal viscosity;
    SReal surfaceTension;
    SReal restDensity;
	SReal particleMass;
    SReal interactionRadius;
	SReal timestep;
    SReal particleRadius;
    SVec3 gravity;

	/*********************
	*  BOUNDARY PARAMS  *
	*********************/
	SReal beta;

	/***********************************
	*  SPH KERNELS PRE-COMPUTED PART  *
	***********************************/
	SReal kpoly;      //muller03
	SReal kpoly_grad; //muller03
	SReal kpress_grad;//muller03

	SReal kvisc_grad; //muller03
	SReal kvisc_denum;//muller03

	SReal ksurf1; //akinci surface tension kernel
	SReal ksurf2; //akinci surface tension kernel

	SReal bpol;//akinci boundary kernel
};

#endif /* ifndef SPH_KERNEL_H */
