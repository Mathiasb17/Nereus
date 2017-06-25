#ifndef SPH_H
#define SPH_H 

#include <vector>

#ifndef GLM_SWIZZLE
#define GLM_SWIZZLE
#endif /* ifndef GLM_SWIZZLE */
#include <glm/glm.hpp>

#include <thrust/host_vector.h>

#include "common.h"
#include "sph_kernel.cuh"

#define MAX_PARTICLE_NUMBER 150000

namespace CFD
{

class SPH
{
public:
	SPH ();
	SPH (SphSimParams params);
	virtual ~SPH ();

	/**********
	*  Initialize and finalize  *
	**********/
	virtual void _intialize();
	virtual void _finalize();

	/********************
	 *  GENERATE FLUID  *
	 ********************/
	virtual void addNewParticle(SVec4 p, SVec4 v);
	virtual void generateParticleCube(SVec4 center, SVec4 size, SVec4 vel);

	/*********************************
	*  PERFORM ONE SIMULATION STEP  *
	*********************************/
	virtual void update();
	
	/**********
	*  GRID  *
	**********/
	void computeGridMinMax();

	/*************
	 *  GETTERS  *
	 *************/
	SReal getGasStiffness() const {return m_params.gasStiffness;}
	SReal getRestDensity() const {return m_params.restDensity;}
	SReal getParticleMass() const {return m_params.particleMass;}
	SReal getParticleRadius() const {return m_params.particleRadius;}
	SReal getTimestep() const {return m_params.timestep;}
	SReal getViscosity() const {return m_params.viscosity;}
	SReal getSurfaceTension() const {return m_params.surfaceTension;}
	SReal getInteractionRadius() const {return m_params.interactionRadius;}

	unsigned int getNumCells() const {return m_params.numCells;}

	SReal* & getPos() {return m_pos;}
	SReal* & getCol() {return m_colors;}
	SReal* & getVel() {return m_vel;}

	SReal* getHostPos() const {return m_pos;}
	SReal* getHostCol() const {return m_colors;}

	unsigned int getNumParticles() const {return m_numParticles;}

	/*************
	*  SETTERS  *
	*************/
	void setGasStiffness(SReal new_stiffness){m_params.gasStiffness = new_stiffness;}
	void setRestDensity(SReal new_restdensity){m_params.restDensity = new_restdensity;}
	void setParticleMass(SReal new_particlemass){m_params.particleMass = new_particlemass;}
	void setViscosity(SReal new_viscosity){m_params.viscosity = new_viscosity;}
	void setSurfaceTension(SReal new_surfacetension){m_params.surfaceTension = new_surfacetension;}

	void setBi(SReal* bi){m_bi = bi;}
	void setVbi(SReal* vbi){m_vbi = vbi;}

	void setNumBoundaries(unsigned int nb){m_num_boundaries = nb;}

	void updateGpuBoundaries(unsigned int nb_boundary_spheres);

protected:
	/********************
	 *  DEVICE MEMBERS  *
	 ********************/
	SReal* m_dpos;
	SReal* m_dvel;
	SReal* m_ddensity;
	SReal* m_dpressure;
	SReal* m_dforces;
	SReal* m_dcolors;

	SReal *m_dSortedPos;
	SReal *m_dSortedVel;
	SReal *m_dSortedDens;
	SReal *m_dSortedPress;
	SReal *m_dSortedForces;
	SReal *m_dSortedCol;

	unsigned int *m_dGridParticleHash; 
	unsigned int *m_dGridParticleIndex;
	unsigned int *m_dCellStart;
	unsigned int *m_dCellEnd;

	SReal* m_dbi;//gpu boundaries
	SReal* m_dvbi;

	SReal* m_dSortedbi;
	SReal* m_dSortedVbi;

	unsigned int* m_dGridBoundaryHash, *m_dGridBoundaryIndex;
	unsigned int *m_dBoundaryCellStart;
	unsigned int *m_dBoundaryCellEnd;

	/******************
	 *  HOST MEMBERS  *
	 ******************/
	unsigned int* m_hParticleHash;
	unsigned int* m_hCellStart;
	unsigned int* m_hCellEnd;
	unsigned int  m_gridSortBits;

	SReal *m_pos;
	SReal *m_vel;
	SReal *m_density;
	SReal *m_pressure;
	SReal *m_forces;
	SReal *m_colors;

	unsigned int m_numParticles;

	SphSimParams m_params;

	SReal* m_bi; //boundary particles
	SReal* m_vbi;//boundary particles volume
	unsigned int m_num_boundaries;
};

} /*  CFD */ 

#endif /* ifndef SPH_H */
