#ifndef SPH_H
#define SPH_H 

#include <vector>

#ifndef GLM_SWIZZLE
#define GLM_SWIZZLE
#endif /* ifndef GLM_SWIZZLE */
#include <glm/glm.hpp>

#include <thrust/host_vector.h>

#include "sph_kernel.cuh"

#define MAX_PARTICLE_NUMBER 150000

namespace CFD
{

class SPH
{
public:
	SPH ();
	virtual ~SPH ();

	/*************
	*  Kernels  *
	*************/
	static float Wdefault(glm::vec3 r, float h);
	static glm::vec3 Wdefault_grad(glm::vec3 r, float h);
	static glm::vec3 Wpressure_grad(glm::vec3 r, float h);
	static float Wviscosity_laplacian(glm::vec3 r, float h);

	/**********
	*  Initialize and finalize  *
	**********/
	void _intialize();
	void _finalize();

	/********************
	 *  GENERATE FLUID  *
	 ********************/
	void addNewParticle(glm::vec4 p, glm::vec4 v);
	void generateParticleCube(glm::vec4 center, glm::vec4 size, glm::vec4 vel);

	/*********************************
	*  PERFORM ONE SIMULATION STEP  *
	*********************************/
	void update();

	/*************
	 *  GETTERS  *
	 *************/
	float getGasStiffness() const {return m_params.gasStiffness;}
	float getRestDensity() const {return m_params.restDensity;}
	float getParticleMass() const {return m_params.particleMass;}
	float getParticleRadius() const {return m_params.particleRadius;}
	float getTimestep() const {return m_params.timestep;}
	float getViscosity() const {return m_params.viscosity;}
	float getSurfaceTension() const {return m_params.surfaceTension;}
	float getInteractionRadius() const {return m_params.interactionRadius;}

	float* & getPos() {return m_pos;}
	float* & getCol() {return m_colors;}
	float* & getVel() {return m_vel;}

	/*************
	*  SETTERS  *
	*************/
	void setGasStiffness(float new_stiffness){m_params.gasStiffness = new_stiffness;}
	void setRestDensity(float new_restdensity){m_params.restDensity = new_restdensity;}
	void setParticleMass(float new_particlemass){m_params.particleMass = new_particlemass;}
	void setViscosity(float new_viscosity){m_params.viscosity = new_viscosity;}
	void setSurfaceTension(float new_surfacetension){m_params.surfaceTension = new_surfacetension;}

public:
	/********************
	 *  DEVICE MEMBERS  *
	 ********************/

	float* m_dpos;
	float* m_dvel;
	float* m_ddensity;
	float* m_dpressure;
	float* m_dforces;
	float* m_dcolors;

	float *m_dSortedPos;
	float *m_dSortedVel;
	float *m_dSortedDens;
	float *m_dSortedPress;
	float *m_dSortedForces;
	float *m_dSortedCol;

	unsigned int *m_dGridParticleHash; 
	unsigned int *m_dGridParticleIndex;
	unsigned int *m_dCellStart;
	unsigned int *m_dCellEnd;

	/******************
	 *  HOST MEMBERS  *
	 ******************/
	unsigned int* m_hParticleHash;
	unsigned int* m_hCellStart;
	unsigned int* m_hCellEnd;
	unsigned int  m_gridSortBits;

	float *m_pos;
	float *m_vel;
	float *m_density;
	float *m_pressure;
	float *m_forces;
	float *m_colors;

	unsigned int m_numParticles;

	SphSimParams m_params;
};

} /*  CFD */ 

#endif /* ifndef SPH_H */
