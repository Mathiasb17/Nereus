#ifndef SPH_H
#define SPH_H 

#include <vector>

#ifndef GLM_SWIZZLE
#define GLM_SWIZZLE
#endif /* ifndef GLM_SWIZZLE */
#include <glm/glm.hpp>

#include <thrust/host_vector.h>

#include "sph_kernel.cuh"

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

	/***************
	*  NEIGHBORS  *
	***************/
	void initNeighbors();
	void ComputeNeighbors();

	/**********************
	*  SPH CALCULATIONS  *
	**********************/
	void ComputeDensitiesAndPressure();
	void ComputeInternalForces();
	void ComputeExternalForces();

	/******************************
	 *  COLLISIONS AND ADVECTION  *
	 ******************************/
	void CollisionDetectionsAndResponses();
	void ComputeImplicitEulerScheme();

	/********************
	 *  GENERATE FLUID  *
	 ********************/
	void addNewParticle(glm::vec4 p);
	void generateParticleCube(glm::vec4 center, glm::vec4 size);

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

	thrust::host_vector<glm::vec4> & getPos() {return m_pos;}
	thrust::host_vector<glm::vec4> & getCol() {return m_colors;}
	thrust::host_vector<glm::vec4> & getVel() {return m_vel;}

	/*************
	*  SETTERS  *
	*************/
	void setGasStiffness(float new_stiffness){m_params.gasStiffness = new_stiffness;}
	void setRestDensity(float new_restdensity){m_params.restDensity = new_restdensity;}
	void setParticleMass(float new_particlemass){m_params.particleMass = new_particlemass;}
	void setViscosity(float new_viscosity){m_params.viscosity = new_viscosity;}
	void setSurfaceTension(float new_surfacetension){m_params.surfaceTension = new_surfacetension;}

public:
	SphSimParams m_params;

	glm::vec4 m_grid_min;
	float m_nb_cell_x;
	float m_nb_cell_y;
	float m_cell_size;

	uint* m_hParticleHash;
	uint* m_hCellStart;
	uint* m_hCellEnd;

	uint   m_gridSortBits;

	glm::vec4* m_dpos;
	glm::vec4* m_dvel;
	float* m_ddensity;
	float* m_dpressure;
	glm::vec4* m_dforces;
	glm::vec4* m_dcolors;

	thrust::host_vector<unsigned int> m_key;
	thrust::host_vector<glm::vec4> m_pos;
	thrust::host_vector<glm::vec4> m_vel;
	thrust::host_vector<float> m_density;
	thrust::host_vector<float> m_pressure;
	thrust::host_vector<glm::vec4> m_forces;
	thrust::host_vector<thrust::host_vector<unsigned int>* > m_neighbors;
	thrust::host_vector<glm::vec4> m_colors;
};

} /*  CFD */ 

#endif /* ifndef SPH_H */
