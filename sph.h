#ifndef SPH_H
#define SPH_H 

#include <vector>
#include <glm/glm.hpp>

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
	void addNewParticle(glm::vec3 p);
	void generateParticleCube(glm::vec3 center, glm::vec3 size);

	/*************
	 *  GETTERS  *
	 *************/
	float getGasStiffness() const {return m_gas_stiffness;}
	float getRestDensity() const {return m_rest_density;}
	float getParticleMass() const {return m_particle_mass;}
	float getParticleRadius() const {return m_particle_radius;}
	float getTimestep() const {return m_timestep;}
	float getViscosity() const {return m_viscosity;}
	float getSurfaceTension() const {return m_surface_tension;}
	float getInteractionRadius() const {return m_interaction_radius;}

	std::vector<glm::vec3> & getPos() {return m_pos;}
	std::vector<glm::vec3> & getCol() {return m_colors;}
	std::vector<glm::vec3> & getVel() {return m_vel;}

	/*************
	*  SETTERS  *
	*************/
	void setGasStiffness(float new_stiffness){m_gas_stiffness = new_stiffness;}
	void setRestDensity(float new_restdensity){m_rest_density = new_restdensity;}
	void setParticleMass(float new_particlemass){m_particle_mass = new_particlemass;}
	void setViscosity(float new_viscosity){m_viscosity = new_viscosity;}
	void setSurfaceTension(float new_surfacetension){m_surface_tension = new_surfacetension;}

public:

	float m_gas_stiffness;
	float m_rest_density;
	float m_particle_mass;
	float m_particle_radius;
	float m_timestep;
	float m_viscosity;
	float m_surface_tension;
	float m_interaction_radius;

	std::vector<glm::vec3> m_pos;
	std::vector<glm::vec3> m_vel;
	std::vector<float> m_density;
	std::vector<float> m_pressure;
	std::vector<glm::vec3> m_forces;
	std::vector<std::vector<unsigned int> > m_neighbors;
	std::vector<glm::vec3> m_colors;
};

} /*  CFD */ 

#endif /* ifndef SPH_H */
