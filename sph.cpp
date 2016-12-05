#include "sph.h"

#include <iostream>
#include <algorithm>

#include <glm/glm.hpp>

#include <thrust/host_vector.h>

namespace CFD
{

static bool compareVel(glm::vec3 v1, glm::vec3 v2)
{
	return glm::length(v1) < glm::length(v2);
}

SPH::SPH ():
	m_gas_stiffness(300.f),
	m_rest_density(998.29),
	m_particle_radius(0.02),
	m_timestep(1E-3f),
	m_viscosity(0.013f),
	m_surface_tension(0.01f),
	m_interaction_radius(0.0457f)
{
	m_particle_mass = powf(m_interaction_radius, 3)*m_rest_density;
}

SPH::~SPH ()
{

}

float SPH::Wdefault(glm::vec3 r, float h)
{
	float l_r = glm::length(r);
	if(l_r > h) return 0.f;
	float a = (315.f / (64.f * M_PI * powf(h,9.f)));
	float b = powf(h*h - l_r*l_r,3.f);
	return a * b;
}

glm::vec3 SPH::Wdefault_grad(glm::vec3 r, float h)
{
	float l_r = length(r);
	float a = -945/(32*M_PI*powf(h,9));
	float b = powf(h*h - l_r*l_r, 2);
	return a*r*b;
}

glm::vec3 SPH::Wpressure_grad(glm::vec3 r, float h)
{
	float l_r = glm::length(r);
	float a = -(45.f/ (M_PI * powf(h,6.f)));
	glm::vec3 b = r / l_r;
	float c = (h-l_r)*(h-l_r);
	return a*b*c;
}

float SPH::Wviscosity_laplacian(glm::vec3 r, float h)
{
	float l_r = glm::length(r);
	float a = 45 / (M_PI * powf(h,6));
	float b = h-l_r;
	return a*b;
}

void SPH::initNeighbors()
{
	for (std::vector<glm::vec3>::iterator i  = m_pos.begin(); i != m_pos.end(); ++i)
	{
		unsigned int index1 = i - m_pos.begin();
		m_neighbors[index1].clear();
	}
}

void SPH::ComputeNeighbors()
{
	for (std::vector<glm::vec3>::iterator i  = m_pos.begin(); i != m_pos.end(); ++i)
	{
		unsigned int index1 = i - m_pos.begin();
		for (std::vector<glm::vec3>::iterator j  = m_pos.begin(); j != m_pos.end(); ++j)
		{
			unsigned int index2 = j - m_pos.begin();
			float len = glm::length(*i - *j);
			if(len > 0 && len <= m_interaction_radius /*&& index1 != index2*/)
			{
				m_neighbors[index1].push_back(index2);
			}
			if(m_neighbors[index1].size() > 40) std::cout << "nb neighbors : " << m_neighbors[index1].size() << std::endl;
		}
	}
}

void SPH::ComputeDensitiesAndPressure()
{
	for (std::vector<glm::vec3>::iterator i  = m_pos.begin(); i != m_pos.end(); ++i)
	{
		unsigned int index1 = i - m_pos.begin();
		float dens = 0.f;

		//#pragma omg parallel for
		for (unsigned int j = 0; j < m_neighbors[index1].size(); j++)
		{
			unsigned int index2 = m_neighbors[index1][j];
			glm::vec3 p_ij = m_pos[index1] - m_pos[index2];
			dens += m_particle_mass * Wdefault(p_ij, m_interaction_radius);
		}
		m_density[index1] = dens;
		m_pressure[index1] = m_gas_stiffness * ( powf(dens/m_rest_density,7) - 1 );

		//std::cout << std::setw(10) << "density : " << dens << " | pressure " << pressure[index1] << std::endl;
	}
}

void SPH::ComputeInternalForces()
{
	//#pragma omg parallel for
	for (std::vector<glm::vec3>::iterator i  = m_pos.begin(); i != m_pos.end(); ++i)
	{
		unsigned int index1 = i - m_pos.begin();
		glm::vec3 pres_grad(0,0,0);
		glm::vec3 vel_lapl(0,0,0);
		glm::vec3 force_surf(0,0,0);

		//#pragma omg parallel for
		for (unsigned int j = 0; j < m_neighbors[index1].size(); j++)
		{
			unsigned index2 = m_neighbors[index1][j];

			//pres
			glm::vec3 p_ij = m_pos[index1] - m_pos[index2];
			float pi_rhoi2 = m_pressure[index1] / powf(m_density[index1],2);
			float pj_rhoj2 = m_pressure[index2] / powf(m_density[index2],2);
			pres_grad += m_particle_mass * (pi_rhoi2 + pj_rhoj2) * Wpressure_grad(p_ij, m_interaction_radius);

			//visc
			float mj_rhoj = m_particle_mass / m_density[index2];
			glm::vec3 v_ij = m_vel[index1] - m_vel[index2];
			float num = glm::dot(p_ij, Wdefault_grad(p_ij, m_interaction_radius));
			float denum = glm::dot(p_ij, p_ij) + 0.01f*(m_interaction_radius*m_interaction_radius);
			vel_lapl += mj_rhoj*v_ij*(num/denum);

			//surface tension
			glm::vec3 b = m_particle_mass * p_ij  * Wdefault(p_ij, m_interaction_radius);

			force_surf += b;
		}
		float a = -(m_surface_tension / m_particle_mass);
		force_surf *= a;

		pres_grad *= m_density[index1];
		vel_lapl *= 2.f;

		glm::vec3 force_pres = -(m_particle_mass/m_density[index1]) * pres_grad;
		glm::vec3 force_visc = (m_particle_mass*m_viscosity) * vel_lapl;

		m_forces[index1] = force_pres + force_visc + force_surf;
	}
}

void SPH::ComputeExternalForces()
{

}

void SPH::CollisionDetectionsAndResponses()
{

}

void SPH::ComputeImplicitEulerScheme()
{
	//compute timestep
	std::vector<glm::vec3>::iterator vel_max_length_it = std::max(m_vel.begin(), m_vel.end());
	float len = glm::length(*vel_max_length_it);
	m_timestep = 0.01 * (m_interaction_radius /  len);

	for (std::vector<glm::vec3>::iterator i  = m_pos.begin(); i != m_pos.end(); ++i)
	{
		unsigned int index1 = i - m_pos.begin();

		m_vel[index1] += m_timestep*m_forces[index1]/m_particle_mass;
		m_pos[index1] += m_timestep*m_vel[index1];

		//is_nan
		if (m_pos[index1].x != m_pos[index1].x || m_pos[index1].y != m_pos[index1].y || m_pos[index1].z != m_pos[index1].z)
		{
			m_pos[index1] = glm::vec3(-100,-100,-100);
			m_vel[index1] = glm::vec3(0,0,0);
			std::cout << "PARTICULE CLAMPEE !" << std::endl;
		}
	}
}

void SPH::addNewParticle(glm::vec3 p)
{
	std::vector<unsigned int> v;
	m_pos.push_back(p);
	m_density.push_back(0.f);
	m_pressure.push_back(0.f);
	m_vel.push_back(glm::vec3(0,0,0));
	m_forces.push_back(glm::vec3(0,0,0));
	m_neighbors.push_back(v);
	m_colors.push_back(glm::vec3(1,0,0));
}

void SPH::generateParticleCube(glm::vec3 center, glm::vec3 size)
{
	for(float x = center.x-size.x/2.f; x <= center.x+size.x/2.f; x += m_particle_radius*2 )
	{
		for(float y = center.y-size.y/2.f; y <= center.y+size.y/2.f; y += m_particle_radius*2 )
		{
			for(float z = center.z-size.z/2.f; z <= center.z+size.z/2.f; z += m_particle_radius*2 )
			{
				addNewParticle(glm::vec3(x,y,z));
			}
		}
	}
	std::cout << "Il y a eu " << m_pos.size() << " particules generees." << std::endl;
}

} /* CFD */ 
