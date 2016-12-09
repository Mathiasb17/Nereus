#include "sph.h"

#include <iostream>
#include <algorithm>

#include <glm/glm.hpp>

#include <thrust/host_vector.h>
#include <thrust/functional.h>
#include <thrust/extrema.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <thrust/sort.h>

namespace CFD
{

typedef thrust::host_vector<int>::iterator   IntIterator;
typedef thrust::host_vector<float>::iterator FloatIterator;
typedef thrust::host_vector<char>::iterator  CharIterator;
typedef thrust::host_vector<double>::iterator  DoubleIterator;
typedef thrust::host_vector<glm::vec4>::iterator  Vec3Iterator;
typedef thrust::host_vector<unsigned int>::iterator  UIntIterator;

typedef thrust::tuple<Vec3Iterator, Vec3Iterator, FloatIterator, FloatIterator, Vec3Iterator, Vec3Iterator> IteratorTuple;
//typedef thrust::tuple<UIntIterator, Vec3Iterator, Vec3Iterator, FloatIterator, FloatIterator, Vec3Iterator, Vec3Iterator> IteratorTuple;

typedef thrust::zip_iterator<IteratorTuple> ZipIterator;


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
	m_interaction_radius(0.0457f),
	m_grid_min(glm::vec4(-2,-2,-2,1)),
	m_nb_cell_x(10),
	m_nb_cell_y(10)
{
	m_particle_mass = powf(m_interaction_radius, 3)*m_rest_density;
	m_cell_size = m_interaction_radius;
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
	for (thrust::host_vector<glm::vec4>::iterator i  = m_pos.begin(); i != m_pos.end(); ++i)
	{
		unsigned int index1 = i - m_pos.begin();

		int k = (int)((m_pos[index1].x-m_grid_min.x)/m_cell_size);
		int l = (int)((m_pos[index1].y-m_grid_min.y)/m_cell_size);
		int m = (int)((m_pos[index1].z-m_grid_min.z)/m_cell_size);

		int K = m_nb_cell_x;
		int L = m_nb_cell_y;

		unsigned int key = k+l*K + m*K*L;

		m_key[index1] = key;
		m_neighbors[index1]->clear();
	}

	//sort particles
	ZipIterator iter(thrust::make_tuple(m_pos.begin(), m_vel.begin(), m_density.begin(), m_pressure.begin(),
					   m_forces.begin(), m_colors.begin()));

	thrust::sort_by_key(m_key.begin(), m_key.end(), iter);
}

void SPH::ComputeNeighbors()
{
	for (thrust::host_vector<glm::vec4>::iterator i  = m_pos.begin(); i != m_pos.end(); ++i)
	{
		unsigned int index1 = i - m_pos.begin();
		for (thrust::host_vector<glm::vec4>::iterator j  = m_pos.begin(); j != m_pos.end(); ++j)
		{
			unsigned int index2 = j - m_pos.begin();
			float len = glm::length(*i - *j);
			if(len > 0 && len <= m_interaction_radius /*&& index1 != index2*/)
			{
				m_neighbors[index1]->push_back(index2);
			}
			if(m_neighbors[index1]->size() > 40) std::cout << "nb neighbors : " << m_neighbors[index1]->size() << std::endl;
		}
	}
}

void SPH::ComputeDensitiesAndPressure()
{
	for (thrust::host_vector<glm::vec4>::iterator i  = m_pos.begin(); i != m_pos.end(); ++i)
	{
		unsigned int index1 = i - m_pos.begin();
		float dens = 0.f;

		//#pragma omg parallel for
		for (unsigned int j = 0; j < m_neighbors[index1]->size(); j++)
		{
			unsigned int index2 = m_neighbors[index1]->data()[j];
			glm::vec4 p_ij = m_pos[index1] - m_pos[index2];
			dens += m_particle_mass * Wdefault(p_ij.xyz(), m_interaction_radius);
		}
		m_density[index1] = dens;
		m_pressure[index1] = m_gas_stiffness * ( powf(dens/m_rest_density,7) - 1 );

		//std::cout << std::setw(10) << "density : " << dens << " | pressure " << pressure[index1] << std::endl;
	}
}

void SPH::ComputeInternalForces()
{
	//#pragma omg parallel for
	for (thrust::host_vector<glm::vec4>::iterator i  = m_pos.begin(); i != m_pos.end(); ++i)
	{
		unsigned int index1 = i - m_pos.begin();
		glm::vec3 pres_grad(0,0,0);
		glm::vec3 vel_lapl(0,0,0);
		glm::vec3 force_surf(0,0,0);

		//#pragma omg parallel for
		for (unsigned int j = 0; j < m_neighbors[index1]->size(); j++)
		{
			unsigned index2 = m_neighbors[index1]->data()[j];

			//pres
			glm::vec4 p_ij = m_pos[index1] - m_pos[index2];
			float pi_rhoi2 = m_pressure[index1] / powf(m_density[index1],2);
			float pj_rhoj2 = m_pressure[index2] / powf(m_density[index2],2);
			pres_grad += m_particle_mass * (pi_rhoi2 + pj_rhoj2) * Wpressure_grad(p_ij.xyz(), m_interaction_radius);

			//visc
			float mj_rhoj = m_particle_mass / m_density[index2];
			glm::vec4 v_ij = m_vel[index1] - m_vel[index2];
			float num = glm::dot(p_ij.xyz(), Wdefault_grad(p_ij.xyz(), m_interaction_radius));
			float denum = glm::dot(p_ij.xyz(), p_ij.xyz()) + 0.01f*(m_interaction_radius*m_interaction_radius);
			vel_lapl += mj_rhoj*v_ij.xyz()*(num/denum);

			//surface tension
			glm::vec3 b = m_particle_mass * p_ij.xyz()  * Wdefault(p_ij.xyz(), m_interaction_radius);

			force_surf += b;
		}
		float a = -(m_surface_tension / m_particle_mass);
		force_surf *= a;

		pres_grad *= m_density[index1];
		vel_lapl *= 2.f;

		glm::vec3 force_pres = -(m_particle_mass/m_density[index1]) * pres_grad;
		glm::vec3 force_visc = (m_particle_mass*m_viscosity) * vel_lapl;

		m_forces[index1] = glm::vec4(force_pres + force_visc + force_surf,0);
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
	//thrust::host_vector<glm::vec3>::iterator vel_max_length_it = thrust::max_element(m_vel.begin(), m_vel.end(), compareVel);
	//float len = glm::length(*vel_max_length_it);
	//m_timestep = 0.01 * (m_interaction_radius /  len);

	m_timestep = 1E-3f;

	for (thrust::host_vector<glm::vec4>::iterator i  = m_pos.begin(); i != m_pos.end(); ++i)
	{
		unsigned int index1 = i - m_pos.begin();

		m_vel[index1] += m_timestep*m_forces[index1]/m_particle_mass;
		m_pos[index1] += m_timestep*m_vel[index1];

		//is_nan
		if (m_pos[index1].x != m_pos[index1].x || m_pos[index1].y != m_pos[index1].y || m_pos[index1].z != m_pos[index1].z)
		{
			m_pos[index1] = glm::vec4(-100,-100,-100,1);
			m_vel[index1] = glm::vec4(0,0,0,0);
			//std::cout << "PARTICULE CLAMPEE !" << std::endl;
		}
	}
}

void SPH::addNewParticle(glm::vec4 p)
{
	thrust::host_vector<unsigned int> *v = new thrust::host_vector<unsigned int>();
	m_pos.push_back(p);
	m_density.push_back(0.f);
	m_pressure.push_back(0.f);
	m_vel.push_back(glm::vec4(0,0,0,0));
	m_forces.push_back(glm::vec4(0,0,0,0));
	m_neighbors.push_back(v);
	m_colors.push_back(glm::vec4(1,0,0,1));
	m_key.push_back(0);
}

void SPH::generateParticleCube(glm::vec4 center, glm::vec4 size)
{
	for(float x = center.x-size.x/2.f; x <= center.x+size.x/2.f; x += m_particle_radius*2 )
	{
		for(float y = center.y-size.y/2.f; y <= center.y+size.y/2.f; y += m_particle_radius*2 )
		{
			for(float z = center.z-size.z/2.f; z <= center.z+size.z/2.f; z += m_particle_radius*2 )
			{
				addNewParticle(glm::vec4(x,y,z,1.f));
			}
		}
	}
	std::cout << "Il y a eu " << m_pos.size() << " particules generees." << std::endl;
}

} /* CFD */ 
