#ifndef PCISPH_H
#define PCISPH_H

#include "sph/sph.h"

namespace CFD
{

class PCISPH : public SPH
{
public:
	PCISPH ();
	virtual ~PCISPH ();

	virtual void _intialize();
	virtual void _finalize();

	virtual void update();

	virtual void addNewParticle(glm::vec4 p, glm::vec4 v);

private:
	/******************
	*  HOST MEMBERS  *
	******************/
	float* m_velstar;
	float* m_posstar;
	float* m_densstar;
	float* m_denserror;
	
	/********************
	*  DEVICE MEMBERS  *
	********************/
	float* m_dvelstar;
	float* m_dposstar;
	float* m_ddensstar;
	float* m_ddenserror;
};

} /* CFD */ 

#endif /* end of include guard: PCISPH_H */
