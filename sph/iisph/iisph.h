#ifndef IISPH_H
#define IISPH_H

#include "sph.h"

namespace CFD
{
	
class IISPH : public CFD::SPH
{
public:
	IISPH ();
	virtual ~IISPH ();

	/****
	* Initialize and finalize *
	****/
	virtual void _intialize();
	virtual void _finalize();

	/*********************************
	*  PERFORM ONE SIMULATION STEP  *
	*********************************/
	void update();

private:
	float* m_dVel_adv;
	float* m_dDensity_adv;
	float* m_dDisplacement_factor;
	float* m_dAdvection_factor;
};

} /* CFD */ 

#endif /* ifndef IISPH_ */
