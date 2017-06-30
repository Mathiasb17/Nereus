#ifndef PCISPH_H
#define PCISPH_H

#include "sph.h"

namespace CFD
{
	
class PCISPH : public CFD::SPH
{
public:
	PCISPH ();
	PCISPH (SphSimParams params);
	virtual ~PCISPH ();

	/****
	* Initialize and finalize *
	****/
	virtual void _initialize();
	virtual void _finalize();

	/*********************************
	*  PERFORM ONE SIMULATION STEP  *
	*********************************/
	void update();

private:
	//float
	SReal * m_dSortedRhoAdv;
	
	//float4
	SReal * m_dSortedVelAdv;
};

} /* CFD */ 

#endif /* ifndef PCISPH_ */
