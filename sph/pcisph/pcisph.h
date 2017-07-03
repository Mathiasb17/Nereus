#ifndef PCISPH_H
#define PCISPH_H

#include "sph.h"

NEREUS_NAMESPACE_BEGIN
	
class PCISPH : public SPH
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
	SReal * m_dSortedForcesAdv;
	SReal * m_dSortedForcesPres;
	SReal * m_dSortedPosAdv;
};

NEREUS_NAMESPACE_END

#endif /* ifndef PCISPH_ */
