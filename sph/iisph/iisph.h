#ifndef IISPH_H
#define IISPH_H

#include "sph.h"

NEREUS_NAMESPACE_BEGIN	

class IISPH : public SPH
{
public:
	IISPH ();
	IISPH (SphSimParams params);
	virtual ~IISPH ();

	/************************************
	*  INITIALIZE AND FINALIZE SOLVER  *
	************************************/
	virtual void _initialize();
	virtual void _finalize();

	/*********************************
	*  PERFORM ONE SIMULATION STEP  *
	*********************************/
	void update();

private:
	//real
	SReal *m_dSortedDensAdv;
	SReal *m_dSortedDensCorr;
	SReal *m_dSortedP_l;
	SReal *m_dSortedPreviousP;
	SReal *m_dSortedAii;

	//vec4
	SReal *m_dSortedVelAdv;
	SReal *m_dSortedForcesAdv;
	SReal *m_dSortedForcesP;
	SReal *m_dSortedDiiFluid;
	SReal *m_dSortedDiiBoundary;
	SReal *m_dSortedSumDij;
	SReal *m_dSortedNormal;

};

NEREUS_NAMESPACE_END 

#endif /* ifndef IISPH_ */
