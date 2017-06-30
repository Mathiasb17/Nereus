#ifndef IISPH_H
#define IISPH_H

#include "sph.h"

namespace CFD
{
	
class IISPH : public CFD::SPH
{
public:
	IISPH ();
	IISPH (SphSimParams params);
	virtual ~IISPH ();

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
	SReal *m_dSortedDensAdv;
	SReal *m_dSortedDensCorr;
	SReal *m_dSortedP_l;
	SReal *m_dSortedPreviousP;
	SReal *m_dSortedAii;

	//float4
	SReal *m_dSortedVelAdv;
	SReal *m_dSortedForcesAdv;
	SReal *m_dSortedForcesP;
	SReal *m_dSortedDiiFluid;
	SReal *m_dSortedDiiBoundary;
	SReal *m_dSortedSumDij;
	SReal *m_dSortedNormal;

};

} /* CFD */ 

#endif /* ifndef IISPH_ */
