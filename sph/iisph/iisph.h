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
	//float
	float *m_dSortedDensAdv;
	float *m_dSortedDensCorr;
	float *m_dSortedP_l;
	float *m_dSortedPreviousP;
	float *m_dSortedAii;

	//float4
	float *m_dSortedVelAdv;
	float *m_dSortedForcesAdv;
	float *m_dSortedForcesP;
	float *m_dSortedDiiFluid;
	float *m_dSortedDiiBoundary;
	float *m_dSortedSumDij;
	float *m_dSortedNormal;

};

} /* CFD */ 

#endif /* ifndef IISPH_ */
