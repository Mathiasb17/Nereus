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

private:
	float* vel_adv;
	float* density_adv;
	float* displacement_factor;
};

} /* CFD */ 

#endif /* ifndef IISPH_ */
