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
	/* data */
};

} /* CFD */ 

#endif /* ifndef IISPH_ */
