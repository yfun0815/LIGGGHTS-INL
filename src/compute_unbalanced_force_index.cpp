/* ----------------------------------------------------------------------
    This is the

    ██╗     ██╗ ██████╗  ██████╗  ██████╗ ██╗  ██╗████████╗███████╗
    ██║     ██║██╔════╝ ██╔════╝ ██╔════╝ ██║  ██║╚══██╔══╝██╔════╝
    ██║     ██║██║  ███╗██║  ███╗██║  ███╗███████║   ██║   ███████╗
    ██║     ██║██║   ██║██║   ██║██║   ██║██╔══██║   ██║   ╚════██║
    ███████╗██║╚██████╔╝╚██████╔╝╚██████╔╝██║  ██║   ██║   ███████║
    ╚══════╝╚═╝ ╚═════╝  ╚═════╝  ╚═════╝ ╚═╝  ╚═╝   ╚═╝   ╚══════╝®

    DEM simulation engine, released by
    DCS Computing Gmbh, Linz, Austria
    http://www.dcs-computing.com, office@dcs-computing.com

    LIGGGHTS® is part of CFDEM®project:
    http://www.liggghts.com | http://www.cfdem.com

    Core developer and main author:
    Christoph Kloss, christoph.kloss@dcs-computing.com

    LIGGGHTS® is open-source, distributed under the terms of the GNU Public
    License, version 2 or later. It is distributed in the hope that it will
    be useful, but WITHOUT ANY WARRANTY; without even the implied warranty
    of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. You should have
    received a copy of the GNU General Public License along with LIGGGHTS®.
    If not, see http://www.gnu.org/licenses . See also top-level README
    and LICENSE files.

    LIGGGHTS® and CFDEM® are registered trade marks of DCS Computing GmbH,
    the producer of the LIGGGHTS® software and the CFDEM®coupling software
    See http://www.cfdem.com/terms-trademark-policy for details.

-------------------------------------------------------------------------
    Contributing author and copyright for this file:
    This file is from LAMMPS
    LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
    http://lammps.sandia.gov, Sandia National Laboratories
    Steve Plimpton, sjplimp@sandia.gov

    Copyright (2003) Sandia Corporation.  Under the terms of Contract
    DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
    certain rights in this software.  This software is distributed under
    the GNU General Public License.
------------------------------------------------------------------------- */

/*
	This fix sums the squared unbalanced forces on each particle for the 
	purpose of calculating  an index of unbalanced forces in the input
	script.
*/
#include <mpi.h>
#include "compute_unbalanced_force_index.h"
#include "compute_pair_gran_local.h"
#include "atom.h"
#include "update.h"
#include "force.h"
#include "domain.h"
#include "group.h"
#include "error.h"
#include "modify.h" 
#include "fix_multisphere.h" 

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

ComputeUnbalancedForceIndex::ComputeUnbalancedForceIndex(LAMMPS *lmp, int &iarg, int narg, char **arg) :
    Compute(lmp, iarg, narg, arg),
    fix_ms(NULL),
    cpgl_(0)
{
    if (narg != 3) error->all(FLERR,"Illegal compute unbalanced/force/index command");

    scalar_flag = 1;
    extscalar = 1;

}

/* ---------------------------------------------------------------------- */

void ComputeUnbalancedForceIndex::init()
{
  fix_ms =  static_cast<FixMultisphere*>(modify->find_fix_style("multisphere",0));
  cpgl_ = static_cast<ComputePairGranLocal*>(modify->find_compute_style_strict("pair/gran/local",0));
  // error if compute does not write forces
  //if(cpgl_->offset_f() < 0)
  // error->all(FLERR,"compute unbalanced/force/index requires a valid ID of a compute pair/gran/local that writes the forces");
  if(cpgl_->offset_squaredforce() < 0)
     error->all(FLERR,"compute unbalanced/force/index requires a valid ID of a compute pair/gran/local that writes the squared forces");

}

/* ---------------------------------------------------------------------- */

int ComputeUnbalancedForceIndex::count()
{
    
    cpgl_->compute_local();
    cpgl_->invoked_flag |= INVOKED_LOCAL;

    return cpgl_->get_ncount();
}

/* ---------------------------------------------------------------------- */

double ComputeUnbalancedForceIndex::compute_scalar()
{
    invoked_scalar = update->ntimestep;

    double **f = atom->f;
    double *rmass = atom->rmass;
    int *mask = atom->mask;
    int nlocal = atom->nlocal;
    double n_total=0;
    double unbalancedForceSquared = 0.0;

    if (rmass)
    {
        for (int i = 0; i < nlocal; i++)
	{
            if (mask[i] & groupbit && (!fix_ms || fix_ms->belongs_to(i) < 0))
            {
                n_total++;
	        unbalancedForceSquared += vectorMag3DSquared(f[i]);
	   }
        }
    }
    else
    {
        for (int i = 0; i < nlocal; i++)
	{
            if (mask[i] & groupbit)
	    {
	        n_total++;
                unbalancedForceSquared += vectorMag3DSquared(f[i]);
	    }
	}
    }
    
    MPI_Sum_Scalar(n_total,world);    
    MPI_Sum_Scalar(unbalancedForceSquared,world);
    // for multispheres we need to get the square of unbalanced forces from the multisphere fix
    if (fix_ms)
    {
        n_total += fix_ms->n_body_all();
        unbalancedForceSquared += fix_ms->extract_unbalancedforce_squared();
    }
    double numerator = unbalancedForceSquared/n_total;

    double n_contacts = count();
    int offset = cpgl_->offset_squaredforce();
    double sumContactForcesSquared = 0;
    double fx, fy, fz, fmag2;
    for (int i = 0; i < n_contacts; i++) {
      sumContactForcesSquared += cpgl_->get_data()[i][offset];
    }    
    MPI_Sum_Scalar(n_contacts,world);
    if (n_contacts < 1) {
      scalar = 1e20;
      return scalar;
    }
    MPI_Sum_Scalar(sumContactForcesSquared,world);
    double denominator = sumContactForcesSquared/n_contacts;

    scalar = ::sqrt(numerator/denominator);
    return scalar;
}

