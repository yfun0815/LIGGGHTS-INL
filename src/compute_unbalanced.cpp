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

#include <mpi.h>
#include "compute_unbalanced.h"
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

ComputeUnbalanced::ComputeUnbalanced(LAMMPS *lmp, int &iarg, int narg, char **arg) :
    Compute(lmp, iarg, narg, arg),
    fix_ms(NULL)
{
    if (narg != 3) error->all(FLERR,"Illegal compute unbalanced command");

    scalar_flag = 1;
    extscalar = 1;
}

/* ---------------------------------------------------------------------- */

void ComputeUnbalanced::init()
{
  fix_ms =  static_cast<FixMultisphere*>(modify->find_fix_style("multisphere",0)); 
}

/* ---------------------------------------------------------------------- */

double ComputeUnbalanced::compute_scalar()
{
    invoked_scalar = update->ntimestep;

    double **f = atom->f;
    double *rmass = atom->rmass;
    int *mask = atom->mask;
    int nlocal = atom->nlocal;

    double unbalancedforce = 0.0;

    if (rmass)
    {
        for (int i = 0; i < nlocal; i++)
	{
	    if (mask[i] & groupbit && (!fix_ms || fix_ms->belongs_to(i) < 0))
	    {
	        unbalancedforce += vectorMag3D(f[i]);
	    }
	}
    }else{
        for (int i = 0; i < nlocal; i++)
	{
	    if (mask[i] & groupbit)
	    {
	        unbalancedforce += vectorMag3D(f[i]);
	    }
	}
    }

    MPI_Allreduce(&unbalancedforce,&scalar,1,MPI_DOUBLE,MPI_SUM,world);

    if (fix_ms)
    {
    // for multispheres we need to get the kinetic energy from the multisphere fix
        scalar += fix_ms->extract_unbalancedforce();
    }
    
    return scalar;
}
