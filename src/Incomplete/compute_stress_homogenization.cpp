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
#include "compute_stress_homogenization.h"
#include "compute_pair_gran_local.h"
#include "atom.h"
#include "update.h"
#include "force.h"
#include "domain.h"
#include "group.h"
#include "error.h"
#include "modify.h" 

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

ComputeStressHomogenization::ComputeStressHomogenization(LAMMPS *lmp, int &iarg, int narg, char **arg) :
    Compute(lmp, iarg, narg, arg),
    cpgl_(0),
    measurement_volume(0)
{
    if (narg != 9) error->all(FLERR,"Illegal compute stress/homogenization command");
    xlo = force->numeric(FLERR,arg[3]);
    xhi = force->numeric(FLERR,arg[4]);
    ylo = force->numeric(FLERR,arg[5]);
    yhi = force->numeric(FLERR,arg[6]);
    zlo = force->numeric(FLERR,arg[7]);
    zhi = force->numeric(FLERR,arg[8]);

    std::cout <<"xlo = " << arg[3] << " " << xlo << std::endl;
    std::cout <<"xhi = " << arg[4] << " " << xhi << std::endl;
    std::cout <<"ylo = " << arg[5] << " " << ylo << std::endl;
    std::cout <<"yhi = " << arg[6] << " " << yhi << std::endl;
    std::cout <<"zlo = " << arg[7] << " " << zlo << std::endl;
    std::cout <<"zhi = " << arg[8] << " " << zhi << std::endl;
    scalar_flag = 1;
    extscalar = 1;

}

/* ---------------------------------------------------------------------- */

void ComputeStressHomogenization::init()
{
  cpgl_ = static_cast<ComputePairGranLocal*>(modify->find_compute_style_strict("pair/gran/local",0));

  // error if compute does not write pos
  if(cpgl_->offset_x1() < 0 || cpgl_->offset_x2() < 0)
        error->all(FLERR,"compute stress/homogenization requires a valid ID of a compute pair/gran/local that writes the positions");
  // error if compute does not write forces
  if(cpgl_->offset_f() < 0)
   error->all(FLERR,"compute stress/homogenization requires a valid ID of a compute pair/gran/local that writes the forces");
  // error if compute does not write multisphere id
  if(cpgl_->offset_f() < 0)
   error->all(FLERR,"compute stress/homogenization requires a valid ID of a compute pair/gran/local that writes the multisphere id's in contact");

  measurement_volume = (xhi-xlo)*(yhi-ylo)*(zhi-zlo);
}

/* ---------------------------------------------------------------------- */

int ComputeStressHomogenization::count()
{
    
    cpgl_->compute_local();
    cpgl_->invoked_flag |= INVOKED_LOCAL;

    return cpgl_->get_ncount();
}

/* ---------------------------------------------------------------------- */

double ComputeStressHomogenization::compute_scalar()
{
    invoked_scalar = update->ntimestep;
    int n_contacts = count();
    int offset_pos1 = cpgl_->offset_x1();
    int offset_pos2 = cpgl_->offset_x2();
    int offset_msid
    int offset_f = cpgl_->offset_f();

    double summation = 0;
    for (int i = 0; i < n_contacts; i++) {
      summation += cpgl_->get_data()[i][offset_f];
    }    
    MPI_Allreduce(&summation,&scalar,1,MPI_DOUBLE,MPI_SUM,world);

    scalar /= measurement_volume;
    return scalar;
}

