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

#include <stdio.h>
#include <string.h>
#include "set_multisphere_all.h"
#include "fix_template_multisphere.h"
#include "fix_multisphere.h"
#include "atom.h"
#include "force.h"
#include "update.h"
#include "respa.h"
#include "error.h"
#include "math_const.h"

using namespace LAMMPS_NS;
using namespace FixConst;

/* ---------------------------------------------------------------------- */

void SetMultisphereAll::command(int narg, char **arg)
{
    int nbody = static_cast<FixMultisphere*>(modify->find_fix_style("multisphere",0))->multisphere_.n_body_all();
    double **quat = static_cast<FixMultisphere*>(modify->find_fix_style("multisphere",0))->multisphere_.quat_.begin();
    double **xcm = static_cast<FixMultisphere*>(modify->find_fix_style("multisphere",0))->multisphere_.xcm_.begin();
    double **vcm = static_cast<FixMultisphere*>(modify->find_fix_style("multisphere",0))->multisphere_.vcm_.begin();
    double **ex_space = static_cast<FixMultisphere*>(modify->find_fix_style("multisphere",0))->multisphere_.ex_space_.begin();
    double **ey_space = static_cast<FixMultisphere*>(modify->find_fix_style("multisphere",0))->multisphere_.ey_space_.begin();
    double **ez_space = static_cast<FixMultisphere*>(modify->find_fix_style("multisphere",0))->multisphere_.ez_space_.begin();
    double **omega = static_cast<FixMultisphere*>(modify->find_fix_style("multisphere",0))->multisphere_.omega_.begin();
    double **inertia = static_cast<FixMultisphere*>(modify->find_fix_style("multisphere",0))->multisphere_.inertia_.begin();
    int iarg = 0;
    int ibody;
    while (iarg < narg) {
        if (strcmp(arg[iarg],"x") == 0) {
            if (iarg+2 > narg) error->all(FLERR,"Illegal set command");
	    for (ibody=0; ibody < nbody; ibody++) xcm[ibody][0] = force->numeric(FLERR,arg[iarg+1]);
            iarg += 2;
        }
        else if (strcmp(arg[iarg],"y") == 0) {
            if (iarg+2 > narg) error->all(FLERR,"Illegal set command");
            for (ibody=0; ibody < nbody; ibody++) xcm[ibody][1] = force->numeric(FLERR,arg[iarg+1]);
            iarg += 2;
        }
        else if (strcmp(arg[iarg],"z") == 0) {
            if (iarg+2 > narg) error->all(FLERR,"Illegal set command");
            for (ibody=0; ibody < nbody; ibody++) xcm[ibody][2] = force->numeric(FLERR,arg[iarg+1]);
            iarg += 2;
        }
        else if (strcmp(arg[iarg],"vx") == 0) {
            if (iarg+2 > narg) error->all(FLERR,"Illegal set command");
            for (ibody=0; ibody < nbody; ibody++) vcm[ibody][0] = force->numeric(FLERR,arg[iarg+1]);
            iarg += 2;
        }
        else if (strcmp(arg[iarg],"vy") == 0) {
            if (iarg+2 > narg) error->all(FLERR,"Illegal set command");
            for (ibody=0; ibody < nbody; ibody++) vcm[ibody][1] = force->numeric(FLERR,arg[iarg+1]);
            iarg += 2;
        }
        else if (strcmp(arg[iarg],"vz") == 0) {
            if (iarg+2 > narg) error->all(FLERR,"Illegal set command");
            for (ibody=0; ibody < nbody; ibody++) vcm[ibody][2] = force->numeric(FLERR,arg[iarg+1]);
            iarg += 2;
        }
        else if (strcmp(arg[iarg],"omegax") == 0) {
            if (iarg+2 > narg) error->all(FLERR,"Illegal set command");
            for (ibody=0; ibody < nbody; ibody++) omega[ibody][0] = force->numeric(FLERR,arg[iarg+1]);
            iarg += 2;
        }
        else if (strcmp(arg[iarg],"omegay") == 0) {
            if (iarg+2 > narg) error->all(FLERR,"Illegal set command");
            for (ibody=0; ibody < nbody; ibody++) omega[ibody][1] = force->numeric(FLERR,arg[iarg+1]);
            iarg += 2;
        }
        else if (strcmp(arg[iarg],"omegaz") == 0) {
            if (iarg+2 > narg) error->all(FLERR,"Illegal set command");
            for (ibody=0; ibody < nbody; ibody++) omega[ibody][2] = force->numeric(FLERR,arg[iarg+1]);
            iarg += 2;
        }

        else if (strcmp(arg[iarg],"quat") == 0) {
            if (iarg+5 > narg) error->all(FLERR,"Illegal set command");
	    for (ibody=0; ibody < nbody; ibody++) {
		double nx = force->numeric(FLERR,arg[iarg+1]);
		double ny = force->numeric(FLERR,arg[iarg+2]);
		double nz = force->numeric(FLERR,arg[iarg+3]);
		double alpha = force->numeric(FLERR,arg[iarg+4]);
		double theta2 = MathConst::MY_PI2 * alpha/180.0;
		double sintheta2 = sin(theta2);
		quat[ibody][0] = cos(theta2);
		quat[ibody][1] = nx * sintheta2;
		quat[ibody][2] = ny * sintheta2;
		quat[ibody][3] = nz * sintheta2;
		MathExtra::q_to_exyz(quat[ibody], ex_space[ibody],ey_space[ibody],ez_space[ibody]);
	    }
            iarg += 5;
        } else if (strcmp(arg[iarg],"inertia") == 0) {
          if (iarg+4 > narg) error->all(FLERR,"Illegal set command");
          if (!strcmp(arg[iarg+1],"NULL") == 0) {
            for (ibody=0; ibody < nbody; ibody++) inertia[ibody][0] = force->numeric(FLERR,arg[iarg+1]);
          }
          if (!strcmp(arg[iarg+2],"NULL") == 0) {
            for (ibody=0; ibody < nbody; ibody++) inertia[ibody][1] = force->numeric(FLERR,arg[iarg+2]);
          }
          if (!strcmp(arg[iarg+3],"NULL") == 0) {
            for (ibody=0; ibody < nbody; ibody++) inertia[ibody][2] = force->numeric(FLERR,arg[iarg+3]);
          }
          iarg += 4;
        } else
            error->all(FLERR,"Illegal set/multisphere command");
    }

    static_cast<FixMultisphere*>(modify->find_fix_style("multisphere",0))->set_xv();
}
