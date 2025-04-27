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

#include <string.h>
#include "compute_unbalanced_atom.h"
#include "atom.h"
#include "update.h"
#include "modify.h"
#include "comm.h"
#include "fix_multisphere.h" 
#include "force.h"
#include "memory.h"
#include "error.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

ComputeUnbalancedAtom::ComputeUnbalancedAtom(LAMMPS *lmp, int &iarg, int narg, char **arg) :
  Compute(lmp, iarg, narg, arg)
{
  if (narg != iarg) error->all(FLERR,"Illegal compute unbalanced/atom command");

  peratom_flag = 1;
  size_peratom_cols = 0;

  nmax = 0;
  unbalanced = NULL;
  fix_ms = NULL; 
}

/* ---------------------------------------------------------------------- */

ComputeUnbalancedAtom::~ComputeUnbalancedAtom()
{
  memory->destroy(unbalanced);
}

/* ---------------------------------------------------------------------- */

void ComputeUnbalancedAtom::init()
{
  int count = 0;
  for (int i = 0; i < modify->ncompute; i++)
    if (strcmp(modify->compute[i]->style,"unbalanced/atom") == 0) count++;
  if (count > 1 && comm->me == 0)
    error->warning(FLERR,"More than one compute unbalanced/atom");
  fix_ms =  static_cast<FixMultisphere*>(modify->find_fix_style("multisphere",0)); 
}

/* ---------------------------------------------------------------------- */

void ComputeUnbalancedAtom::compute_peratom()
{
  invoked_peratom = update->ntimestep;

  // grow unbalanced array if necessary

  if (atom->nlocal > nmax) {
    memory->destroy(unbalanced);
    nmax = atom->nmax;
    memory->create(unbalanced,nmax,"unbalanced/atom:unbalanced");
    vector_atom = unbalanced;
  }

  // compute kinetic energy for each atom in group

  double **f = atom->f;
  double *rmass = atom->rmass;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  if (rmass)
    for (int i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit && (!fix_ms || fix_ms->belongs_to(i) < 0)) { 
        unbalanced[i] = vectorMag3D(f[i]);
      } else unbalanced[i] = 0.0;
    }

  else
    for (int i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit) {
        unbalanced[i] = vectorMag3D(f[i]);
      } else unbalanced[i] = 0.0;
    }
}

/* ----------------------------------------------------------------------
   memory usage of local atom-based array
------------------------------------------------------------------------- */

double ComputeUnbalancedAtom::memory_usage()
{
  double bytes = nmax * sizeof(double);
  return bytes;
}
