from cpymad.madx import Madx
import xtrack as xt

madx = Madx()
madx.input('''

call,file="fccee_z_hbu_sol_on_off.seq";

option, echo, info, warn;

! set,format="-18s","11.5g";  ! my standard compact format
! set,format="-20s","13.7g";  ! my standard compact format  --  2 digits more useful for exact tune matching
  set,format="-20s","23.17g"; ! full precision

select, flag=twiss, column=name,keyword,s,betx,alfx, mux,bety,alfy,muy,x,px,y,py,t,pt,dx,dpx,dy,dpy;   ! name, and basic twiss,   order as without select

pbeam :=   45.6;
EXbeam = 5.05e-10;
EYbeam = 1.01026e-12;
Nbun :=  10000;
NPar := 243000000000.0;
sigt=0.0155; sige=0.109; ! bunch length [m] and relative energy spread with Beamstrahlung

Ebeam := sqrt( pbeam^2 + emass^2 );

beam, particle = positron, npart = Npar, kbunch=Nbun, energy = ebeam, SIGT=sigt, SIGE=sige, radiate=false, EX=EXbeam, EY=EYbeam, bv=+1;

use, sequence = fccee_p_ring;

title,"fccee_z_hbu_sol_on";
twiss, chrom, rmatrix, file = "fccee_z_hbu_sol_on.tfs";

''')

line = xt.Line.from_madx_sequence(madx.sequence.fccee_p_ring)