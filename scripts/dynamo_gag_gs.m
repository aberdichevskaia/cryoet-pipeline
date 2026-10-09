% The Dynamo branch of the Gag average, set up as the pipeline branch runs:
% the same particles (dynamo/export-membrane), the same mask file, the same
% search, and two independent halves started from one average band-passed to
% [1 6 2], which is the pipeline's START_LOWPASS_ANGSTROM=57.6 at 64 px of
% 5.4 A. The recipe is sel45_gs_eo's (outputs/mra-pick-qc/SEL45_DYNAMO_SESSION.md,
% step 4), the parameter names those the MRA arms used on the cluster.
%
% In the Dynamo console, from outputs/hiv/dynamo/export-membrane, one line at a
% time: the 1.1.591 console breaks on a line holding two commands and refuses
% '@'. Then compute gag_gs_eo with scripts/dynamo_project.sbatch.

t = dread('initial.tbl');
a = dynamo_average('particles','fmask','table','table',t,'fmin',1,'fcompensate',true,'nmask','../mask_sphere_250A_box64.mrc','verbosity',1);
dwrite(a.average,'gag_avg.em');
v = dynamo_bandpass(a.average,[1 6 2]);
dwrite(v,'gag_avg_lp58.em');
dynamo_vpr gag_gs
dynamo_vpr_put gag_gs -data particles
dynamo_vpr_put gag_gs -table initial.tbl
dynamo_vpr_put gag_gs -template gag_avg_lp58.em
dynamo_vpr_put gag_gs -mask ../mask_sphere_250A_box64.mrc
dynamo_vpr_put gag_gs -ite_r1 3
dynamo_vpr_put gag_gs -dim_r1 64
dynamo_vpr_put gag_gs -cr_r1 15
dynamo_vpr_put gag_gs -cs_r1 5
dynamo_vpr_put gag_gs -ir_r1 15
dynamo_vpr_put gag_gs -is_r1 5
dynamo_vpr_put gag_gs -rf_r1 2
dynamo_vpr_put gag_gs -rff_r1 2
dynamo_vpr_put gag_gs -high_r1 2
dynamo_vpr_put gag_gs -low_r1 16
dynamo_vpr_put gag_gs -lim_r1 [3,3,3]
dynamo_vpr_put gag_gs -limm_r1 1
dynamo_vpr_put gag_gs -sym_r1 c1
dynamo_vpr_put gag_gs -mra 0
dynamo_vpr_put gag_gs -apix 5.4
dynamo_vpr_put gag_gs -destination standalone
dynamo_vpr_put gag_gs -cores 16
dynamo_vpr_eo('gag_gs')
dynamo_vpr_check gag_gs_eo
dynamo_vpr_unfold gag_gs_eo
