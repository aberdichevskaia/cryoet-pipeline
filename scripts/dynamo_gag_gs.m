% The Dynamo branch of the Gag average, set up as the pipeline branch runs:
% the same particles (dynamo/export-membrane), the same mask file, the same
% search, and two independent halves started from one average band-passed to
% [1 6 2], which is the pipeline's START_LOWPASS_ANGSTROM=57.6 at 64 px of
% 5.4 A. The recipe is sel45_gs_eo's (outputs/mra-pick-qc/SEL45_DYNAMO_SESSION.md,
% step 4).
%
% In the Dynamo console ("$D/bin/run_dynamo.sh" "$D"), from
% outputs/hiv/dynamo/export-membrane, one line at a time: the 1.1.591 console
% breaks on a line holding two commands and refuses '@'. Then compute
% gag_half_eo with scripts/dynamo_project.sbatch.
%
% Created with dcp.new, as sel45_gs was. A bare `dynamo_vpr` project leaves the
% initial and classification masks unset, and dynamo_vpr_eo refuses it
% ("contains errors": file_fmask_initial, file_mask_classification,
% folder_data), which is how the first attempt, gag_gs, ended on 2026-10-09.

t = dread('initial.tbl');
a = dynamo_average('particles','fmask','table','table',t,'fmin',1,'fcompensate',true,'nmask','../mask_sphere_250A_box64.mrc','verbosity',1);
dwrite(a.average,'gag_avg.em');
v = dynamo_bandpass(a.average,[1 6 2]);
dwrite(v,'gag_avg_lp58.em');
dcp.new('gag_half','data','particles','table','initial.tbl','template','gag_avg_lp58.em','mask','../mask_sphere_250A_box64.mrc','show',0);
dynamo_vpr_put gag_half -ite 3 -dim 64 -cr 15 -cs 5 -ir 15 -is 5 -rf 2 -rff 2 -high 2 -low 16 -lim 3 -limm 1 -sym c1 -mra 0 -apix 5.4 -dst standalone -cores 16
dynamo_vpr_check gag_half
dynamo_vpr_eo('gag_half')
dynamo_vpr_unfold gag_half_eo
