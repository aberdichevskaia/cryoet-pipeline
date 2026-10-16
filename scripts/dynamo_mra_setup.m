% Make one Dynamo multireference (MRA) project in an export folder of
% scripts/export_dynamo.py (particles/, initial.tbl), 5 iterations, ready to run.
%
% Set the project's name and one starting reference per channel, then evaluate
% this file from the export folder; in a shell (as scripts/dynamo_mra_mdc.sbatch
% does):
%   dynamo "project = 'mra_nss'; seeds = {'subset', 'subset', 'noise'}; cores = 32; eval(fileread('<repo>/scripts/dynamo_mra_setup.m'))"
% or the same line without `dynamo "..."` in the Dynamo console. Not as
% `dynamo <file>`: that joins the lines with semicolons and loses the variables.
%
% Kinds of starting reference:
%   'subset'  the average of a random subset of the particles at their exported
%             (pytom) poses
%   'blob'    the average of a random subset at random orientations: a
%             featureless blob with the particles' radial profile
%   'noise'   white noise
% Every channel starts with the whole table; mra 1 lets particles swap. Channel k
% averages the k-th block of `subset` particles of one seeded random order, so the
% subsets of a project are disjoint and channel k takes the same particles in
% every project; the random orientations and the noise are the same in every
% project too. Optional: cores (default 8, the job's cores on a cluster),
% mask_radius (default 18 px) and mask_edge (default 2 px).
%
% It writes <project>_seeds/ (the seeds, seed_manifest.txt with tag and channel,
% seed_kinds.txt for the classification report), <project>_fmasks/,
% <project>_tables/ and the project <project>, then checks and unfolds it. It
% refuses to start if any of them exists. The mask, mra_mask_r<radius>_e<edge>.em,
% is shared by every project that asks for the same one.
%
% The search card is the one of refine223 and the CZII MRA arms, made for
% ribosomes at bin8 (9.784 A/px, 48 px box); check it before any other data.
% Loops and ifs stay on one line and no comment trails code, so the lines can
% also be pasted into the console one by one.

if ~exist('project', 'var') || ~exist('seeds', 'var'), error('dynamo_mra_setup: set project and seeds first, e.g. project = ''mra_nss''; seeds = {''subset'', ''subset'', ''noise''};'); end
if ~exist('cores', 'var'), cores = 8; end
if ~exist('mask_radius', 'var'), mask_radius = 18; end
if ~exist('mask_edge', 'var'), mask_edge = 2; end
% Particles averaged into each 'subset' or 'blob' seed, and the one seed for the
% random order, orientations and noise of every project.
subset = 300;
rng_seed = 20261016;
nref = numel(seeds);

if ~ischar(project) || ~isvarname(project), error('dynamo_mra_setup: project must be a plain name such as mra_nss'); end
if ~iscellstr(seeds) || nref < 2 || ~all(ismember(seeds, {'subset', 'blob', 'noise'})), error('dynamo_mra_setup: seeds must list at least two of ''subset'', ''blob'' and ''noise'''); end
if exist([project '_seeds'], 'dir') || exist([project '_fmasks'], 'dir') || exist([project '_tables'], 'dir') || exist(project, 'dir') || exist([project '.vpr'], 'file'), error('dynamo_mra_setup: an earlier run left %s_seeds, %s_fmasks, %s_tables or %s here; move them away first', project, project, project, project); end

t = dread('initial.tbl'); n = size(t, 1);
files = dir('particles/particle_*'); if isempty(files), error('dynamo_mra_setup: no particles/particle_* here; run it from the export folder'); end
box = size(dread(fullfile('particles', files(1).name)), 1);
if nref * subset > n, error('dynamo_mra_setup: %d particles cannot give %d disjoint subsets of %d', n, nref, subset); end

% Soft sphere about the particles' centre, (N + 1) / 2 in Dynamo's one-based
% voxels, where the export puts them: ones out to mask_radius, then a soft edge
% of mask_edge voxels (18 and 2: 0.5 at 19.7 px, 385 A across at bin8, the radius
% of the picking template's mask). The same mask aligns, classifies and smooths,
% as in the CZII MRA arms.
mask_file = sprintf('mra_mask_r%g_e%g.em', mask_radius, mask_edge);
if ~exist(mask_file, 'file'), dwrite(dynamo_sphere(mask_radius, box, (box + 1) / 2 * [1 1 1], mask_edge), mask_file); end

rng(rng_seed); p = randperm(n); random_angles = [360 * rand(n, 1), acosd(2 * rand(n, 1) - 1), 360 * rand(n, 1)]; noise = randn(box, box, box, max(nref, 3));
group = zeros(n, 1); for k = 1:nref, if ~strcmp(seeds{k}, 'noise'), group(p((k - 1) * subset + 1:k * subset)) = k; end; end
mkdir([project '_seeds']); mkdir([project '_fmasks']); mkdir([project '_tables']);

% 'subset' and 'blob' seeds are averaged as the MRA averages a class: the
% table's wedges, Fourier compensation with fmin 1, normalised under the mask;
% their fmasks are the binarised Fourier weights of those averages. A blob keeps
% the table's shifts, so each particle sits within 1.7 px of its centre. A noise
% seed has a full Fourier mask. In a test on 49 ribo-mdc particles a white-noise
% seed got no particle, and Dynamo drops a channel left without particles for good.
for k = 1:nref, seed_file = sprintf('%s_seeds/template_initial_ref_%03d.em', project, k); fmask_file = sprintf('%s_fmasks/fmask_initial_ref_%03d.em', project, k); if strcmp(seeds{k}, 'noise'), dwrite(noise(:, :, :, k), seed_file); dwrite(ones(box, box, box), fmask_file); fprintf('seed %d: noise\n', k); else, tk = t; if strcmp(seeds{k}, 'blob'), tk(:, 7:9) = random_angles; end; a = dynamo_average('particles', 'fmask', 'table', 'table', tk, 'tags', t(group == k, 1)', 'fmin', 1, 'fcompensate', true, 'verbosity', 0, 'nmask', mask_file); dwrite(a.average, seed_file); dwrite(a.fmask, fmask_file); fprintf('seed %d: %s of %d particles\n', k, seeds{k}, sum(group == k)); end; end
dlmwrite(sprintf('%s_seeds/seed_manifest.txt', project), [t(:, 1) group], ' ');
fid = fopen(sprintf('%s_seeds/seed_kinds.txt', project), 'w'); for k = 1:nref, fprintf(fid, '%d %s %d\n', k, seeds{k}, sum(group == k)); end; fclose(fid);

% Every channel starts from the whole table. With nref > 1 the project accepts
% only folders for tables, templates and fmasks.
for k = 1:nref, dwrite(t, sprintf('%s_tables/table_initial_ref_%03d.tbl', project, k)); end

dynamo_vpr(project);
dynamo_vpr_put(project, '-data', 'particles');
dynamo_vpr_put(project, '-nref', num2str(nref));
dynamo_vpr_put(project, '-mra', '1');
dynamo_vpr_put(project, '-template', [project '_seeds']);
dynamo_vpr_put(project, '-table', [project '_tables']);
dynamo_vpr_put(project, '-fmask', [project '_fmasks']);
dynamo_vpr_put(project, '-mask', mask_file);
dynamo_vpr_put(project, '-cmask', mask_file);
dynamo_vpr_put(project, '-smask', mask_file);

% Cone and in-plane 15/5, two refinement levels with factor 2, band 2..16
% (235..29 A at 9.784 A/px), shifts within 3 px of the box centre. limm 1, not 2:
% under limm 2 Dynamo's spline shift crashed the CZII run.
dynamo_vpr_put(project, '-ite_r1', '5');
dynamo_vpr_put(project, '-ite_r2', '0');
dynamo_vpr_put(project, '-ite_r3', '0');
dynamo_vpr_put(project, '-cr_r1', '15');
dynamo_vpr_put(project, '-cs_r1', '5');
dynamo_vpr_put(project, '-ir_r1', '15');
dynamo_vpr_put(project, '-is_r1', '5');
dynamo_vpr_put(project, '-rf_r1', '2');
dynamo_vpr_put(project, '-rff_r1', '2');
dynamo_vpr_put(project, '-high_r1', '2');
dynamo_vpr_put(project, '-low_r1', '16');
dynamo_vpr_put(project, '-dim_r1', num2str(box));
dynamo_vpr_put(project, '-lim_r1', '[3,3,3]');
dynamo_vpr_put(project, '-limm_r1', '1');
dynamo_vpr_put(project, '-sym_r1', 'c1');
dynamo_vpr_put(project, '-thrm', '0');
dynamo_vpr_put(project, '-destination', 'standalone');
dynamo_vpr_put(project, '-cores', num2str(cores));

dynamo_vpr_check(project);
dynamo_vpr_unfold(project);
fprintf('dynamo_mra_setup: project %s is ready, seeds %s, %d cores\n', project, strjoin(seeds, ' '), cores);
clear project seeds cores mask_radius mask_edge
