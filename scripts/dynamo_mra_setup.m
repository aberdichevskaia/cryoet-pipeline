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
% schedule picks the search:
%   'fixed'           (default) one round of 5 iterations: band 2..16 Fourier px,
%                     cone and in-plane 15 deg in 5 deg steps, as refine223 and the
%                     CZII MRA arms searched
%   'coarse_to_fine'  5 rounds of 1 iteration, the low-pass moving in equal steps
%                     of Fourier px from first_lowpass (default 40 A) to
%                     last_lowpass (default 22 A: short of Nyquist, where a search
%                     without independent halves fits noise most easily), the
%                     angular step following it as atan(resolution / diameter)
%                     with diameter the particle's (default 290 A, the 80S), and
%                     cone and in-plane 3 steps wide, as the fixed card is
% Every round is given every search parameter, nref and mra included: a parameter
% set without a round goes to round 1 only.
%
% It writes <project>_seeds/ (the seeds, seed_manifest.txt with tag and channel,
% seed_kinds.txt for the classification report), <project>_fmasks/,
% <project>_tables/ and the project <project>, then checks and unfolds it. It
% refuses to start if any of them exists. The mask, mra_mask_r<radius>_e<edge>.em,
% is shared by every project that asks for the same one.
%
% Both searches were set for ribosomes at bin8 (9.784 A/px, 48 px box); check
% them before any other data.
% Loops and ifs stay on one line and no comment trails code, so the lines can
% also be pasted into the console one by one.

if ~exist('project', 'var') || ~exist('seeds', 'var'), error('dynamo_mra_setup: set project and seeds first, e.g. project = ''mra_nss''; seeds = {''subset'', ''subset'', ''noise''};'); end
if ~exist('cores', 'var'), cores = 8; end
if ~exist('mask_radius', 'var'), mask_radius = 18; end
if ~exist('mask_edge', 'var'), mask_edge = 2; end
if ~exist('schedule', 'var'), schedule = 'fixed'; end
if ~exist('first_lowpass', 'var'), first_lowpass = 40; end
if ~exist('last_lowpass', 'var'), last_lowpass = 22; end
if ~exist('diameter', 'var'), diameter = 290; end
% Particles averaged into each 'subset' or 'blob' seed, and the one seed for the
% random order, orientations and noise of every project.
subset = 300;
rng_seed = 20261016;
nref = numel(seeds);

if ~ischar(project) || ~isvarname(project), error('dynamo_mra_setup: project must be a plain name such as mra_nss'); end
if ~iscellstr(seeds) || nref < 2 || ~all(ismember(seeds, {'subset', 'blob', 'noise'})), error('dynamo_mra_setup: seeds must list at least two of ''subset'', ''blob'' and ''noise'''); end
if ~ischar(schedule) || ~any(strcmp(schedule, {'fixed', 'coarse_to_fine'})), error('dynamo_mra_setup: schedule must be ''fixed'' or ''coarse_to_fine'''); end
if exist([project '_seeds'], 'dir') || exist([project '_fmasks'], 'dir') || exist([project '_tables'], 'dir') || exist(project, 'dir') || exist([project '.vpr'], 'file'), error('dynamo_mra_setup: an earlier run left %s_seeds, %s_fmasks, %s_tables or %s here; move them away first', project, project, project, project); end

t = dread('initial.tbl'); n = size(t, 1);
files = dir('particles/particle_*'); if isempty(files), error('dynamo_mra_setup: no particles/particle_* here; run it from the export folder'); end
box = size(dread(fullfile('particles', files(1).name)), 1);
apix = t(1, 36); if ~(apix > 0), error('dynamo_mra_setup: column 36 of initial.tbl holds no pixel size'); end
if strcmp(schedule, 'coarse_to_fine') && ~(2 * apix <= last_lowpass && last_lowpass < first_lowpass), error('dynamo_mra_setup: last_lowpass must lie between Nyquist (%.1f A) and first_lowpass (%g A)', 2 * apix, first_lowpass); end
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

% One Dynamo round per stage of the search. In every round: cone and in-plane 3
% steps wide (Dynamo's cone_range is the cone's full opening), two refinement
% levels with factor 2, high-pass at Fourier px 2, shifts within 3 px of the box
% centre (limm 1, not 2: under limm 2 Dynamo's spline shift crashed the CZII run),
% C1, no thresholding. The fixed card gives band 2..16 (235..29 A at 9.784 A/px)
% and 5 deg steps; coarse_to_fine from 40 to 22 A at 9.784 A/px and 290 A gives
% low-passes of 12, 14, 17, 19 and 21 px (39, 34, 28, 25 and 22 A) and steps of
% 7.7, 6.6, 5.4, 4.9 and 4.4 deg.
if strcmp(schedule, 'fixed'), iterations = 5; lows = 16; steps = 5; else, lows = round(linspace(box * apix / first_lowpass, box * apix / last_lowpass, 5)); iterations = ones(size(lows)); steps = round(10 * atand(box * apix ./ lows / diameter)) / 10; end
rounds = numel(lows);
for r = 1:rounds, s = sprintf('_r%d', r); dynamo_vpr_put(project, ['-ite' s], num2str(iterations(r))); dynamo_vpr_put(project, ['-nref' s], num2str(nref)); dynamo_vpr_put(project, ['-mra' s], '1'); dynamo_vpr_put(project, ['-thrm' s], '0'); end
for r = 1:rounds, s = sprintf('_r%d', r); dynamo_vpr_put(project, ['-cr' s], num2str(3 * steps(r))); dynamo_vpr_put(project, ['-cs' s], num2str(steps(r))); dynamo_vpr_put(project, ['-ir' s], num2str(3 * steps(r))); dynamo_vpr_put(project, ['-is' s], num2str(steps(r))); end
for r = 1:rounds, s = sprintf('_r%d', r); dynamo_vpr_put(project, ['-rf' s], '2'); dynamo_vpr_put(project, ['-rff' s], '2'); dynamo_vpr_put(project, ['-high' s], '2'); dynamo_vpr_put(project, ['-low' s], num2str(lows(r))); dynamo_vpr_put(project, ['-dim' s], num2str(box)); end
for r = 1:rounds, s = sprintf('_r%d', r); dynamo_vpr_put(project, ['-lim' s], '[3,3,3]'); dynamo_vpr_put(project, ['-limm' s], '1'); dynamo_vpr_put(project, ['-sym' s], 'c1'); end
for r = rounds + 1:8, dynamo_vpr_put(project, sprintf('-ite_r%d', r), '0'); end
for r = 1:rounds, fprintf('round %d: %d iteration(s), low-pass %d px (%.1f A), cone and in-plane %.1f deg in %.1f deg steps\n', r, iterations(r), lows(r), box * apix / lows(r), 3 * steps(r), steps(r)); end
dynamo_vpr_put(project, '-destination', 'standalone');
dynamo_vpr_put(project, '-cores', num2str(cores));

dynamo_vpr_check(project);
dynamo_vpr_unfold(project);
fprintf('dynamo_mra_setup: project %s is ready, seeds %s, schedule %s, %d cores\n', project, strjoin(seeds, ' '), schedule, cores);
clear project seeds cores mask_radius mask_edge schedule first_lowpass last_lowpass diameter
