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
% diameter (A, the particle's widest; default 290, the 80S), mask_radius (px;
% default the particle's radius plus 3 px, round(diameter / 2 / pixel + 3): 18
% for the 80S at bin8, 16 for RyR1 at bin8) and mask_edge (default 2 px), or
% mask_file, a mask of any shape in the particles' box that replaces the sphere
% (scripts/make_mra_mask.py makes one that leaves a membrane out), and symmetry
% (default 'c1'), imposed about the box's z axis in every round.
%
% Dynamo imposes symmetry about the z axis through voxel N/2 (zero-based), and
% the export centres particles on (N - 1) / 2: the RyR1 template's and the
% starting-pose average's C4 axes sat at centre + (-0.45, -0.45) px (2026-10-20,
% scripts/rotational_symmetry.py --order 4). axis_offset ([dx dy] or [dx dy dz]
% px, as that script prints it; default none) moves every particle so that an
% axis found there lands on Dynamo's: each particle's shift grows by the offset
% turned into its own frame, before the seeds and tables are made.
%
% schedule picks the search:
%   'fixed'           (default) one round of 5 iterations: band 2..16 Fourier px,
%                     cone and in-plane 15 deg in 5 deg steps, as refine223 and the
%                     CZII MRA arms searched
%   'coarse_to_fine'  5 rounds of 1 iteration, the low-pass moving in equal steps
%                     from first_lowpass (default 40 A) to last_lowpass (default
%                     22 A: short of Nyquist, where a search without independent
%                     halves fits noise most easily), the angular step following
%                     it as atan(resolution / diameter), and cone and in-plane 3
%                     steps wide, as the fixed card is. lowpass_steps says equal
%                     in what: 'angstrom' (default since 2026-10-19) in
%                     resolution; 'fourier' in Fourier px, i.e. frequency, as the
%                     ribo-mdc runs of 2026-10-17 were made. Both are rounded to
%                     whole Fourier px, which is what Dynamo takes
% global_rounds (default 0) makes the first rounds search every orientation
% instead of a cone: cone and in-plane 360 deg in global_step (default 45) deg
% steps, refined by factor 2 until the step is no coarser than that round's
% atan(resolution / diameter), the way dynamo_align's own global search refines
% its 60 deg grid. For blob seeds, which carry no orientation to search near
% (the PI, 2026-10-22: bbn with a 360 deg search, ssn with the cone, or ssn with
% 360 deg for two rounds and then the cone). The grid sets the cost: 45 deg
% refined three times is 4.3 times round 1's cone search per particle and
% channel, a 9 deg grid would be 340 times (counted with dynamo_search.angle_list).
% Every round is given every search parameter, nref and mra included: a parameter
% set without a round goes to round 1 only.
%
% It writes <project>_seeds/ (the seeds, seed_manifest.txt with tag and channel,
% seed_kinds.txt for the classification report), <project>_fmasks/,
% <project>_tables/ and the project <project>, then checks and unfolds it. It
% refuses to start if any of them exists. The sphere mask,
% mra_mask_r<radius>_e<edge>.em, is shared by every project that asks for the
% same one; a mask_file is only read.
%
% Both searches were set for ribosomes at bin8 (9.784 A/px, 48 px box); check
% them before any other data.
% Loops and ifs stay on one line and no comment trails code, so the lines can
% also be pasted into the console one by one.

if ~exist('project', 'var') || ~exist('seeds', 'var'), error('dynamo_mra_setup: set project and seeds first, e.g. project = ''mra_nss''; seeds = {''subset'', ''subset'', ''noise''};'); end
if ~exist('cores', 'var'), cores = 8; end
if ~exist('mask_radius', 'var'), mask_radius = []; end
if ~exist('mask_edge', 'var'), mask_edge = 2; end
if ~exist('schedule', 'var'), schedule = 'fixed'; end
if ~exist('first_lowpass', 'var'), first_lowpass = 40; end
if ~exist('last_lowpass', 'var'), last_lowpass = 22; end
if ~exist('diameter', 'var'), diameter = 290; end
if ~exist('lowpass_steps', 'var'), lowpass_steps = 'angstrom'; end
if ~exist('mask_file', 'var'), mask_file = ''; end
if ~exist('symmetry', 'var'), symmetry = 'c1'; end
if ~exist('axis_offset', 'var'), axis_offset = [0 0 0]; end
if ~exist('global_rounds', 'var'), global_rounds = 0; end
if ~exist('global_step', 'var'), global_step = 45; end
% Particles averaged into each 'subset' or 'blob' seed, and the one seed for the
% random order, orientations and noise of every project.
subset = 300;
rng_seed = 20261016;
nref = numel(seeds);

if ~ischar(project) || ~isvarname(project), error('dynamo_mra_setup: project must be a plain name such as mra_nss'); end
if ~iscellstr(seeds) || nref < 2 || ~all(ismember(seeds, {'subset', 'blob', 'noise'})), error('dynamo_mra_setup: seeds must list at least two of ''subset'', ''blob'' and ''noise'''); end
if ~ischar(schedule) || ~any(strcmp(schedule, {'fixed', 'coarse_to_fine'})), error('dynamo_mra_setup: schedule must be ''fixed'' or ''coarse_to_fine'''); end
if ~ischar(lowpass_steps) || ~any(strcmp(lowpass_steps, {'fourier', 'angstrom'})), error('dynamo_mra_setup: lowpass_steps must be ''fourier'' or ''angstrom'''); end
if ~ischar(symmetry) || isempty(regexp(symmetry, '^[cd][0-9]+$', 'once')), error('dynamo_mra_setup: symmetry must be like ''c1'' or ''c4'''); end
if ~ischar(mask_file), error('dynamo_mra_setup: mask_file must be a file name'); end
if ~isnumeric(axis_offset) || ~any(numel(axis_offset) == [2 3]) || ~all(isfinite(axis_offset)), error('dynamo_mra_setup: axis_offset must be [dx dy] or [dx dy dz] in px, as scripts/rotational_symmetry.py prints it'); end
axis_offset = [axis_offset(:)' zeros(1, 3 - numel(axis_offset))];
if ~isscalar(global_rounds) || ~any(global_rounds == 0:(1 + 4 * strcmp(schedule, 'coarse_to_fine'))), error('dynamo_mra_setup: global_rounds must be a whole number from 0 to the schedule''s rounds (5 for coarse_to_fine, 1 for fixed)'); end
if ~isscalar(global_step) || ~(global_step > 0 && global_step <= 180), error('dynamo_mra_setup: global_step must be an angle in degrees, above 0 and at most 180'); end
if exist([project '_seeds'], 'dir') || exist([project '_fmasks'], 'dir') || exist([project '_tables'], 'dir') || exist(project, 'dir') || exist([project '.vpr'], 'file'), error('dynamo_mra_setup: an earlier run left %s_seeds, %s_fmasks, %s_tables or %s here; move them away first', project, project, project, project); end

t = dread('initial.tbl'); n = size(t, 1);
% A particle is its reference turned by its angles and shifted by its shift, so
% a point at centre + axis_offset of the reference sits at its shift plus the
% offset turned by those angles: that sum becomes the shift. dynamo_rot turns
% content by the transpose of dynamo_euler2matrix: on synthetic particles made
% with dynamo_rot (2026-10-20) d + R'o gave the reference moved by -o at
% correlation 0.9994, d + Ro a blur of it at 0.978.
if any(axis_offset), for i = 1:n, t(i, 4:6) = t(i, 4:6) + (dynamo_euler2matrix(t(i, 7:9))' * axis_offset(:))'; end; end
files = dir('particles/particle_*'); if isempty(files), error('dynamo_mra_setup: no particles/particle_* here; run it from the export folder'); end
box = size(dread(fullfile('particles', files(1).name)), 1);
apix = t(1, 36); if ~(apix > 0), error('dynamo_mra_setup: column 36 of initial.tbl holds no pixel size'); end
if strcmp(schedule, 'coarse_to_fine') && ~(2 * apix <= last_lowpass && last_lowpass < first_lowpass), error('dynamo_mra_setup: last_lowpass must lie between Nyquist (%.1f A) and first_lowpass (%g A)', 2 * apix, first_lowpass); end
if nref * subset > n, error('dynamo_mra_setup: %d particles cannot give %d disjoint subsets of %d', n, nref, subset); end
% The rule for every mask (Anna, 2026-10-16): the particle's radius where it is
% widest, plus 3 px, then a soft edge. 290 A at 9.784 A/px gives the 18 px the
% ribo-mdc projects were made with; 340 A at 13.44 A/px gives 16 for RyR1.
if isempty(mask_file) && isempty(mask_radius), mask_radius = round(diameter / 2 / apix + 3); end
if isempty(mask_file) && mask_radius + mask_edge > box / 2, error('dynamo_mra_setup: a mask of %g px with a %g px edge does not fit the %d px box; give a smaller mask_radius or diameter', mask_radius, mask_edge, box); end
if ~isempty(mask_file) && ~exist(mask_file, 'file'), error('dynamo_mra_setup: no mask_file %s', mask_file); end
if ~isempty(mask_file) && ~isequal(size(dread(mask_file)), [box box box]), error('dynamo_mra_setup: mask_file %s is not a %d px box like the particles', mask_file, box); end

% Soft sphere about the particles' centre, (N + 1) / 2 in Dynamo's one-based
% voxels, where the export puts them: ones out to mask_radius, then a soft edge
% of mask_edge voxels (18 and 2: 0.5 at 19.7 px, 385 A across at bin8 for the
% 80S). The same mask aligns, classifies and smooths, as in the CZII MRA arms.
% A mask_file given instead is used as it is, for all three.
if isempty(mask_file), mask_file = sprintf('mra_mask_r%g_e%g.em', mask_radius, mask_edge); mask_note = sprintf('sphere %g px with a %g px edge', mask_radius, mask_edge); else, mask_note = mask_file; end
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

% One Dynamo round per stage of the search. In every round but the global ones:
% cone and in-plane 3 steps wide (Dynamo's cone_range is the cone's full
% opening), two refinement levels with factor 2; in a global round 360 deg in
% global_step steps and as many levels as reach the round's step. In every round:
% high-pass at Fourier px 2, shifts within 3 px of the box
% centre (limm 1, not 2: under limm 2 Dynamo's spline shift crashed the CZII run),
% the symmetry (C1 unless given), no thresholding. The fixed card gives band
% 2..16 (235..29 A at 9.784 A/px) and 5 deg steps; coarse_to_fine from 40 to
% 22 A at 9.784 A/px and 290 A gives
% low-passes of 12, 14, 17, 19 and 21 px (39, 34, 28, 25 and 22 A) and steps of
% 7.7, 6.6, 5.4, 4.9 and 4.4 deg. With lowpass_steps 'angstrom' the low-passes
% are equal steps of resolution instead, rounded to the same whole Fourier px.
if strcmp(schedule, 'fixed'), iterations = 5; lows = 16; steps = 5; elseif strcmp(lowpass_steps, 'angstrom'), lows = round(box * apix ./ linspace(first_lowpass, last_lowpass, 5)); iterations = ones(size(lows)); steps = round(10 * atand(box * apix ./ lows / diameter)) / 10; else, lows = round(linspace(box * apix / first_lowpass, box * apix / last_lowpass, 5)); iterations = ones(size(lows)); steps = round(10 * atand(box * apix ./ lows / diameter)) / 10; end
rounds = numel(lows);
cone_ranges = 3 * steps; samplings = steps; refines = 2 * ones(size(steps));
if global_rounds > 0, g = 1:global_rounds; cone_ranges(g) = 360; samplings(g) = global_step; refines(g) = max(2, ceil(log2(global_step ./ steps(g)))); end
for r = 1:rounds, s = sprintf('_r%d', r); dynamo_vpr_put(project, ['-ite' s], num2str(iterations(r))); dynamo_vpr_put(project, ['-nref' s], num2str(nref)); dynamo_vpr_put(project, ['-mra' s], '1'); dynamo_vpr_put(project, ['-thrm' s], '0'); end
for r = 1:rounds, s = sprintf('_r%d', r); dynamo_vpr_put(project, ['-cr' s], num2str(cone_ranges(r))); dynamo_vpr_put(project, ['-cs' s], num2str(samplings(r))); dynamo_vpr_put(project, ['-ir' s], num2str(cone_ranges(r))); dynamo_vpr_put(project, ['-is' s], num2str(samplings(r))); end
for r = 1:rounds, s = sprintf('_r%d', r); dynamo_vpr_put(project, ['-rf' s], num2str(refines(r))); dynamo_vpr_put(project, ['-rff' s], '2'); dynamo_vpr_put(project, ['-high' s], '2'); dynamo_vpr_put(project, ['-low' s], num2str(lows(r))); dynamo_vpr_put(project, ['-dim' s], num2str(box)); end
for r = 1:rounds, s = sprintf('_r%d', r); dynamo_vpr_put(project, ['-lim' s], '[3,3,3]'); dynamo_vpr_put(project, ['-limm' s], '1'); dynamo_vpr_put(project, ['-sym' s], symmetry); end
for r = rounds + 1:8, dynamo_vpr_put(project, sprintf('-ite_r%d', r), '0'); end
for r = 1:rounds, if r <= global_rounds, search = sprintf('every orientation, 360 deg in %g deg steps refined %d times by 2 to %.1f deg', samplings(r), refines(r), samplings(r) / 2^refines(r)); else, search = sprintf('cone and in-plane %.1f deg in %.1f deg steps', cone_ranges(r), samplings(r)); end; fprintf('round %d: %d iteration(s), low-pass %d px (%.1f A), %s\n', r, iterations(r), lows(r), box * apix / lows(r), search); end
dynamo_vpr_put(project, '-destination', 'standalone');
dynamo_vpr_put(project, '-cores', num2str(cores));

dynamo_vpr_check(project);
dynamo_vpr_unfold(project);
fprintf('dynamo_mra_setup: project %s is ready, seeds %s, schedule %s (%s steps), %d global round(s) in %g deg steps, mask %s, symmetry %s, axis moved from centre + [%g %g %g] px, %d cores\n', project, strjoin(seeds, ' '), schedule, lowpass_steps, global_rounds, global_step, mask_note, symmetry, axis_offset, cores);
clear project seeds cores mask_radius mask_edge mask_file mask_note symmetry axis_offset global_rounds global_step schedule first_lowpass last_lowpass diameter lowpass_steps
