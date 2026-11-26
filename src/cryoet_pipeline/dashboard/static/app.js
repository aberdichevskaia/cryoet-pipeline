"use strict";

// The page draws itself from the server's documents. Every position it draws
// (circles, slab bands) comes from the document in the picture's own pixels;
// the page only scales them to the screen.

const state = {
  overview: null,
  doc: null,
  base: "",
  images: { xy: [], xz: [] },
  index: { xy: 0, xz: 0 },
  threshold: -Infinity,
  showBelow: false,
  selected: null,
  brightness: 1,
  contrast: 1,
};

const $ = (id) => document.getElementById(id);

async function api(path) {
  const response = await fetch(path, { credentials: "same-origin" });
  const text = await response.text();
  let body;
  try { body = JSON.parse(text); } catch { body = { error: text.trim() || response.statusText }; }
  if (!response.ok) throw new Error(body.error || response.statusText);
  return body;
}

function setStatus(text, isError = false) {
  const status = $("status");
  status.textContent = text;
  status.classList.toggle("error", isError);
}

// ---- colours ---------------------------------------------------------------

const RAMP = [[56, 189, 248], [163, 230, 53], [250, 204, 21]];

function rampColour(fraction, alpha = 1) {
  const f = Math.min(1, Math.max(0, fraction)) * (RAMP.length - 1);
  const i = Math.min(RAMP.length - 2, Math.floor(f));
  const t = f - i;
  const c = RAMP[i].map((v, k) => Math.round(v + (RAMP[i + 1][k] - v) * t));
  return `rgba(${c[0]}, ${c[1]}, ${c[2]}, ${alpha})`;
}

function scoreFraction(i) {
  const p = state.doc.picks;
  const span = p.maxScore - p.minScore;
  return span > 0 ? (p.score[i] - p.minScore) / span : 1;
}

// ---- formatting --------------------------------------------------------------

const nm = (px) => (px * state.doc.tomogram.voxel_angstrom / 10).toFixed(0);

function scoreText(i) {
  const p = state.doc.picks;
  const raw = p.score[i].toFixed(4);
  return p.sigma ? `${raw} (${p.sigma[i].toFixed(1)}σ)` : raw;
}

// ---- loading -----------------------------------------------------------------

async function start() {
  try {
    state.overview = await api("/api/project");
  } catch (error) {
    setStatus(error.message, true);
    return;
  }
  const overview = state.overview;
  $("project").textContent = overview.project + (overview.variant ? ` · ${overview.variant}` : "");
  document.title = `${overview.project} · results`;
  const select = $("series");
  for (const series of overview.series) {
    const option = document.createElement("option");
    option.value = series.id;
    option.textContent = series.id + (
      series.picks ? ""
        : series.tomogram ? " · tomogram only"
          : series.problems.length ? " · could not be read" : " · no tomogram");
    select.append(option);
  }
  select.addEventListener("change", () => loadSeries(select.value));
  const wanted = decodeURIComponent(location.hash.slice(1));
  const first = overview.series.find((s) => s.id === wanted)
    || overview.series.find((s) => s.picks)
    || overview.series.find((s) => s.tomogram)
    || overview.series[0];
  if (!first) {
    setStatus("This project has no series yet.");
    return;
  }
  select.value = first.id;
  loadSeries(first.id);
}

async function loadSeries(id) {
  history.replaceState(null, "", `#${encodeURIComponent(id)}`);
  setStatus(`Reading ${id}. The first look at a series cuts its slabs, which takes a few seconds.`);
  $("viewers").hidden = true;
  $("side").hidden = true;
  let doc;
  try {
    doc = await api(`/api/series/${encodeURIComponent(id)}`);
  } catch (error) {
    setStatus(`${id}: ${error.message}`, true);
    return;
  }
  if ($("series").value !== id) return; // another series was chosen meanwhile
  state.doc = doc;
  state.selected = null;
  state.base = `/img/${encodeURIComponent(id)}/${doc.key}/`;
  showProblems(doc.problems);
  $("side").hidden = false;
  if (!doc.tomogram) {
    setStatus(doc.problems.length
      ? `${id}: no tomogram shown; what could not be read is listed below.`
      : `${id} has no tomogram yet.`);
    $("scores-panel").hidden = true;
    $("pick-panel").hidden = true;
    return;
  }
  for (const name of ["xy", "xz"]) {
    const stack = doc.tomogram.stacks[name];
    state.images[name] = Array.from({ length: stack.count }, (_, k) => {
      const image = new Image();
      image.decoding = "async";
      image.onload = () => { if (state.index[name] === k) draw(name); };
      image.src = `${state.base}${name}_${String(k).padStart(3, "0")}.png`;
      return image;
    });
    const slider = $(`${name}-slider`);
    slider.max = String(stack.count - 1);
    state.index[name] = Math.floor((stack.count - 1) / 2);
    slider.value = String(state.index[name]);
  }
  setUpPicks();
  let caption = doc.tomogram.caption;
  if (doc.picks) {
    caption += ` Circles: a sphere with ${doc.picks.radius_source} ` +
      `(${doc.picks.radius_px.toFixed(1)} voxels, ${nm(doc.picks.radius_px)} nm).`;
  }
  $("caption").textContent = caption;
  $("viewers").hidden = false;
  setStatus("");
  drawAll();
}

function showProblems(problems) {
  const list = $("problems");
  list.replaceChildren(...(problems || []).map((text) => {
    const item = document.createElement("li");
    item.textContent = text;
    return item;
  }));
  $("problems-panel").hidden = !problems || problems.length === 0;
}

function setUpPicks() {
  const doc = state.doc;
  const p = doc.picks;
  $("scores-panel").hidden = !p;
  $("pick-panel").hidden = !p;
  $("pick-figure").hidden = true;
  $("pick-details").textContent = "Click a circle, or step through the picks by score.";
  if (!p) return;
  p.minScore = Math.min(...p.score);
  p.maxScore = Math.max(...p.score);
  state.threshold = p.minScore;
  $("threshold").value = "0";
  const template = $("template-figure");
  template.hidden = !doc.template;
  if (doc.template) $("template-tile").src = `${state.base}template.png`;
  updateThreshold();
}

// ---- drawing -----------------------------------------------------------------

function visible(i) {
  return state.doc.picks.score[i] >= state.threshold;
}

function draw(name) {
  const doc = state.doc;
  if (!doc || !doc.tomogram) return;
  const stack = doc.tomogram.stacks[name];
  const viewer = $(`${name}-viewer`);
  const stage = viewer.querySelector(".stage");
  const picture = viewer.querySelector("canvas.picture");
  const overlay = viewer.querySelector("canvas.overlay");
  const cssWidth = stage.clientWidth;
  if (cssWidth === 0) return;
  const scale = cssWidth / stack.width; // screen pixels per picture pixel
  const cssHeight = stack.height * scale;
  const dpr = window.devicePixelRatio || 1;
  stage.style.height = `${cssHeight}px`;
  for (const canvas of [picture, overlay]) {
    canvas.width = Math.round(cssWidth * dpr);
    canvas.height = Math.round(cssHeight * dpr);
    canvas.style.width = `${cssWidth}px`;
    canvas.style.height = `${cssHeight}px`;
  }
  const k = state.index[name];
  const image = state.images[name][k];
  const context = picture.getContext("2d");
  if (image && image.complete && image.naturalWidth) {
    context.drawImage(image, 0, 0, picture.width, picture.height);
  }
  picture.style.filter = `brightness(${state.brightness}) contrast(${state.contrast})`;

  const ink = overlay.getContext("2d");
  ink.setTransform(dpr * scale, 0, 0, dpr * scale, 0, 0);
  ink.clearRect(0, 0, stack.width, stack.height);
  const line = 1 / scale; // one screen pixel, in picture pixels

  // Where the other view's slab cuts this one.
  const other = name === "xy" ? "xz" : "xy";
  const [top, middle, bottom] = stack.bands[state.index[other]];
  ink.fillStyle = "rgba(255, 255, 255, 0.07)";
  ink.fillRect(0, top, stack.width, bottom - top);
  ink.setLineDash([4 * line, 4 * line]);
  ink.strokeStyle = "rgba(255, 255, 255, 0.45)";
  ink.lineWidth = line;
  ink.beginPath();
  ink.moveTo(0, middle);
  ink.lineTo(stack.width, middle);
  ink.stroke();
  ink.setLineDash([]);

  let shown = 0;
  const p = doc.picks;
  if (p) {
    for (const [i, u, v, r] of doc.overlay[name][k]) {
      const above = visible(i);
      if (!above && !state.showBelow) continue;
      if (above) shown += 1;
      ink.beginPath();
      ink.arc(u, v, r, 0, 2 * Math.PI);
      if (i === state.selected) {
        ink.lineWidth = 3 * line;
        ink.strokeStyle = "#ffffff";
      } else if (above) {
        ink.lineWidth = 1.5 * line;
        ink.strokeStyle = rampColour(scoreFraction(i));
      } else {
        ink.lineWidth = line;
        ink.strokeStyle = "rgba(160, 160, 160, 0.7)";
        ink.setLineDash([3 * line, 3 * line]);
      }
      ink.stroke();
      ink.setLineDash([]);
    }
  }
  const centre = doc.tomogram.stacks[name].centres[k];
  const axis = name === "xy" ? "z" : "y";
  let label = `${axis} = ${nm(centre)} nm · slab ${k + 1} of ${stack.count}`;
  if (p) label += ` · ${shown} pick${shown === 1 ? "" : "s"}`;
  $(`${name}-label`).textContent = label;
}

function drawAll() {
  draw("xy");
  draw("xz");
}

function drawHistogram() {
  const p = state.doc.picks;
  const canvas = $("histogram");
  const dpr = window.devicePixelRatio || 1;
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  canvas.width = Math.round(width * dpr);
  canvas.height = Math.round(height * dpr);
  const ink = canvas.getContext("2d");
  ink.setTransform(dpr, 0, 0, dpr, 0, 0);
  ink.clearRect(0, 0, width, height);
  const values = p.sigma || p.score;
  const low = Math.min(...values);
  const high = Math.max(...values);
  const bins = 40;
  const span = high > low ? high - low : 1;
  const counts = new Array(bins).fill(0);
  for (const value of values) {
    counts[Math.min(bins - 1, Math.floor(((value - low) / span) * bins))] += 1;
  }
  const most = Math.max(...counts, 1);
  const left = 4;
  const right = width - 4;
  const base = height - 18;
  const top = 6;
  const x = (value) => left + ((value - low) / span) * (right - left);
  const thresholdValue = p.sigma && p.search_sigma
    ? state.threshold / p.search_sigma : state.threshold;
  for (let b = 0; b < bins; b += 1) {
    const from = low + (b / bins) * span;
    const to = low + ((b + 1) / bins) * span;
    const barHeight = (counts[b] / most) * (base - top);
    const above = to > thresholdValue;
    ink.fillStyle = rampColour((b + 0.5) / bins, above ? 0.9 : 0.25);
    ink.fillRect(x(from) + 0.5, base - barHeight, Math.max(1, x(to) - x(from) - 1), barHeight);
  }
  ink.fillStyle = getComputedStyle(document.body).color;
  ink.font = "11px system-ui, sans-serif";
  ink.textBaseline = "top";
  ink.textAlign = "left";
  ink.fillText(low.toFixed(p.sigma ? 1 : 3), left, base + 3);
  ink.textAlign = "right";
  ink.fillText(high.toFixed(p.sigma ? 1 : 3), right, base + 3);
  ink.textAlign = "center";
  ink.fillText(p.sigma ? "score (σ of the search's noise)" : "score", (left + right) / 2, base + 3);
  if (p.sigma && p.floor_sigma != null && p.floor_sigma >= low && p.floor_sigma <= high) {
    ink.strokeStyle = "rgba(248, 113, 113, 0.9)";
    ink.setLineDash([3, 3]);
    ink.beginPath();
    ink.moveTo(x(p.floor_sigma), top);
    ink.lineTo(x(p.floor_sigma), base);
    ink.stroke();
    ink.setLineDash([]);
    ink.fillStyle = "rgba(248, 113, 113, 0.95)";
    ink.textAlign = "left";
    ink.fillText("noise: 1 peak", Math.min(x(p.floor_sigma) + 3, right - 70), top);
  }
  ink.strokeStyle = getComputedStyle(document.body).color;
  ink.lineWidth = 1.5;
  ink.beginPath();
  ink.moveTo(x(thresholdValue), top);
  ink.lineTo(x(thresholdValue), base);
  ink.stroke();
}

// ---- threshold -------------------------------------------------------------------

function updateThreshold() {
  const p = state.doc.picks;
  const fraction = Number($("threshold").value) / 1000;
  state.threshold = p.minScore + fraction * (p.maxScore - p.minScore);
  // The lowest pick at or above the slider stands for it, so the line and the count agree.
  const kept = p.score.filter((s) => s >= state.threshold);
  const cut = kept.length ? Math.min(...kept) : state.threshold;
  state.threshold = cut;
  const sigmaText = p.search_sigma ? ` (${(cut / p.search_sigma).toFixed(1)}σ)` : "";
  $("threshold-label").textContent =
    `At or above ${cut.toFixed(4)}${sigmaText}: ${kept.length} of ${p.count} picks`;
  $("toml-line").textContent =
    `[pick]\ncut_off = ${cut.toFixed(4)}  # ${kept.length} of the ${p.count} picks on this page`;
  if (state.selected != null && !visible(state.selected) && !state.showBelow) {
    state.selected = null;
    $("pick-figure").hidden = true;
  }
  drawHistogram();
  drawAll();
}

// ---- picks ----------------------------------------------------------------------

function selectPick(i) {
  const doc = state.doc;
  const p = doc.picks;
  state.selected = i;
  state.index.xy = p.xy_slab[i];
  state.index.xz = p.xz_slab[i];
  $("xy-slider").value = String(state.index.xy);
  $("xz-slider").value = String(state.index.xz);
  const voxel = doc.tomogram.voxel_angstrom / 10;
  $("pick-details").textContent =
    `Rank ${p.rank[i]} of ${p.count} · score ${scoreText(i)} · ` +
    `x ${p.x[i]}, y ${p.y[i]}, z ${p.z[i]} voxels · ` +
    `${(p.z[i] * voxel).toFixed(0)} nm above the bottom`;
  const caption = p.oriented
    ? "XY, XZ and YZ through the pick, turned into the template's frame by its angles"
    : "XY, XZ and YZ through the pick, not turned: the picks carry no angles";
  $("pick-tile-caption").textContent = caption +
    (p.low_pass_angstrom ? `, low-passed to the search's ${p.low_pass_angstrom} Å.` : ".");
  $("pick-tile").src = `${state.base}pick_${i}.png`;
  $("pick-figure").hidden = false;
  drawAll();
}

function step(direction) {
  const p = state.doc && state.doc.picks;
  if (!p) return;
  const order = p.order.filter((i) => visible(i));
  if (order.length === 0) return;
  const at = state.selected == null ? -1 : order.indexOf(state.selected);
  let next;
  if (at === -1) next = direction > 0 ? 0 : order.length - 1;
  else next = Math.min(order.length - 1, Math.max(0, at + direction));
  selectPick(order[next]);
}

function pictureCoordinates(name, event) {
  const stack = state.doc.tomogram.stacks[name];
  const stage = $(`${name}-viewer`).querySelector(".stage");
  const scale = stage.clientWidth / stack.width;
  const box = event.target.getBoundingClientRect();
  return { u: (event.clientX - box.left) / scale, v: (event.clientY - box.top) / scale, scale };
}

function onViewerClick(name, event) {
  const doc = state.doc;
  if (!doc || !doc.tomogram) return;
  const { u, v, scale } = pictureCoordinates(name, event);
  if (doc.picks) {
    let best = null;
    for (const [i, cu, cv, r] of doc.overlay[name][state.index[name]]) {
      if (!visible(i) && !state.showBelow) continue;
      const distance = Math.hypot(u - cu, v - cv);
      if (distance <= r + 4 / scale && (best === null || distance < best.distance)) {
        best = { i, distance };
      }
    }
    if (best) {
      selectPick(best.i);
      return;
    }
  }
  // Elsewhere: move the other view to the slab that holds this row.
  const other = name === "xy" ? "xz" : "xy";
  const bands = doc.tomogram.stacks[name].bands;
  const hit = bands.findIndex(([top, , bottom]) => v >= top && v < bottom);
  if (hit >= 0) {
    state.index[other] = hit;
    $(`${other}-slider`).value = String(hit);
    drawAll();
  }
}

// ---- wiring -------------------------------------------------------------------------

for (const name of ["xy", "xz"]) {
  $(`${name}-slider`).addEventListener("input", (event) => {
    state.index[name] = Number(event.target.value);
    drawAll();
  });
  $(`${name}-viewer`).querySelector("canvas.overlay")
    .addEventListener("click", (event) => onViewerClick(name, event));
}

$("threshold").addEventListener("input", updateThreshold);
$("show-below").addEventListener("change", (event) => {
  state.showBelow = event.target.checked;
  drawAll();
});
$("previous").addEventListener("click", () => step(-1));
$("next").addEventListener("click", () => step(1));
$("copy-toml").addEventListener("click", async () => {
  try {
    await navigator.clipboard.writeText($("toml-line").textContent);
    $("copy-toml").textContent = "Copied";
    setTimeout(() => { $("copy-toml").textContent = "Copy"; }, 1500);
  } catch {
    $("copy-toml").textContent = "Select and copy";
  }
});
$("brightness").addEventListener("input", (event) => {
  state.brightness = Number(event.target.value);
  drawAll();
});
$("contrast").addEventListener("input", (event) => {
  state.contrast = Number(event.target.value);
  drawAll();
});
$("reset-display").addEventListener("click", () => {
  state.brightness = 1;
  state.contrast = 1;
  $("brightness").value = "1";
  $("contrast").value = "1";
  drawAll();
});

document.addEventListener("keydown", (event) => {
  if (!state.doc || !state.doc.tomogram) return;
  const target = event.target;
  if (target instanceof HTMLInputElement || target instanceof HTMLSelectElement) return;
  const stack = state.doc.tomogram.stacks.xy;
  if (event.key === "ArrowUp" || event.key === "ArrowDown") {
    const change = event.key === "ArrowUp" ? 1 : -1;
    state.index.xy = Math.min(stack.count - 1, Math.max(0, state.index.xy + change));
    $("xy-slider").value = String(state.index.xy);
    drawAll();
    event.preventDefault();
  } else if (event.key === "n") {
    step(1);
  } else if (event.key === "p") {
    step(-1);
  }
});

let resizeTimer = null;
window.addEventListener("resize", () => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(() => {
    drawAll();
    if (state.doc && state.doc.picks) drawHistogram();
  }, 100);
});

start();
