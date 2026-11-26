"use strict";

// The classification views (app.js routes to them): a run's classes over its
// iterations, and the FSC between two half runs. As in app.js, everything drawn comes
// from the server's documents; a map's pictures are PNGs the server cut, and the maps
// themselves go to Mol*.

// The classification report's channel colours (reports/classification.py), so that a
// class has one colour here and in the PDF.
const CHANNEL_COLOURS = ["#2a78d6", "#eb6834", "#1baf7a", "#8e5bd0", "#c9407a", "#7a7a72"];
const channelColour = (channel) => CHANNEL_COLOURS[(channel - 1) % CHANNEL_COLOURS.length];
const hexNumber = (hex) => parseInt(hex.slice(1), 16);
const inkColour = () => getComputedStyle(document.body).color;

const mra = { run: null, base: "", iteration: 0, section: 0, sprites: new Map() };

// ---- a run -------------------------------------------------------------------------

async function loadRun(name) {
  setStatus(`Reading ${name}.`);
  $("mra-classes").replaceChildren();
  let run;
  try {
    run = await api(`/api/mra/${encodeURIComponent(name)}`);
  } catch (error) {
    setStatus(`${name}: ${error.message}`, true);
    return;
  }
  if ($("series").value !== `mra/${name}`) return; // something else was chosen meanwhile
  mra.run = run;
  mra.base = `/mra/${encodeURIComponent(name)}/${run.key}/`;
  mra.sprites = new Map();
  mra.iteration = run.iterations.length - 1;
  $("mra-iteration").max = String(run.iterations.length - 1);
  $("mra-iteration").value = String(mra.iteration);
  mra.section = Math.floor((run.box_px - 1) / 2);
  $("mra-section").max = String(run.box_px - 1);
  $("mra-section").value = String(mra.section);
  showRunFacts(run);
  showParameters(run.parameters);
  showList("mra-problems", "mra-problems-panel", run.problems);
  $("mra-caption").textContent =
    "Each map on one grey scale of its own, values as they are, y and z upwards, as the " +
    "classification report draws it: one z section at a time, chosen by the slider, then " +
    "the central XY, XZ and YZ sections, through the box's middle voxel, or the mean of " +
    "the two about the middle of an even box.";
  $("mra-3d-note").textContent = state.overview.viewer3d
    ? "“Show in 3D” on a class draws it here (Mol*, 5 MB the first time)."
    : "Mol* is not installed next to this server: run scripts/fetch_viewer_assets.py.";
  setStatus(run.finished ? "" : `${name} has not finished an iteration yet.`);
  renderIteration();
}

function showRunFacts(run) {
  const pixel = run.pixel_angstrom;
  const facts = [
    ["What", run.software],
    ["Particles", run.particles.toLocaleString() +
      (run.decoys ? `, of them ${run.decoys} decoys (${run.decoy_kind})` : "")],
    ["Box", `${run.box_px} px` + (pixel ? `, ${(run.box_px * pixel / 10).toFixed(1)} nm` : "")],
    ["Pixel size", pixel ? `${pixel} Å` : "not recorded"],
    ["Iterations", `${run.finished} finished of ${run.planned ?? "?"} planned`],
    ["Protein side", `${run.protein_sign > 0 ? "positive" : "negative"}: ${run.protein_sign_source}`],
  ];
  fillFacts("mra-facts", facts);
}

function fillFacts(id, facts) {
  const list = $(id);
  list.replaceChildren();
  for (const [term, value] of facts) {
    const dt = document.createElement("dt");
    dt.textContent = term;
    const dd = document.createElement("dd");
    dd.textContent = value;
    list.append(dt, dd);
  }
}

function showParameters(parameters) {
  const table = $("mra-parameters");
  table.replaceChildren();
  let group = null;
  for (const one of parameters) {
    if (one.group !== group) {
      group = one.group;
      const row = table.insertRow();
      const header = document.createElement("th");
      header.colSpan = 2;
      header.textContent = group;
      row.append(header);
    }
    const row = table.insertRow();
    row.insertCell().textContent = one.name;
    row.insertCell().textContent = one.value;
  }
}

function showList(listId, panelId, items) {
  $(listId).replaceChildren(...(items || []).map((text) => {
    const item = document.createElement("li");
    item.textContent = text;
    return item;
  }));
  $(panelId).hidden = !items || items.length === 0;
}

function renderIteration() {
  const run = mra.run;
  const iteration = run.iterations[mra.iteration];
  let label;
  if (iteration.number === 0) {
    label = "0 · the starting references";
  } else {
    label = `${iteration.number} of ${run.finished} finished`;
    if (iteration.summary) label += ` · ${iteration.summary}`;
    if (iteration.changed != null) {
      label += ` · ${iteration.changed} particle${iteration.changed === 1 ? "" : "s"} changed class`;
    }
  }
  $("mra-iteration-label").textContent = label;
  $("mra-classes").replaceChildren(...iteration.classes.map((one) => classCard(iteration, one)));
  updateSectionLabel();
  drawSizes();
}

function describeClass(iteration, one) {
  if (iteration.number === 0) return one.note || "starting reference";
  const parts = [`${one.members} particle${one.members === 1 ? "" : "s"}`];
  if (one.decoys) parts.push(`${one.decoys} of them decoys`);
  if (one.median_cc != null) parts.push(`median CC ${one.median_cc.toFixed(3)}`);
  return parts.join(" · ");
}

function classCard(iteration, one) {
  const card = document.createElement("article");
  card.className = "class-card";
  card.style.setProperty("--channel", channelColour(one.channel));
  const title = document.createElement("h3");
  title.textContent = iteration.number === 0 ? `Reference ${one.channel}` : `Class ${one.channel}`;
  const numbers = document.createElement("p");
  numbers.className = "numbers";
  numbers.textContent = describeClass(iteration, one);
  card.append(title, numbers);
  if (iteration.number > 0 && !one.map) {
    const missing = document.createElement("p");
    missing.className = "note";
    missing.textContent = "No average was written for this class.";
    card.append(missing);
    return card;
  }
  const stem = `class_${iteration.number}_${one.channel}`;
  const canvas = document.createElement("canvas");
  canvas.dataset.stem = stem;
  const views = document.createElement("img");
  views.alt = `${title.textContent}: central XY, XZ and YZ sections`;
  views.src = `${mra.base}${stem}_views.png`;
  const viewsLabel = document.createElement("div");
  viewsLabel.className = "views-label";
  for (const name of ["XY", "XZ", "YZ"]) {
    const span = document.createElement("span");
    span.textContent = name;
    viewsLabel.append(span);
  }
  const button = document.createElement("button");
  button.type = "button";
  button.textContent = "Show in 3D";
  button.disabled = !state.overview.viewer3d;
  button.addEventListener("click", () => showClass3d(iteration.number, one.channel));
  card.append(canvas, views, viewsLabel, button);
  sprite(stem).then(() => drawSection(canvas));
  return card;
}

// A map's z sections, every one, in one PNG, the first at the top (dashboard/mra.py).
function sprite(stem) {
  if (!mra.sprites.has(stem)) {
    const image = new Image();
    const loaded = new Promise((resolve) => {
      image.onload = resolve;
      image.onerror = resolve;
    });
    image.src = `${mra.base}${stem}_sections.png`;
    mra.sprites.set(stem, { image, loaded });
  }
  return mra.sprites.get(stem).loaded;
}

function drawSection(canvas) {
  const entry = mra.sprites.get(canvas.dataset.stem);
  if (!entry || !entry.image.naturalWidth) return;
  const image = entry.image;
  const depth = mra.run.box_px;
  const blockHeight = image.naturalHeight / depth;
  const cssWidth = canvas.clientWidth;
  if (cssWidth === 0) return;
  const dpr = window.devicePixelRatio || 1;
  const cssHeight = cssWidth * blockHeight / image.naturalWidth;
  canvas.style.height = `${cssHeight}px`;
  canvas.width = Math.round(cssWidth * dpr);
  canvas.height = Math.round(cssHeight * dpr);
  const ink = canvas.getContext("2d");
  ink.imageSmoothingEnabled = false;
  ink.clearRect(0, 0, canvas.width, canvas.height);
  ink.drawImage(image, 0, mra.section * blockHeight, image.naturalWidth, blockHeight,
    0, 0, canvas.width, canvas.height);
}

function updateSectionLabel() {
  const run = mra.run;
  const offset = mra.section - (run.box_px - 1) / 2;
  const away = run.pixel_angstrom
    ? `${(offset * run.pixel_angstrom).toFixed(1)} Å`
    : `${offset.toFixed(1)} voxels`;
  $("mra-section-label").textContent =
    `${mra.section + 1} of ${run.box_px} · ${away} from the box centre`;
}

function drawSections() {
  for (const canvas of $("mra-classes").querySelectorAll("canvas")) drawSection(canvas);
}

function drawSizes() {
  const run = mra.run;
  const canvas = $("mra-sizes");
  const dpr = window.devicePixelRatio || 1;
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  if (width === 0) return;
  canvas.width = Math.round(width * dpr);
  canvas.height = Math.round(height * dpr);
  const ink = canvas.getContext("2d");
  ink.setTransform(dpr, 0, 0, dpr, 0, 0);
  ink.clearRect(0, 0, width, height);
  ink.font = "11px system-ui, sans-serif";
  ink.fillStyle = inkColour();
  const count = run.finished;
  if (count === 0) {
    ink.fillText("No iteration has finished yet.", 8, 20);
    return;
  }
  const left = 40;
  const right = width - 90;
  const top = 10;
  const bottom = height - 22;
  const most = Math.max(1, ...Object.values(run.sizes).flat().filter((one) => one != null));
  const x = (number) => count === 1 ? (left + right) / 2
    : left + ((number - 1) / (count - 1)) * (right - left);
  const y = (value) => bottom - (value / most) * (bottom - top);

  ink.strokeStyle = "rgba(128, 128, 128, 0.35)";
  ink.lineWidth = 1;
  ink.beginPath();
  ink.moveTo(left, bottom);
  ink.lineTo(right, bottom);
  ink.stroke();
  ink.textAlign = "right";
  ink.textBaseline = "middle";
  ink.fillText("0", left - 6, bottom);
  ink.fillText(String(most), left - 6, top);
  ink.textAlign = "center";
  ink.textBaseline = "top";
  const step = Math.max(1, Math.ceil(count / Math.max(1, Math.floor((right - left) / 32))));
  for (let number = 1; number <= count; number += step) ink.fillText(String(number), x(number), bottom + 5);

  const selected = run.iterations[mra.iteration].number;
  if (selected > 0) {
    ink.strokeStyle = inkColour();
    ink.setLineDash([3, 3]);
    ink.beginPath();
    ink.moveTo(x(selected), top);
    ink.lineTo(x(selected), bottom);
    ink.stroke();
    ink.setLineDash([]);
  }

  let legend = top + 4;
  for (const [key, sizes] of Object.entries(run.sizes)) {
    const channel = Number(key);
    const colour = channelColour(channel);
    ink.strokeStyle = colour;
    ink.fillStyle = colour;
    ink.lineWidth = 2;
    ink.beginPath();
    let drawing = false;
    sizes.forEach((value, index) => {
      if (value == null) {
        drawing = false;
        return;
      }
      if (drawing) ink.lineTo(x(index + 1), y(value));
      else ink.moveTo(x(index + 1), y(value));
      drawing = true;
    });
    ink.stroke();
    sizes.forEach((value, index) => {
      if (value == null) return;
      ink.beginPath();
      ink.arc(x(index + 1), y(value), 3, 0, 2 * Math.PI);
      ink.fill();
    });
    ink.textAlign = "left";
    ink.textBaseline = "middle";
    ink.fillRect(right + 14, legend - 1, 12, 3);
    ink.fillText(`class ${channel}`, right + 30, legend);
    legend += 16;
  }
  canvas.onclick = (event) => {
    const box = canvas.getBoundingClientRect();
    const at = event.clientX - box.left;
    let best = 1;
    for (let number = 1; number <= count; number += 1) {
      if (Math.abs(x(number) - at) < Math.abs(x(best) - at)) best = number;
    }
    const index = run.iterations.findIndex((one) => one.number === best);
    if (index >= 0) {
      mra.iteration = index;
      $("mra-iteration").value = String(index);
      renderIteration();
    }
  };
}

async function showClass3d(number, channel) {
  const run = mra.run;
  const note = $("mra-3d-note");
  note.textContent = "Loading Mol*…";
  const level = run.protein_sign;
  const volumes = [{ url: `${mra.base}class_${number}_${channel}.mrc`, type: "relative",
    value: level, color: hexNumber(channelColour(channel)) }];
  const withMask = run.mask && $("mra-3d-mask").checked;
  if (withMask) {
    volumes.push({ url: `${mra.base}mask.mrc`, type: "absolute", value: 0.5, color: 0x9aa1aa,
      alpha: 0.2 });
  }
  try {
    await showVolumes("mra-3d-slot", volumes);
    note.textContent =
      `${number === 0 ? `Reference ${channel}` : `Class ${channel} after iteration ${number}`}, ` +
      `its surface at ${level > 0 ? "+" : "−"}1σ on the protein's side` +
      (withMask ? ", in the alignment mask" : "") +
      ". Mol*'s panel on the right changes the level.";
  } catch (error) {
    note.textContent = error.message;
  }
}

$("mra-iteration").addEventListener("input", (event) => {
  mra.iteration = Number(event.target.value);
  renderIteration();
});
$("mra-section").addEventListener("input", (event) => {
  mra.section = Number(event.target.value);
  updateSectionLabel();
  drawSections();
});

// ---- an FSC ------------------------------------------------------------------------

const fscView = { doc: null };

async function loadFsc(name) {
  setStatus(`Reading ${name}.`);
  let doc;
  try {
    doc = await api(`/api/fsc/${encodeURIComponent(name)}`);
  } catch (error) {
    setStatus(`${name}: ${error.message}`, true);
    return;
  }
  if ($("series").value !== `fsc/${name}`) return;
  fscView.doc = doc;
  showFscFacts();
  $("fsc-caption").textContent =
    "The curve with the halves' alignment mask applied, read where it falls through 0.5 and " +
    "0.143, interpolated in spatial frequency as scripts/fsc_phase_randomization.py reads it; " +
    "a/b where it falls through a level more than once, first and last.";
  const button = $("fsc-show-3d");
  button.hidden = false;
  button.disabled = !state.overview.viewer3d || !doc.maps.includes("mean.mrc");
  $("fsc-3d-note").textContent = !doc.maps.includes("mean.mrc")
    ? `No ${doc.name}_mean.mrc beside the curves.`
    : state.overview.viewer3d ? "Mol* is loaded the first time it is asked for (5 MB)."
      : "Mol* is not installed next to this server: run scripts/fetch_viewer_assets.py.";
  setStatus("");
  drawFsc();
}

function lastValue(values) {
  for (let index = values.length - 1; index >= 0; index -= 1) {
    if (values[index] != null) return values[index];
  }
  return null;
}

// As scripts/fsc_phase_randomization.py's crossing_text.
function crossingText(curve, level) {
  const found = curve.falls[level] || [];
  if (found.length === 0) return "never crossed";
  const last = lastValue(curve.correlation);
  if (found.length === 1 || (last != null && last >= Number(level))) {
    return `${found[0].toFixed(1)} Å`;
  }
  return `${found[0].toFixed(1)}/${found[found.length - 1].toFixed(1)} Å`;
}

function showFscFacts() {
  const doc = fscView.doc;
  const facts = [
    ["Halves", doc.halves.join(" and ") || "not named"],
    ["Iteration", doc.iteration != null ? `${doc.iteration}, of both` : "not recorded"],
    ["Started from", doc.start_poses === "picked" ? "their picks' poses"
      : "the poses the first run refined"],
    ["Pixel size", doc.pixel_angstrom ? `${doc.pixel_angstrom} Å, Nyquist ` +
      `${(2 * doc.pixel_angstrom).toFixed(1)} Å` : "not recorded"],
  ];
  const labels = $("fsc-unmasked").checked ? ["masked", "unmasked"] : ["masked"];
  for (const label of labels) {
    const curve = doc.curves[label];
    if (!curve) continue;
    const name = label === "masked" ? "" : " without the mask";
    facts.push([`0.5${name}`, crossingText(curve, "0.5")]);
    facts.push([`0.143${name}`, crossingText(curve, "0.143")]);
    const rise = curve.rise;
    if (rise && rise.dip_correlation != null) {
      facts.push([`Toward Nyquist${name}`,
        `rises again, ${rise.dip_correlation.toFixed(2)} at ${rise.dip_resolution_angstrom.toFixed(1)} Å ` +
        `to ${rise.peak_correlation.toFixed(2)} at ${rise.peak_resolution_angstrom.toFixed(1)} Å ` +
        `(${rise.sigma.toFixed(0)}σ)`]);
    }
  }
  fillFacts("fsc-facts", facts);
}

const NICE_RESOLUTIONS = [200, 100, 60, 50, 40, 30, 25, 20, 15, 12, 10, 8, 6, 5, 4, 3, 2];

function drawFsc() {
  const doc = fscView.doc;
  if (!doc) return;
  const canvas = $("fsc-chart");
  const dpr = window.devicePixelRatio || 1;
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  if (width === 0) return;
  canvas.width = Math.round(width * dpr);
  canvas.height = Math.round(height * dpr);
  const ink = canvas.getContext("2d");
  ink.setTransform(dpr, 0, 0, dpr, 0, 0);
  ink.clearRect(0, 0, width, height);
  ink.font = "11px system-ui, sans-serif";
  const colour = inkColour();
  const labels = $("fsc-unmasked").checked ? ["unmasked", "masked"] : ["masked"];
  const curves = labels.map((label) => [label, doc.curves[label]]).filter(([, curve]) => curve);
  const frequencies = curves.flatMap(([, curve]) => curve.frequency);
  const highest = doc.nyquist_frequency || Math.max(...frequencies);
  const lowestValue = Math.min(0, ...curves.flatMap(([, curve]) =>
    curve.correlation.filter((one) => one != null)));
  const left = 44;
  const right = width - 12;
  const top = 10;
  const bottom = height - 40;
  const x = (frequency) => left + (frequency / highest) * (right - left);
  const y = (value) => bottom - ((value - lowestValue) / (1 - lowestValue)) * (bottom - top);

  ink.strokeStyle = "rgba(128, 128, 128, 0.35)";
  ink.lineWidth = 1;
  ink.fillStyle = colour;
  ink.textAlign = "right";
  ink.textBaseline = "middle";
  for (const value of [0, 1]) {
    ink.beginPath();
    ink.moveTo(left, y(value));
    ink.lineTo(right, y(value));
    ink.stroke();
    ink.fillText(String(value), left - 6, y(value));
  }
  ink.setLineDash([4, 4]);
  for (const value of [0.5, 0.143]) {
    ink.beginPath();
    ink.moveTo(left, y(value));
    ink.lineTo(right, y(value));
    ink.stroke();
    ink.fillText(String(value), left - 6, y(value));
  }
  ink.setLineDash([]);

  ink.textAlign = "center";
  ink.textBaseline = "top";
  let lastLabel = -Infinity;
  for (const resolution of NICE_RESOLUTIONS) {
    const frequency = 1 / resolution;
    if (frequency > highest * 1.0001) break;
    const at = x(frequency);
    if (at - lastLabel < 34) continue;
    ink.beginPath();
    ink.moveTo(at, bottom);
    ink.lineTo(at, bottom + 4);
    ink.stroke();
    ink.fillText(String(resolution), at, bottom + 6);
    lastLabel = at;
  }
  ink.fillText("resolution (Å), on a scale of spatial frequency", (left + right) / 2, bottom + 22);

  for (const [label, curve] of curves) {
    ink.strokeStyle = label === "masked" ? CHANNEL_COLOURS[0] : "rgba(154, 161, 170, 0.9)";
    ink.lineWidth = label === "masked" ? 2 : 1.5;
    ink.beginPath();
    let drawing = false;
    curve.frequency.forEach((frequency, index) => {
      const value = curve.correlation[index];
      if (value == null || frequency > highest * 1.0001) {
        drawing = false;
        return;
      }
      if (drawing) ink.lineTo(x(frequency), y(value));
      else ink.moveTo(x(frequency), y(value));
      drawing = true;
    });
    ink.stroke();
  }
  const masked = doc.curves.masked;
  if (masked) {
    ink.fillStyle = CHANNEL_COLOURS[0];
    ink.textAlign = "left";
    ink.textBaseline = "bottom";
    for (const level of ["0.5", "0.143"]) {
      const found = masked.falls[level] || [];
      if (!found.length) continue;
      const at = x(1 / found[0]);
      ink.beginPath();
      ink.arc(at, y(Number(level)), 4, 0, 2 * Math.PI);
      ink.fill();
      ink.fillText(`${found[0].toFixed(1)} Å`, at + 6, y(Number(level)) - 3);
    }
  }
}

$("fsc-unmasked").addEventListener("change", () => {
  showFscFacts();
  drawFsc();
});

$("fsc-show-3d").addEventListener("click", async () => {
  const doc = fscView.doc;
  const note = $("fsc-3d-note");
  note.textContent = "Loading Mol*…";
  const base = `/fsc/${encodeURIComponent(doc.name)}/${doc.key}/`;
  const level = doc.protein_sign;
  const volumes = [{ url: `${base}mean.mrc`, type: "relative", value: level, color: 0xfacc15 }];
  if (doc.maps.includes("mask.mrc")) {
    volumes.push({ url: `${base}mask.mrc`, type: "absolute", value: 0.5, color: 0x9aa1aa,
      alpha: 0.2 });
  }
  try {
    await showVolumes("fsc-3d-slot", volumes);
    $("fsc-show-3d").hidden = true;
    note.textContent =
      `The mean of the two half maps, its surface at ${level > 0 ? "+" : "−"}1σ on the ` +
      `protein's side (${doc.protein_sign_source})` +
      (doc.maps.includes("mask.mrc") ? "; the mask at 0.5, translucent" : "") +
      ". Mol*'s panel on the right changes the level.";
  } catch (error) {
    note.textContent = error.message;
  }
});

window.addEventListener("resize", () => {
  if (!$("mra-view").hidden && mra.run) {
    drawSizes();
    drawSections();
  }
  if (!$("fsc-view").hidden) drawFsc();
});
