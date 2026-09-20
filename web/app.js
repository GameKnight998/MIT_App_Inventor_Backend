/* Image Locator web interface.
 *
 * Chooses the endpoint from what was dropped (one image, a video, or several
 * images -> a case), then renders the answer as a map plus the evidence behind
 * it. The point of showing rejected candidates and the reasoning trace is that
 * an unexplained pin is not usable intelligence: the user needs to see WHY.
 */

'use strict';

const els = {
  dropzone: document.getElementById('dropzone'),
  fileInput: document.getElementById('fileInput'),
  fileList: document.getElementById('fileList'),
  analyzeBtn: document.getElementById('analyzeBtn'),
  resetBtn: document.getElementById('resetBtn'),
  modeHint: document.getElementById('modeHint'),
  progress: document.getElementById('progress'),
  progressText: document.getElementById('progressText'),
  error: document.getElementById('error'),
  results: document.getElementById('results'),
  placeName: document.getElementById('placeName'),
  address: document.getElementById('address'),
  coords: document.getElementById('coords'),
  badges: document.getElementById('badges'),
  confValue: document.getElementById('confValue'),
  confFill: document.getElementById('confFill'),
  confNote: document.getElementById('confNote'),
  warning: document.getElementById('warning'),
  mapLink: document.getElementById('mapLink'),
  cards: document.getElementById('cards'),
  rawJson: document.getElementById('rawJson'),
  health: document.getElementById('health'),
};

let files = [];
let map = null;
let layer = null;

/* ------------------------------------------------------------------ utils */

const esc = (value) =>
  String(value ?? '').replace(/[&<>"']/g, (c) => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
  })[c]);

const kb = (bytes) =>
  bytes > 1048576
    ? `${(bytes / 1048576).toFixed(1)} MB`
    : `${Math.max(1, Math.round(bytes / 1024))} KB`;

const isVideo = (file) =>
  file.type.startsWith('video/') || /\.(mp4|mov|webm|mkv|avi|3gp)$/i.test(file.name);

function show(el, visible) {
  el.hidden = !visible;
}

/* ------------------------------------------------------------------ health */

fetch('/health')
  .then((r) => r.json())
  .then((h) => {
    const bits = [`provider: ${h.vision_provider || 'n/a'}`];
    if (h.vision_model) bits.push(h.vision_model);
    if (!h.vision_enabled) bits.push('vision key missing');
    els.health.textContent = bits.join(' · ');
    els.health.className = 'status ' + (h.vision_enabled ? 'ok' : 'bad');
  })
  .catch(() => {
    els.health.textContent = 'Backend unreachable';
    els.health.className = 'status bad';
  });

/* -------------------------------------------------------------- file input */

els.dropzone.addEventListener('click', () => els.fileInput.click());
els.dropzone.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); els.fileInput.click(); }
});

['dragenter', 'dragover'].forEach((evt) =>
  els.dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    els.dropzone.classList.add('dragover');
  })
);

['dragleave', 'drop'].forEach((evt) =>
  els.dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    els.dropzone.classList.remove('dragover');
  })
);

els.dropzone.addEventListener('drop', (e) => setFiles(e.dataTransfer.files));
els.fileInput.addEventListener('change', () => setFiles(els.fileInput.files));
els.resetBtn.addEventListener('click', reset);
els.analyzeBtn.addEventListener('click', analyze);

function setFiles(list) {
  files = Array.from(list || []);
  els.fileList.innerHTML = files
    .map((f) => `<div class="file-row"><span>${esc(f.name)}</span><span>${kb(f.size)}</span></div>`)
    .join('');
  show(els.fileList, files.length > 0);
  show(els.resetBtn, files.length > 0);
  els.analyzeBtn.disabled = files.length === 0;
  els.modeHint.textContent = describeMode();
  show(els.error, false);
}

function describeMode() {
  if (!files.length) return '';
  if (files.length > 1) return `${files.length} images → cross-referenced case`;
  return isVideo(files[0]) ? 'Video → key frames fused' : 'Single image → full analysis';
}

function reset() {
  files = [];
  els.fileInput.value = '';
  els.fileList.innerHTML = '';
  show(els.fileList, false);
  show(els.resetBtn, false);
  show(els.results, false);
  show(els.error, false);
  els.analyzeBtn.disabled = true;
  els.modeHint.textContent = '';
}

/* --------------------------------------------------------------- analysing */

async function analyze() {
  if (!files.length) return;

  const many = files.length > 1;
  const endpoint = many ? '/case' : isVideo(files[0]) ? '/analyze-video' : '/analyze';

  const body = new FormData();
  if (many) files.forEach((f, i) => body.append(`image${i}`, f, f.name));
  else body.append('image', files[0], files[0].name);

  els.analyzeBtn.disabled = true;
  show(els.progress, true);
  show(els.error, false);
  show(els.results, false);
  els.progressText.textContent = many
    ? `Analysing ${files.length} images and cross-referencing them…`
    : 'Analysing…';

  try {
    const resp = await fetch(endpoint, { method: 'POST', body });
    const data = await resp.json();
    if (!resp.ok || data.success === false) {
      throw new Error(data.error || `Request failed (HTTP ${resp.status})`);
    }
    if (many) renderCase(data);
    else renderSingle(data);
  } catch (err) {
    els.error.textContent = err.message || 'Analysis failed.';
    show(els.error, true);
  } finally {
    show(els.progress, false);
    els.analyzeBtn.disabled = false;
  }
}

/* ----------------------------------------------------------------- shared */

function setConfidence(value, note) {
  const pct = Math.round((value || 0) * 100);
  els.confValue.textContent = `${pct}%`;
  els.confFill.style.width = `${pct}%`;
  els.confFill.className =
    'conf-fill ' + (pct < 35 ? 'low' : pct < 65 ? 'mid' : '');
  els.confNote.textContent = note || confidenceWording(pct);
}

function confidenceWording(pct) {
  if (pct >= 80) return 'Strong: multiple independent checks agree.';
  if (pct >= 60) return 'Moderate: corroborated, but not conclusive.';
  if (pct >= 35) return 'Weak: treat as a lead to verify, not an answer.';
  return 'Very weak: the image lacks distinctive, checkable detail.';
}

function badge(text, kind) {
  return `<span class="badge ${kind || ''}">${esc(text)}</span>`;
}

function verifiedBadge(status) {
  const map = {
    verified: ['Map verified', 'good'],
    partial: ['Partially verified', 'warn'],
    mismatch: ['Map contradicts', 'bad'],
    unavailable: ['Unverified (map offline)', 'warn'],
    skipped: ['No checkable features', ''],
  };
  const [text, kind] = map[status] || [`Verification: ${status}`, ''];
  return badge(text, kind);
}

function ensureMap() {
  if (map) return;
  map = L.map('map', { scrollWheelZoom: false });
  L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '&copy; OpenStreetMap contributors',
    maxZoom: 19,
  }).addTo(map);
  layer = L.layerGroup().addTo(map);
}

function marker(lat, lon, color, popup) {
  return L.circleMarker([lat, lon], {
    radius: 8,
    color: color,
    fillColor: color,
    fillOpacity: 0.65,
    weight: 2,
  }).bindPopup(popup);
}

const COLORS = { primary: '#3fb950', alt: '#d29922', rejected: '#f85149', cluster: '#58a6ff' };

function drawMap(points) {
  ensureMap();
  layer.clearLayers();
  const bounds = [];

  points.forEach((p) => {
    if (p.lat == null || p.lon == null) return;
    marker(p.lat, p.lon, COLORS[p.kind] || COLORS.alt, p.popup).addTo(layer);
    bounds.push([p.lat, p.lon]);
    if (p.radius) {
      L.circle([p.lat, p.lon], {
        radius: p.radius,
        color: COLORS[p.kind] || COLORS.alt,
        weight: 1,
        fillOpacity: 0.08,
      }).addTo(layer);
    }
  });

  if (!bounds.length) return;
  if (bounds.length === 1) map.setView(bounds[0], 12);
  else map.fitBounds(bounds, { padding: [40, 40], maxZoom: 13 });

  // Leaflet needs a nudge when its container was hidden while sizing.
  setTimeout(() => map.invalidateSize(), 80);
}

function card(title, inner, wide) {
  return `<div class="card${wide ? ' span-2' : ''}"><h3>${esc(title)}</h3>${inner}</div>`;
}

function list(items) {
  return `<ul>${items.map((i) => `<li>${i}</li>`).join('')}</ul>`;
}

/* ---------------------------------------------------------- single result */

function renderSingle(d) {
  els.placeName.textContent = d.location_name || 'Location could not be determined';
  els.address.textContent = d.address && d.address !== d.location_name ? d.address : '';
  const radiusBits = [];
  if (d.defined_radius) radiusBits.push(`defined radius ${d.defined_radius}`);
  if (d.precision_m) radiusBits.push(`±${d.precision_m} m estimated`);
  if (d.meets_defined_radius === true) radiusBits.push('meets target');
  if (d.meets_defined_radius === false) radiusBits.push('coarser than target');
  els.coords.textContent =
    d.latitude != null
      ? `${d.coordinates}${radiusBits.length ? ` · ${radiusBits.join(' · ')}` : ''}`
      : radiusBits.join(' · ');

  els.mapLink.href = d.map_url || '#';
  show(els.mapLink, Boolean(d.map_url));

  const badges = [badge(d.source, 'info'), verifiedBadge(d.verified)];
  if (d.authenticity && d.authenticity !== 'authentic' && d.authenticity !== 'unknown') {
    badges.push(badge(`Authenticity: ${d.authenticity.replace(/_/g, ' ')}`, 'bad'));
  }
  if (d.street_level && d.street_level.refined) badges.push(badge('Street-level match', 'good'));
  if (d.defined_radius) {
    badges.push(
      badge(
        d.meets_defined_radius === true
          ? `Within ${d.defined_radius}`
          : `Defined radius ${d.defined_radius}`,
        d.meets_defined_radius === true ? 'good' : 'info'
      )
    );
  }
  if (d.media_type === 'video' && d.video) {
    badges.push(badge(`${d.video.frames_analyzed} frames fused`, 'info'));
  }
  if (d.image_quality && d.image_quality.enhanced) {
    badges.push(badge('Image enhanced', 'info'));
  }
  els.badges.innerHTML = badges.join('');

  setConfidence(d.confidence);

  els.warning.textContent = d.warning || '';
  show(els.warning, Boolean(d.warning));

  const points = [];
  if (d.latitude != null) {
    points.push({
      lat: d.latitude,
      lon: d.longitude,
      kind: 'primary',
      radius: d.precision_m && d.precision_m < 20000 ? d.precision_m : null,
      popup: `<strong>${esc(d.location_name)}</strong><br>Best estimate · ${Math.round(
        (d.confidence || 0) * 100
      )}%`,
    });
  }
  (d.alternatives || []).forEach((a) => {
    if (a.latitude == null) return;
    points.push({
      lat: a.latitude,
      lon: a.longitude,
      kind: 'alt',
      popup: `<strong>${esc(a.name)}</strong><br>Alternative · ${Math.round(
        (a.confidence || 0) * 100
      )}%`,
    });
  });
  (d.rejected || []).forEach((r) => {
    if (r.latitude == null) return;
    points.push({
      lat: r.latitude,
      lon: r.longitude,
      kind: 'rejected',
      popup: `<strong>${esc(r.name)}</strong><br>Ruled out: ${esc(r.reason)}`,
    });
  });
  drawMap(points);

  els.cards.innerHTML = buildSingleCards(d);
  els.rawJson.textContent = JSON.stringify(d, null, 2);
  show(els.results, true);
  els.results.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function buildSingleCards(d) {
  const out = [];

  const street = d.street_level;
  if (street && street.attempted) {
    const clues = street.street_clues || {};
    const rows = [];
    ['street_names', 'house_numbers', 'business_names', 'transit_stops', 'postal_codes']
      .forEach((key) => {
        if (clues[key] && clues[key].length) {
          rows.push(
            `<div><span>${key.replace(/_/g, ' ')}</span><span>${esc(
              clues[key].slice(0, 3).join(', ')
            )}</span></div>`
          );
        }
      });
    if (street.matched_query) {
      rows.push(`<div><span>matched</span><span>${esc(street.matched_query)}</span></div>`);
    }
    out.push(
      card(
        'Street-level refinement',
        `<p class="muted">${esc(street.note)}</p>` +
          (rows.length ? `<div class="kv">${rows.join('')}</div>` : '')
      )
    );
  }

  const v = d.details && d.details.verification;
  if (v) {
    const rows = [];
    if (v.confirmed && v.confirmed.length) {
      rows.push(`<div><span>confirmed nearby</span><span>${esc(v.confirmed.join(', '))}</span></div>`);
    }
    if (v.missing && v.missing.length) {
      rows.push(`<div><span>not found</span><span>${esc(v.missing.join(', '))}</span></div>`);
    }
    if (v.context && v.context.length) {
      rows.push(`<div><span>context</span><span>${esc(v.context.join(', '))}</span></div>`);
    }
    if (v.radius_m) rows.push(`<div><span>search radius</span><span>${v.radius_m} m</span></div>`);
    if (d.landmark_check && d.landmark_check.note) {
      rows.push(`<div><span>named features</span><span>${esc(d.landmark_check.note)}</span></div>`);
    }
    if (rows.length) out.push(card('Map verification', `<div class="kv">${rows.join('')}</div>`));
  }

  if (d.rejected && d.rejected.length) {
    out.push(
      card(
        'Ruled out',
        list(
          d.rejected.map(
            (r) => `<span class="struck">${esc(r.name)}</span> &mdash; ${esc(r.reason)}`
          )
        )
      )
    );
  }

  const veh = d.vehicle;
  if (veh && veh.available && veh.vehicles_present) {
    const rows = (veh.vehicles || []).map((x) => {
      const label = [x.make, x.model].filter(Boolean).join(' ') || x.body_style || 'vehicle';
      const detail = [x.year_range, x.color].filter(Boolean).join(', ');
      return `<div><span>${esc(label)}</span><span>${esc(detail)}</span></div>`;
    });
    (veh.plates || []).forEach((p) => {
      rows.push(
        `<div><span>plate</span><span>${esc(
          p.text || p.format_description || ''
        )}${p.region_implied ? ` (${esc(p.region_implied)})` : ''}</span></div>`
      );
    });
    let inner = rows.length ? `<div class="kv">${rows.join('')}</div>` : '';
    if (d.vehicle_check && d.vehicle_check.note) {
      inner += `<p class="muted" style="margin-top:10px">${esc(d.vehicle_check.note)}</p>`;
    }
    out.push(card('Vehicles & plates', inner));
  }

  const env = [];
  if (d.solar && d.solar.note) env.push(`Sun: ${esc(d.solar.note)}`);
  if (d.climate_check && d.climate_check.note) env.push(`Climate: ${esc(d.climate_check.note)}`);
  if (d.elevation_m != null) env.push(`Ground elevation ≈ ${d.elevation_m} m`);
  if (env.length) out.push(card('Physical cross-checks', list(env)));

  const f = d.forensics;
  if (f && f.notes && f.notes.length) {
    out.push(card('File forensics', list(f.notes.map(esc))));
  }

  if (d.nearby_places && d.nearby_places.length) {
    out.push(
      card(
        'Nearby notable places',
        list(
          d.nearby_places
            .slice(0, 6)
            .map(
              (p) =>
                `<a href="${esc(p.url)}" target="_blank" rel="noopener">${esc(
                  p.title
                )}</a> &mdash; ${Math.round((p.distance_m || 0) / 100) / 10} km`
            )
        )
      )
    );
  }

  if (d.reasoning_trace && d.reasoning_trace.length) {
    const stages = d.reasoning_trace
      .map(
        (s) =>
          `<div class="trace-stage"><h4>${esc(s.stage)}</h4>${list(
            (s.details || []).map(esc)
          )}</div>`
      )
      .join('');
    out.push(card('How this was worked out', stages, true));
  }

  return out.join('');
}

/* ------------------------------------------------------------ case result */

function renderCase(d) {
  const consensus = d.consensus;
  els.placeName.textContent = consensus
    ? consensus.location_name || 'Consensus location'
    : 'No consensus across the images';
  els.address.textContent = d.summary || '';
  els.coords.textContent = consensus
    ? `Latitude ${consensus.latitude} Longitude ${consensus.longitude}`
    : '';

  els.mapLink.href = consensus ? consensus.map_url : '#';
  show(els.mapLink, Boolean(consensus));

  const badges = [
    badge(`Case ${d.case_id}`, 'info'),
    badge(`${d.images_in_case} image(s)`, ''),
    badge(`${d.images_located} located`, d.images_located ? 'good' : 'warn'),
  ];
  if (consensus) {
    badges.push(badge(`${consensus.images_agreeing}/${consensus.of_images} agree`, 'good'));
  }
  els.badges.innerHTML = badges.join('');

  setConfidence(
    consensus ? consensus.combined_confidence : 0,
    consensus
      ? `Combined from ${consensus.images_agreeing} agreeing image(s), spread ${consensus.spread_km} km.`
      : 'Images disagree, so no combined estimate is offered.'
  );

  els.warning.textContent = d.warning || '';
  show(els.warning, Boolean(d.warning));

  const points = [];
  (d.clusters || []).forEach((c, i) => {
    points.push({
      lat: c.latitude,
      lon: c.longitude,
      kind: i === 0 && consensus ? 'cluster' : 'alt',
      popup:
        `<strong>${esc(c.location_name || 'Cluster')}</strong><br>` +
        `${c.images_agreeing} image(s) · ${Math.round(c.combined_confidence * 100)}%<br>` +
        c.members.map((m) => esc(m.filename)).join('<br>'),
    });
  });
  drawMap(points);

  const rows = (d.results || [])
    .map(
      (r) =>
        `<div><span>${esc(r.filename || 'image')}</span><span>${esc(
          r.location_name || 'unknown'
        )} · ${Math.round((r.confidence || 0) * 100)}%</span></div>`
    )
    .join('');

  const clusterCards = (d.clusters || [])
    .map((c) =>
      card(
        `Cluster · ${c.images_agreeing} image(s)`,
        `<div class="kv">` +
          `<div><span>place</span><span>${esc(c.location_name || '—')}</span></div>` +
          `<div><span>combined confidence</span><span>${Math.round(
            c.combined_confidence * 100
          )}%</span></div>` +
          `<div><span>spread</span><span>${c.spread_km} km</span></div>` +
          `<div><span>images</span><span>${c.members
            .map((m) => esc(m.filename))
            .join(', ')}</span></div>` +
          `</div>`
      )
    )
    .join('');

  els.cards.innerHTML =
    card('Per-image results', `<div class="kv">${rows}</div>`, true) +
    clusterCards +
    (d.images_unlocated && d.images_unlocated.length
      ? card('Not located', list(d.images_unlocated.map((f) => esc(f || 'image'))))
      : '');

  els.rawJson.textContent = JSON.stringify(d, null, 2);
  show(els.results, true);
  els.results.scrollIntoView({ behavior: 'smooth', block: 'start' });
}
