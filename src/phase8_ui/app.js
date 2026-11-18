/* PathSnap evaluator dashboard — talks to the FastAPI backend (src/phase7_api). */
(function () {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const COLORS = { gps: "#1f6feb", ml: "#f59e0b", hmm: "#059669", ref: "#7c3aed", chg: "#e11d48" };

  // ---------- backend URL: ?api=... > saved > same origin (served by API) > localhost ----------
  function defaultApi() {
    const q = new URLSearchParams(location.search).get("api");
    if (q) return q;
    try { const s = localStorage.getItem("pathsnap.api"); if (s) return s; } catch (e) { /* storage blocked */ }
    if (location.protocol.startsWith("http") && location.pathname.startsWith("/ui")) return location.origin;
    return "http://localhost:8000";
  }
  let API = defaultApi().replace(/\/+$/, "");
  $("apiUrl").value = API;

  async function call(path, opts) {
    const res = await fetch(API + path, Object.assign({ headers: { "Content-Type": "application/json" } }, opts || {}));
    let body = null;
    try { body = await res.json(); } catch (e) { /* non-JSON */ }
    if (!res.ok) {
      const d = body && body.detail;
      const msg = typeof d === "string" ? d : d ? JSON.stringify(d) : res.status + " " + res.statusText;
      throw new Error(msg);
    }
    return body;
  }

  // ---------- map ----------
  const map = L.map("map", { preferCanvas: true, zoomControl: true }).setView([0, 0], 2);
  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 19, attribution: "&copy; OpenStreetMap contributors",
  }).addTo(map);
  const layers = {  // draw order: GPS underneath, predictions on top
    gps: L.layerGroup().addTo(map),
    ref: L.layerGroup().addTo(map),
    ml: L.layerGroup().addTo(map),
    hmm: L.layerGroup().addTo(map),
    chg: L.layerGroup().addTo(map),
  };
  L.control.layers(null, {
    "Ground truth (reference route)": layers.ref,
    "ML-only prediction": layers.ml,
    "HMM / Viterbi prediction": layers.hmm,
    "GPS observations": layers.gps,
    "Points changed by HMM": layers.chg,
  }, { collapsed: false, position: "topright" }).addTo(map);
  const renderer = L.canvas({ padding: 0.3 });

  function clearMap() { Object.values(layers).forEach((g) => g.clearLayers()); }

  // ---------- status / trips / metrics ----------
  function setStatus(kind, text) {
    const el = $("status");
    el.className = "pill " + (kind === "ok" ? "pill-ok" : kind === "bad" ? "pill-bad" : "pill-wait");
    el.textContent = text;
  }
  function showError(msg) { const e = $("error"); e.textContent = msg; e.hidden = !msg; }
  const pct = (x) => (x === null || x === undefined ? "–" : (100 * x).toFixed(2) + "%");
  const ms = (x) => (x === null || x === undefined ? "–" : x >= 1000 ? (x / 1000).toFixed(2) + " s" : Math.round(x) + " ms");

  async function connect() {
    showError("");
    setStatus("wait", "Checking backend…");
    $("runBtn").disabled = true;
    try {
      const h = await call("/health");
      if (h.model_loaded) setStatus("ok", "Connected · " + h.model + " + " + h.decoder);
      else setStatus("bad", "Backend up, model NOT loaded");
      if (h.error) showError("Backend: " + h.error);
      const t = await call("/trips");
      const sel = $("tripSelect");
      sel.innerHTML = "";
      t.trips.forEach((tr) => {
        const o = document.createElement("option");
        o.value = tr.trip_id;
        o.textContent = tr.trip_id + " · " + tr.n_points + " GPS points";
        sel.appendChild(o);
      });
      sel.disabled = t.trips.length === 0;
      $("runBtn").disabled = !h.model_loaded || t.trips.length === 0;
      loadMetrics();
    } catch (e) {
      setStatus("bad", "Backend unreachable");
      showError("Cannot reach " + API + " (" + e.message + ").\nStart the backend and check the URL / firewall.");
    }
  }

  async function loadMetrics() {
    try {
      const m = await call("/metrics");
      const mc = m.model_comparison;
      if (mc) {
        const rows = mc.models.map((r) =>
          "<tr class='" + (r.model === mc.best_model ? "best" : "") + "'><td>" + r.model.replace(/_/g, " ") +
          "</td><td>" + pct(r.cv_top1_mean) + "</td><td>" + pct(r.test_top1) + "</td><td>" +
          (r.test_logloss != null ? r.test_logloss.toFixed(4) : "–") + "</td></tr>").join("");
        $("modelTable").innerHTML = "<table class='mtable small'><thead><tr><th>Model</th><th>CV Top-1</th><th>Test Top-1</th><th>Log loss</th></tr></thead><tbody>" +
          rows + "</tbody></table><p class='note'>Selected by grouped-CV Top-1 (80 train trips): <strong>" + mc.best_model + "</strong>.</p>";
      }
      const he = m.hmm_evaluation;
      if (he && he.overall) {
        const o = he.overall;
        $("hmmOverall").innerHTML = "<p class='note'>All " + he.trips + " student-test trips (" + o.hmm.points +
          " labelled points): ML-only <strong>" + pct(o.ml_only.top1_accuracy) + "</strong> → HMM <strong>" +
          pct(o.hmm.top1_accuracy) + "</strong> Top-1; continuity (≤3 arcs) " + pct(o.ml_only.continuity_reach3) +
          " → " + pct(o.hmm.continuity_reach3) + ".</p>";
      }
    } catch (e) {
      $("modelTable").textContent = "Metrics unavailable: " + e.message;
    }
  }

  // ---------- run ----------
  function drawLines(group, lines, style, label) {
    lines.forEach((ln) => L.polyline(ln, Object.assign({ renderer }, style)).bindTooltip(label, { sticky: true }).addTo(group));
  }

  function render(r) {
    clearMap();
    $("mapEmpty").hidden = true;
    const ml = r.ml_only_arc_per_point, hmm = r.hmm_arc_per_point;
    r.gps.forEach((p, i) => {  // GPS first so the routes are drawn on top
      const tip = "Point " + i + "<br>ML-only arc: " + ml[i] + "<br>HMM arc: " + hmm[i];
      L.circleMarker(p, { renderer, radius: 2, color: COLORS.gps, weight: 0, fillOpacity: 0.45 }).bindTooltip(tip).addTo(layers.gps);
    });
    if (r.reference_geometry) drawLines(layers.ref, r.reference_geometry, { color: COLORS.ref, weight: 10, opacity: 0.35 }, "Ground truth (reference route)");
    drawLines(layers.ml, r.ml_only_geometry, { color: COLORS.ml, weight: 4, opacity: 0.95, dashArray: "6 6" }, "ML-only prediction");
    drawLines(layers.hmm, r.hmm_geometry, { color: COLORS.hmm, weight: 3, opacity: 1 }, "HMM / Viterbi prediction");
    r.gps.forEach((p, i) => {
      const tip = "Point " + i + "<br>ML-only arc: " + ml[i] + "<br>HMM arc: " + hmm[i];
      if (ml[i] !== hmm[i]) {
        L.circleMarker(p, { renderer, radius: 5, color: COLORS.chg, weight: 1.5, fill: false }).bindTooltip(tip + "<br><em>changed by HMM</em>").addTo(layers.chg);
      }
    });
    if (r.gps.length) map.fitBounds(L.latLngBounds(r.gps), { padding: [30, 30] });

    $("resTrip").textContent = r.trip_id;
    const m = r.metrics;
    const t = r.timing_ms;
    $("kTime").textContent = ms(t.inference_total);
    const rows = $("metricRows");
    rows.innerHTML = "";
    if (m) {
      $("kMl").textContent = pct(m.ml_only.top1_accuracy);
      $("kHmm").textContent = pct(m.hmm.top1_accuracy);
      const spec = [
        ["Top-1 accuracy (points)", "top1_accuracy", true],
        ["Continuity – direct", "continuity_direct", true],
        ["Continuity – ≤3 arcs", "continuity_reach3", true],
        ["Route arc precision", "route_precision", true],
        ["Route arc recall", "route_recall", true],
        ["Route length F1", "route_length_f1", true],
      ];
      spec.forEach(([label, key, higher]) => {
        const a = m.ml_only[key], b = m.hmm[key];
        const tr = document.createElement("tr");
        const cells = [label, pct(a), pct(b)];
        cells.forEach((c, j) => {
          const td = document.createElement("td");
          td.textContent = c;
          if (a != null && b != null && a !== b && ((j === 2 && (higher ? b > a : b < a)) || (j === 1 && (higher ? a > b : a < b)))) td.className = "win";
          tr.appendChild(td);
        });
        rows.appendChild(tr);
      });
    } else {
      $("kMl").textContent = "n/a"; $("kHmm").textContent = "n/a";
    }
    const d = $("details");
    d.innerHTML = "";
    const det = [
      ["GPS points", r.n_points],
      ["Points with candidates", r.n_points_with_candidates],
      ["Candidate rows scored", r.n_candidate_rows],
      ["Road arcs in network", r.n_arcs_in_network],
      ["Points changed by HMM", r.points_changed_by_hmm],
      ["Viterbi chains", r.hmm_diagnostics.chains],
      ["Disconnected steps", r.hmm_diagnostics.disconnected_steps],
      ["Load", ms(t.load)], ["Candidates", ms(t.candidates)], ["Features", ms(t.features)],
      ["LightGBM", ms(t.model)], ["HMM / Viterbi", ms(t.hmm)], ["Server round trip", ms(r.server_ms)],
      ["Cached response", r.cached ? "yes" : "no"],
    ];
    det.forEach(([k, v]) => {
      const dt = document.createElement("dt"); dt.textContent = k;
      const dd = document.createElement("dd"); dd.textContent = v === undefined ? "–" : v;
      d.appendChild(dt); d.appendChild(dd);
    });
    $("resultCard").hidden = false;
  }

  async function run() {
    const trip = $("tripSelect").value;
    if (!trip) return;
    showError("");
    $("loading").hidden = false;
    $("runBtn").disabled = true;
    try {
      const r = await call("/predict", { method: "POST", body: JSON.stringify({ trip_id: trip }) });
      render(r);
    } catch (e) {
      showError("Prediction failed: " + e.message);
    } finally {
      $("loading").hidden = true;
      $("runBtn").disabled = false;
    }
  }

  function reset() {
    clearMap();
    showError("");
    $("resultCard").hidden = true;
    $("mapEmpty").hidden = false;
    map.setView([0, 0], 2);
  }

  $("runBtn").addEventListener("click", run);
  $("resetBtn").addEventListener("click", reset);
  $("connectBtn").addEventListener("click", () => {
    API = $("apiUrl").value.trim().replace(/\/+$/, "");
    try { localStorage.setItem("pathsnap.api", API); } catch (e) { /* ignore */ }
    connect();
  });
  connect();
})();
