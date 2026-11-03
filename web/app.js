// ZeroDefect operator UI (vanilla JS, no build step)
const $ = (s) => document.querySelector(s);
const state = { stream: null, busy: false, autoTimer: null, voice: true, status: null };

// ---------------------------------------------------------------- helpers
async function api(path, form) {
  const res = await fetch(path, form ? { method: "POST", body: form } : {});
  const ct = res.headers.get("content-type") || "";
  const body = ct.includes("json") ? await res.json() : await res.text();
  if (!res.ok) throw new Error((body && body.detail) || res.statusText);
  return body;
}
function esc(s) { return String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])); }
function say(text) {
  if (!state.voice || !text || !("speechSynthesis" in window)) return;
  speechSynthesis.cancel();
  const u = new SpeechSynthesisUtterance(text);
  u.rate = 1.05;
  speechSynthesis.speak(u);
}
function beep(ok) {
  try {
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    const o = ctx.createOscillator(), g = ctx.createGain();
    o.frequency.value = ok ? 880 : 220; o.type = ok ? "sine" : "square";
    g.gain.value = 0.08; o.connect(g); g.connect(ctx.destination); o.start();
    o.stop(ctx.currentTime + (ok ? 0.12 : 0.35));
  } catch (_) {}
}
function setStatus(el, msg, err = false) { el.textContent = msg; el.classList.toggle("err", err); }

// ---------------------------------------------------------------- tabs
document.querySelectorAll(".tab").forEach((t) => t.addEventListener("click", () => showTab(t.dataset.tab)));
function showTab(name) {
  document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t.dataset.tab === name));
  document.querySelectorAll(".panel").forEach((p) => p.classList.toggle("active", p.id === "tab-" + name));
  if (name === "report") refreshReportTab();
  if (name === "inspect") refreshCounters();
}

// ---------------------------------------------------------------- status
async function refreshStatus() {
  const s = await api("/api/status");
  state.status = s;
  const spark = $("#chip-spark"), sam = $("#chip-sam");
  spark.textContent = s.meta_api ? `Muse Spark: live (${s.spark_model})` : "Muse Spark: offline (local explainer)";
  sam.textContent = s.meta_api ? `SAM 3.1: live` : "SAM 3.1: offline (local segmenter)";
  spark.className = "chip " + (s.meta_api ? "live" : "local");
  sam.className = "chip " + (s.meta_api ? "live" : "local");
  if (s.golden) {
    $("#golden-view").className = "";
    $("#golden-view").innerHTML = `<img src="${s.golden}" alt="Golden sample">`;
    const p = s.profile;
    $("#golden-meta").textContent = `${p.name} · finds "${p.part_prompt}" · ${s.golden_count} golden view(s) · ` +
      (s.calibrated_with ? `calibrated with ${s.calibrated_with} OK part(s)` : "auto threshold");
    $("#p-name").value = p.name; $("#p-prompt").value = p.part_prompt; $("#p-notes").value = p.notes || "";
    $("#p-rot").checked = !!p.rotation_invariant;
    $("#sens").value = p.sensitivity; $("#sens-val").textContent = (+p.sensitivity).toFixed(2);
    $("#c-camera").disabled = false;
  }
  return s;
}

// ---------------------------------------------------------------- camera
async function startCamera() {
  if (state.stream) return true;
  if (!navigator.mediaDevices?.getUserMedia) {
    alert("Camera needs https or localhost. Use the Photo button instead."); return false;
  }
  try {
    state.stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: { ideal: "environment" }, width: { ideal: 1920 }, height: { ideal: 1080 } }, audio: false });
  } catch (e) { alert("Could not open camera: " + e.message); return false; }
  const v = $("#video");
  v.srcObject = state.stream; await v.play(); resetAspect();
  $("#cam-placeholder").hidden = true; $("#annotated").hidden = true;
  $("#cam-toggle").textContent = "Stop camera";
  return true;
}
function stopCamera() {
  state.stream?.getTracks().forEach((t) => t.stop());
  state.stream = null; $("#video").srcObject = null;
  $("#cam-toggle").textContent = "Start camera";
  if ($("#annotated").hidden) $("#cam-placeholder").hidden = false;
}
function grabFrame() {
  const v = $("#video"), c = $("#grab");
  c.width = v.videoWidth; c.height = v.videoHeight;
  c.getContext("2d").drawImage(v, 0, 0);
  return new Promise((r) => c.toBlob(r, "image/jpeg", 0.92));
}
function resetAspect() { $(".camwrap").style.aspectRatio = ""; }
$("#cam-toggle").onclick = () => (state.stream ? stopCamera() : startCamera());

// ---------------------------------------------------------------- golden sample
async function registerGolden(blobs) {
  const st = $("#g-status");
  const f = new FormData();
  blobs.forEach((b, i) => f.append("files", b, `golden${i}.jpg`));
  f.append("name", $("#p-name").value); f.append("part_prompt", $("#p-prompt").value);
  f.append("notes", $("#p-notes").value); f.append("rotation_invariant", $("#p-rot").checked);
  setStatus(st, "Learning what good looks like…");
  try {
    await api("/api/golden", f);
    await refreshStatus();
    // new product: clear the previous product's results from the inspect view
    $("#parts").innerHTML = ""; $("#annotated").hidden = true; $("#verdict").hidden = true;
    $("#cam-placeholder").hidden = !!state.stream; $(".camwrap").style.aspectRatio = ""; $("#meta-line").textContent = "";
    setStatus(st, "✓ Golden sample learned. Go to Inspect.");
    say("Golden sample learned.");
  } catch (e) { setStatus(st, "Error: " + e.message, true); }
}
$("#g-camera").onclick = async () => {
  showTab("inspect");
  if (!(await startCamera())) return;
  setTimeout(async () => { const b = await grabFrame(); showTab("setup"); registerGolden([b]); }, 1200);
};
$("#g-file").onchange = (e) => e.target.files.length && registerGolden([...e.target.files]);
$("#g-demo").onclick = async () => {
  const kind = { gear: "gear", washer: "washer", bracket: "bracket" }[guessKind()] || "gear";
  const b = await (await fetch(`/api/demo/frame?kind=${kind}&golden=true&seed=1`)).blob();
  registerGolden([b]);
};
function guessKind() {
  const t = ($("#p-prompt").value + " " + $("#p-name").value).toLowerCase();
  return t.includes("washer") ? "washer" : (t.includes("pcb") || t.includes("board") || t.includes("bracket")) ? "bracket" : "gear";
}

// calibration
async function calibrate(blobs) {
  const st = $("#c-status"), f = new FormData();
  blobs.forEach((b, i) => f.append("files", b, `ok${i}.jpg`));
  setStatus(st, "Calibrating…");
  try { const r = await api("/api/calibrate", f); await refreshStatus(); setStatus(st, `✓ Calibrated with ${r.calibration_parts} OK part(s).`); }
  catch (e) { setStatus(st, "Error: " + e.message, true); }
}
$("#c-file").onchange = (e) => e.target.files.length && calibrate([...e.target.files]);
$("#c-camera").onclick = async () => {
  showTab("inspect"); if (!(await startCamera())) return;
  setTimeout(async () => { const b = await grabFrame(); showTab("setup"); calibrate([b]); }, 1000);
};

// ---------------------------------------------------------------- inspect
async function inspectBlob(blob) {
  if (state.busy) return;
  if (!state.status?.ready) { alert("Register a golden sample first (step 1)."); showTab("setup"); return; }
  state.busy = true;
  const v = $("#verdict");
  v.hidden = false; v.className = "verdict busy"; v.textContent = "Inspecting…";
  $("#inspect-btn").disabled = true;
  try {
    const f = new FormData(); f.append("file", blob, "frame.jpg");
    const r = await api("/api/inspect", f);
    renderResult(r);
  } catch (e) {
    v.className = "verdict REVIEW"; v.textContent = "Error";
    $("#parts").innerHTML = `<p class="status err">${esc(e.message)}</p>`;
  } finally { state.busy = false; $("#inspect-btn").disabled = false; }
}
$("#inspect-btn").onclick = async () => {
  if (state.stream) return inspectBlob(await grabFrame());
  $("#i-file").click();
};
$("#i-file").onchange = (e) => { if (e.target.files[0]) { stopCamera(); inspectBlob(e.target.files[0]); } e.target.value = ""; };
$("#i-demo").onclick = async () => {
  stopCamera();
  const b = await (await fetch(`/api/demo/frame?kind=${guessKind()}`)).blob();
  inspectBlob(b);
};
$("#auto").onchange = (e) => {
  clearInterval(state.autoTimer);
  if (e.target.checked) state.autoTimer = setInterval(async () => {
    if (state.busy) return;
    if (state.stream) inspectBlob(await grabFrame());
    else $("#i-demo").click();
  }, 1000 * +$("#auto-int").value);
};

function sevDots(n) { let s = '<span class="sev" title="severity ' + n + '/5">'; for (let i = 1; i <= 5; i++) s += `<i class="${i <= n ? "on" : ""}"></i>`; return s + "</span>"; }

function renderResult(r) {
  const img = $("#annotated");
  img.onload = () => { $(".camwrap").style.aspectRatio = `${img.naturalWidth} / ${img.naturalHeight}`; };
  img.src = r.annotated; img.hidden = false;
  if (!state.stream) $("#cam-placeholder").hidden = true;
  if (state.stream) { // keep the live view but show the result for a moment
    $("#video").style.opacity = 0; setTimeout(() => { $("#video").style.opacity = 1; img.hidden = true; }, 2500);
  }
  const v = $("#verdict"); v.className = "verdict " + r.verdict; v.textContent = r.verdict;
  beep(r.verdict === "PASS"); say(r.speech);
  $("#meta-line").textContent = `${r.parts.length} part(s) · parts via ${r.parts_source} · ${r.elapsed_ms} ms`;
  const order = { FAIL: 0, REVIEW: 1, PASS: 2 };
  const parts = [...r.parts].sort((a, b) => order[a.verdict] - order[b.verdict] || a.index - b.index);
  $("#parts").innerHTML = parts.map((p) => {
    if (p.verdict === "PASS") return `<div class="part PASS compact"><img src="${p.crop}" alt="">
      <div><span class="title">#${p.index + 1} PASS</span><div class="small muted">${esc((p.inspector || {}).explanation || "Matches the golden sample.")}</div></div>
      <span class="tag">score ${p.score.toFixed(2)}</span>
      <button class="small ghost" onclick="feedback(${p.id}, 'defect', this)">Missed?</button></div>`;
    const ins = p.inspector || {};
    const type = (!ins.defect_type || ins.defect_type === "other") ? "defect" : ins.defect_type.replaceAll("_", " ");
    const disagree = p.detector_verdict !== (ins.verdict || p.detector_verdict) ? ` · detector said ${p.detector_verdict}` : "";
    return `<div class="part ${p.verdict}">
      <div class="head"><span class="title">#${p.index + 1} ${p.verdict}${p.verdict !== "PASS" ? " · " + esc(type) : ""}</span>
      <span class="tag">score ${p.score.toFixed(2)} · ${esc(ins.source || "")}</span></div>
      ${p.verdict !== "PASS" ? `<div class="expl">${esc(ins.explanation)}</div>
        <div class="small muted">Where: ${esc(ins.location || "–")} · severity ${sevDots(ins.severity || 0)}${p.defect_area_pct != null ? ` · ${p.defect_area_pct}% of part` : ""}${p.defect_mask_source ? ` · mask: ${esc(p.defect_mask_source)}` : ""}${esc(disagree)}</div>
        ${ins.likely_cause ? `<div class="cause">Likely cause: ${esc(ins.likely_cause)}</div>` : ""}` : `<div class="small muted">${esc(ins.explanation || "Matches the golden sample.")}</div>`}
      <div class="imgs"><img src="${p.crop}" alt="part"><img src="${p.heatmap}" alt="anomaly heatmap"></div>
      <div class="row gap">${p.verdict !== "PASS"
        ? `<button class="small secondary" onclick="feedback(${p.id}, 'ok', this)">Actually OK — learn it</button><button class="small ghost" onclick="feedback(${p.id}, 'defect', this)">Confirm defect</button>`
        : `<button class="small ghost" onclick="feedback(${p.id}, 'defect', this)">Missed a defect?</button>`}</div>
    </div>`;
  }).join("");
  refreshCounters();
}
window.feedback = async (id, label, btn) => {
  btn.disabled = true; const f = new FormData(); f.append("label", label);
  try { await api(`/api/feedback/${id}`, f); btn.textContent = label === "ok" ? "✓ Learned as OK" : "✓ Logged"; await refreshStatus(); refreshCounters(); }
  catch (e) { btn.textContent = "Error"; }
};
$("#sens").oninput = (e) => ($("#sens-val").textContent = (+e.target.value).toFixed(2));
$("#sens").onchange = async (e) => { const f = new FormData(); f.append("value", e.target.value); try { await api("/api/sensitivity", f); } catch (_) {} };

async function refreshCounters() {
  try {
    const s = (await api("/api/stats?hours=24")).summary;
    $("#k-total").textContent = s.total_parts; $("#k-pass").textContent = s.pass + s.false_alarms_corrected;
    $("#k-fail").textContent = s.fail; $("#k-review").textContent = s.review;
  } catch (_) {}
}

// ---------------------------------------------------------------- report tab
function mdToHtml(md) {
  const lines = esc(md).split("\n"); let html = "", inList = false;
  for (let l of lines) {
    l = l.replace(/\*\*(.+?)\*\*/g, "<b>$1</b>").replace(/^#{1,4}\s*(.*)$/, "<b>$1</b>");
    if (/^\s*[-*]\s+/.test(l)) { if (!inList) { html += "<ul>"; inList = true; } html += "<li>" + l.replace(/^\s*[-*]\s+/, "") + "</li>"; }
    else { if (inList) { html += "</ul>"; inList = false; } html += l.trim() ? `<p>${l}</p>` : ""; }
  }
  return html + (inList ? "</ul>" : "");
}
async function refreshReportTab() {
  const st = await api("/api/stats?hours=24");
  const hours = st.by_hour.slice(-12);
  const W = 600, H = 170, max = Math.max(1, ...hours.map((h) => h.total));
  const bw = W / Math.max(1, hours.length);
  $("#hour-chart").innerHTML = hours.length ? `<svg viewBox="0 0 ${W} ${H + 20}" preserveAspectRatio="none">${hours.map((h, i) => {
    const th = (h.total / max) * H, fh = (h.fail / max) * H, x = i * bw + 4;
    return `<rect x="${x}" y="${H - th}" width="${bw - 8}" height="${th}" fill="#1f3b2a"/><rect x="${x}" y="${H - fh}" width="${bw - 8}" height="${fh}" fill="#ef4444"/>
      <text x="${x + (bw - 8) / 2}" y="${H + 14}" fill="#8a99ad" font-size="11" text-anchor="middle">${h.hour.slice(-5)}</text>
      <text x="${x + (bw - 8) / 2}" y="${H - th - 4}" fill="#e6edf5" font-size="11" text-anchor="middle">${h.defect_rate_pct}%</text>`;
  }).join("")}</svg>` : '<p class="muted">No inspections yet.</p>';
  const bmax = Math.max(1, ...st.breakdown.map((b) => b.count));
  $("#breakdown").innerHTML = st.breakdown.length ? st.breakdown.map((b) => `<div class="bar-row"><span>${esc(b.defect_type === "other" ? "defect" : b.defect_type.replaceAll("_", " "))}</span>
    <div class="bar" style="width:${(b.count / bmax) * 100}%"></div><b>${b.count}</b></div>`).join("") : '<p class="muted">No defects logged.</p>';
  const hist = await api("/api/history?limit=40");
  $("#history").innerHTML = hist.map((h) => `<div class="hist">${h.thumb_url ? `<img src="${h.thumb_url}" alt="" loading="lazy">` : ""}
    <div><b class="${h.verdict}">${h.verdict === "PASS_OVERRIDE" ? "OK (taught)" : h.verdict}</b> ${esc(h.verdict === "PASS" ? "" : (!h.defect_type || h.defect_type === "other" ? "defect" : h.defect_type.replaceAll("_", " ")))}<br>
    <span class="muted">${new Date(h.ts * 1000).toLocaleTimeString()}</span></div></div>`).join("");
}
$("#report-btn").onclick = async () => {
  const btn = $("#report-btn"); btn.disabled = true; $("#report").innerHTML = "Muse Spark is reading the log…";
  try {
    const f = new FormData(); f.append("hours", 12);
    const r = await api("/api/report", f);
    $("#report").classList.remove("muted"); $("#report").innerHTML = mdToHtml(r.markdown);
    $("#report-trace").textContent = r.trace?.length ? `Agent used: ${r.trace.map((t) => t.tool).join(" → ")} · ${r.source}` : `source: ${r.source}`;
  } catch (e) { $("#report").textContent = "Error: " + e.message; }
  btn.disabled = false;
};
$("#reset-log").onclick = async () => { if (confirm("Clear the inspection log?")) { await api("/api/reset_log", new FormData()); refreshReportTab(); refreshCounters(); } };
$("#voice-toggle").onclick = () => { state.voice = !state.voice; $("#voice-toggle").textContent = state.voice ? "🔊 Voice on" : "🔇 Voice off"; };

refreshStatus().then((s) => { if (s.ready) showTab("inspect"); }).catch(() => {});
