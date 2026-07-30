/**
 * NeuraScan AI — app.js
 * Two-panel UI: Upload+Analysis (left) | Chat (right)
 */

/* ── State ───────────────────────────────────────── */
const App = {
  sessionId:      null,
  filename:       null,
  originalName:   null,
  zoom:           1,
  inverted:       false,
  detections:     [],
  ragContext:     [],
  modelLabel:     '',
  report:         '',
  zoomTimer:      null,
};

/* ── Init ────────────────────────────────────────── */
async function init() {
  await createSession();
  await checkStatus();
  setupUpload();
  setupChatInput();
  setupKeyboard();
}

async function createSession() {
  try {
    const r = await fetch('/api/session/new', { method: 'POST' });
    const d = await r.json();
    App.sessionId = d.session_id;
  } catch { toast('Failed to start session', 'error'); }
}

async function checkStatus() {
  try {
    const r = await fetch('/api/status');
    const d = await r.json();
    const badge = document.getElementById('model-badge');
    const dot   = document.getElementById('online-dot');
    const lbl   = document.getElementById('online-label');

    if (d.llm_available) {
      if (badge) badge.textContent = `LLM`;
      lbl && (lbl.textContent = 'Online');
    } else {
      if (badge) badge.textContent = 'Demo Mode';
      dot  && dot.classList.replace('online-dot', 'offline-dot');
      lbl  && (lbl.textContent = 'Demo');
    }
  } catch {}
}

/* ── Upload ──────────────────────────────────────── */
function setupUpload() {
  const zone  = document.getElementById('upload-zone');
  const input = document.getElementById('file-input');

  zone.addEventListener('dragenter', e => { e.preventDefault(); zone.classList.add('drag-over'); });
  zone.addEventListener('dragover',  e => { e.preventDefault(); });
  zone.addEventListener('dragleave', e => { if (!zone.contains(e.relatedTarget)) zone.classList.remove('drag-over'); });
  zone.addEventListener('drop', e => {
    e.preventDefault(); zone.classList.remove('drag-over');
    if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
  });

  input.addEventListener('change', e => { if (e.target.files[0]) handleFile(e.target.files[0]); });

  document.addEventListener('paste', e => {
    const f = Array.from(e.clipboardData?.files || []).find(f => f.type.startsWith('image/'));
    if (f) handleFile(f);
  });
}

async function handleFile(file) {
  if (!file.type.startsWith('image/')) { toast('Please upload an image file', 'error'); return; }

  const fd = new FormData();
  fd.append('image', file);

  try {
    const r = await fetch('/api/upload', { method: 'POST', body: fd });
    const d = await r.json();
    if (!d.success) throw new Error(d.error);

    App.filename     = d.filename;
    App.originalName = d.original;

    // Show preview
    showPreview(d.url, d.original);
    toast(`Uploaded: ${d.original}`, 'success');
  } catch (e) {
    toast(`Upload failed: ${e.message}`, 'error');
  }
}

function showPreview(url, name) {
  // Hide upload zone, show preview
  document.getElementById('upload-area').style.display = 'none';
  const pw = document.getElementById('preview-wrap');
  pw.classList.add('show');

  document.getElementById('preview-img').src = url;
  document.getElementById('preview-name').textContent = name;

  // Reset zoom/invert
  App.zoom = 1; App.inverted = false;
  applyImgTransform();
}

function changeImage() {
  document.getElementById('upload-area').style.display = 'block';
  document.getElementById('preview-wrap').classList.remove('show');
  document.getElementById('file-input').value = '';
  App.filename = null;
  App.detections = [];
  resetSummaryCards();
  clearReport();
}

/* ── Viewer controls ─────────────────────────────── */
function applyImgTransform() {
  const img = document.getElementById('preview-img');
  if (img) {
    img.style.transform = `scale(${App.zoom})`;
    img.style.filter    = App.inverted ? 'invert(1)' : '';
  }
  // Show zoom indicator
  const ind = document.getElementById('zoom-ind');
  if (ind) {
    ind.textContent = `${Math.round(App.zoom * 100)}%`;
    ind.style.opacity = '1';
    clearTimeout(App.zoomTimer);
    App.zoomTimer = setTimeout(() => { ind.style.opacity = '0'; }, 1200);
  }
}

function doZoom(factor) {
  App.zoom = Math.min(Math.max(App.zoom * factor, 0.2), 6);
  applyImgTransform();
}

function resetView() {
  App.zoom = 1; App.inverted = false;
  applyImgTransform();
}

function invertImg() {
  App.inverted = !App.inverted;
  const btn = document.getElementById('invert-btn');
  btn?.classList.toggle('active', App.inverted);
  applyImgTransform();
}

function setupKeyboard() {
  document.addEventListener('keydown', e => {
    if (e.target.matches('input,textarea,select')) return;
    if (e.key === '+' || e.key === '=') doZoom(1.2);
    if (e.key === '-') doZoom(0.83);
    if (e.key === '0') resetView();
    if (e.key === 'i') invertImg();
    if (e.key === 'Enter') runPipeline();
  });

  // Scroll to zoom on preview image
  document.getElementById('preview-img-container')?.addEventListener('wheel', e => {
    e.preventDefault();
    doZoom(e.deltaY < 0 ? 1.1 : 0.9);
  }, { passive: false });
}

/* ── Pipeline: Detect → Analyze ─────────────────── */
async function runPipeline() {
  if (!App.filename) { toast('Upload an image first', 'error'); return; }

  

  setProcessing(true);

  try {
    // Step 1: YOLO + RAG
    setLabel('Running YOLOv8…');
    await stepAnim(['ps1', 'ps2']);

    const dr = await fetch('/api/detect', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        filename: App.filename, model: 'brain_tumor',
        session_id: App.sessionId,
      }),
    });
    const dd = await dr.json();
    if (!dd.success) throw new Error(dd.error || 'Detection failed');

    await stepAnim(['ps3']);

    App.detections  = dd.detections;
    App.ragContext  = dd.rag_context;
    App.modelLabel  = dd.model_label;

    // Update summary cards immediately
    updateSummaryCards(dd);

    // Show annotated image if available
    if (dd.annotated_url) {
      document.getElementById('preview-img').src = dd.annotated_url + '?t=' + Date.now();
    }

    // Step 2: LLM
    setLabel('LLM generating report…');
    await stepAnim(['ps4']);

    const lr = await fetch('/api/analyze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        filename:    App.filename,
        detections:  dd.detections,
        rag_context: dd.rag_context,
        model_label: dd.model_label,
        session_id:  App.sessionId,
      }),
    });
    const ld = await lr.json();
    if (!ld.success) throw new Error(ld.error || 'LLM failed');

    App.report = ld.report;
    renderReport(ld.report, ld.demo_mode);

    toast('Analysis complete ✓', 'success');

    // Add chat message prompting follow-ups
    addAIMessage(
      `✅ **Analysis complete!**\n\nI've detected **${dd.detections.length > 0 ? dd.detections.map(d => d.class).join(', ') : 'No tumor'}** in the MRI scan.\n\nYou can now:\n- Ask me about the **treatment options**\n- Ask about **medicines and dosages**\n- Type a **city name** to find specialists near you\n- Ask anything about the findings`
    );

  } catch (err) {
    toast(`Error: ${err.message}`, 'error');
  } finally {
    setProcessing(false);
  }
}

/* ── Summary Cards ───────────────────────────────── */
function updateSummaryCards(data) {
  const s   = data.summary;
  const det = data.detections;

  // Modality
  setCard('card-modality', 'Brain MRI', 'normal');

  // Detected class
  if (det.length === 0) {
    setCard('card-findings', 'No Tumor Detected', 'normal');
    setCard('card-confidence', '—', '');
    setCard('card-urgency', 'Routine', 'routine');
  } else {
    const top = det[0];
    setCard('card-findings', top.class, top.class === 'No Tumor' ? 'normal' : 'detected');
    setCard('card-confidence', `${(top.confidence * 100).toFixed(0)}%`,
      top.confidence > 0.8 ? 'detected' : top.confidence > 0.5 ? 'moderate' : 'normal');
    setCard('card-urgency',
      top.class === 'No Tumor' ? 'Routine' :
      top.class === 'Glioma'   ? 'URGENT'  : 'Moderate',
      top.class === 'No Tumor' ? 'routine' :
      top.class === 'Glioma'   ? 'urgent'  : 'moderate'
    );
  }

  // Detection badges
  const badgeArea = document.getElementById('det-badges');
  if (badgeArea && det.length > 0) {
    badgeArea.innerHTML = det.map(d => {
      const cls = d.class.replace(' ', '').toLowerCase();
      return `<span class="det-badge badge-${cls}">${d.class} · ${(d.confidence*100).toFixed(0)}%</span>`;
    }).join('');
    document.getElementById('badge-row').style.display = 'block';
  }
}

function setCard(id, value, statusClass) {
  const el = document.getElementById(id);
  if (!el) return;
  el.textContent = value;
  el.className = `sc-value ${statusClass}`;
}

function resetSummaryCards() {
  ['card-modality','card-findings','card-confidence','card-urgency'].forEach(id => {
    setCard(id, '—', 'empty');
  });
  const br = document.getElementById('badge-row');
  if (br) br.style.display = 'none';
}

/* ── Report Rendering ────────────────────────────── */
function renderReport(markdown, demoMode) {
  const area = document.getElementById('report-scroll');
  area.innerHTML = '';

  if (demoMode) {
    const warn = document.createElement('div');
    warn.className = 'demo-warn';
    warn.innerHTML = '⚠ Demo mode — add <code>GOOGLE_API_KEY</code> to <code>.env</code> for real LLM analysis';
    area.appendChild(warn);
  }

  // Parse sections from markdown and render as collapsible cards
  const sections = parseSections(markdown);

  const sectionMeta = [
    { key: '1.',  css: 'rs-overview',  icon: '🧠', title: 'Image Overview' },
    { key: '2.',  css: 'rs-findings',  icon: '🔍', title: 'Key Findings' },
    { key: '3.',  css: 'rs-diagnosis', icon: '🏥', title: 'Diagnosis & Confidence' },
    { key: '4.',  css: 'rs-stage',     icon: '📊', title: 'Stage & Severity' },
    { key: '5.',  css: 'rs-causes',    icon: '🦠', title: 'Causes & Risk Factors' },
    { key: '6.',  css: 'rs-treatment', icon: '💊', title: 'Treatment Options' },
    { key: '7.',  css: 'rs-medicines', icon: '💉', title: 'Medicines & Dosages' },
    { key: '8.',  css: 'rs-prognosis', icon: '📈', title: 'Prognosis' },
    { key: '9.',  css: 'rs-patient',   icon: '🤝', title: 'Patient Summary' },
  ];

  if (sections.length === 0) {
    // Fallback: render whole thing in one card
    const card = makeSection('rs-overview', '📋', 'Full Report', markdownToHtml(markdown), false);
    area.appendChild(card);
    return;
  }

  sections.forEach((sec, i) => {
    const meta = sectionMeta.find(m => sec.title.includes(m.key)) || { css: 'rs-overview', icon: '📄', title: sec.title };
    const card = makeSection(meta.css, meta.icon, sec.title, markdownToHtml(sec.body), i > 2);
    area.appendChild(card);
  });
}

function makeSection(cssClass, icon, title, bodyHtml, collapsed) {
  const div = document.createElement('div');
  div.className = `report-section ${cssClass}`;
  div.innerHTML = `
    <div class="rs-header ${collapsed ? 'collapsed' : ''}" onclick="toggleSection(this)">
      <span class="rs-icon">${icon}</span>
      <span class="rs-title">${escHtml(title)}</span>
      <span class="rs-chevron">▼</span>
    </div>
    <div class="rs-body" style="${collapsed ? 'display:none' : ''}">${bodyHtml}</div>
  `;
  return div;
}

function toggleSection(header) {
  const body = header.nextElementSibling;
  const collapsed = body.style.display === 'none';
  body.style.display = collapsed ? 'block' : 'none';
  header.classList.toggle('collapsed', !collapsed);
}

function clearReport() {
  const area = document.getElementById('report-scroll');
  area.innerHTML = `
    <div class="report-placeholder">
      <span class="report-placeholder-icon">📋</span>
      Upload an image and click <strong>Analyze</strong> to generate a full clinical report
      including diagnosis, stage, treatment options, medicines, and prognosis.
    </div>`;
}

/* ── Parse markdown sections ─────────────────────── */
function parseSections(md) {
  const lines    = md.split('\n');
  const sections = [];
  let current    = null;

  for (const line of lines) {
    const h3 = line.match(/^###\s+(.+)/);
    if (h3) {
      if (current) sections.push(current);
      current = { title: h3[1].replace(/[🧠🔍🏥📊🦠💊💉📈🤝]/g, '').trim(), body: '' };
    } else if (current) {
      current.body += line + '\n';
    }
  }
  if (current) sections.push(current);
  return sections;
}

/* ── Chat ────────────────────────────────────────── */
function setupChatInput() {
  const inp = document.getElementById('chat-inp');
  inp?.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendChat(); }
  });
  inp?.addEventListener('input', () => {
    inp.style.height = 'auto';
    inp.style.height = Math.min(inp.scrollHeight, 100) + 'px';
  });
}

function sendChip(text) {
  const inp = document.getElementById('chat-inp');
  if (inp) inp.value = text;
  sendChat();
}

async function sendChat() {
  const inp = document.getElementById('chat-inp');
  const q   = inp?.value.trim();
  if (!q) return;
  inp.value = ''; inp.style.height = 'auto';

  // Hide welcome if shown
  document.getElementById('chat-welcome')?.remove();

  addUserMessage(q);
  showTyping();

  document.getElementById('chat-send').disabled = true;

  try {
    const r = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question: q, session_id: App.sessionId }),
    });
    const d = await r.json();

hideTyping();

if (!d || typeof d !== "object") {

    addAIMessage("⚠️ Server error");

    return;

}

if (d.error) {

    addAIMessage("⚠️ " + d.error);

    return;

}

addAIMessage(d.response || "No response");
  } catch (e) {
    hideTyping();
    addAIMessage(`⚠️ Error: ${e.message}`);
  } finally {
    document.getElementById('chat-send').disabled = false;
    document.getElementById('chat-inp')?.focus();
  }
}

function addUserMessage(text) {
  const msgs = document.getElementById('chat-messages');
  const div  = document.createElement('div');
  div.className = 'msg user';
  div.innerHTML = `
    <div class="msg-avatar user">You</div>
    <div class="msg-bubble">${escHtml(text)}</div>`;
  msgs.appendChild(div);
  msgs.scrollTop = msgs.scrollHeight;
}

function addAIMessage(text) {
  const msgs = document.getElementById('chat-messages');
  const div  = document.createElement('div');
  div.className = 'msg';
  div.innerHTML = `
    <div class="msg-avatar ai">🤖</div>
    <div class="msg-bubble">${markdownToHtml(text)}</div>`;
  msgs.appendChild(div);
  msgs.scrollTop = msgs.scrollHeight;
}

function showTyping() {
  const msgs = document.getElementById('chat-messages');
  const div  = document.createElement('div');
  div.className = 'msg'; div.id = 'typing-msg';
  div.innerHTML = `
    <div class="msg-avatar ai">🤖</div>
    <div class="msg-bubble">
      <div class="typing-bubble">
        <div class="t-dot"></div><div class="t-dot"></div><div class="t-dot"></div>
      </div>
    </div>`;
  msgs.appendChild(div);
  msgs.scrollTop = msgs.scrollHeight;
}

function hideTyping() { document.getElementById('typing-msg')?.remove(); }

/* ── Processing overlay ──────────────────────────── */
function setProcessing(on) {
  const ov  = document.getElementById('proc-overlay');
  const btn = document.getElementById('run-btn');
  ov?.classList.toggle('active', on);
  if (btn) btn.disabled = on;
  if (!on) {
    ['ps1','ps2','ps3','ps4'].forEach(id => {
      const el = document.getElementById(id);
      el?.classList.remove('active','done');
    });
  }
}

function setLabel(t) {
  const el = document.getElementById('proc-label');
  if (el) el.textContent = t;
}

async function stepAnim(ids) {
  for (const id of ids) {
    const el = document.getElementById(id);
    if (!el) continue;
    el.classList.add('active');
    await sleep(260);
    el.classList.remove('active');
    el.classList.add('done');
  }
}

/* ── Markdown → HTML ─────────────────────────────── */
function markdownToHtml(md) {
  if (!md) return '';
  let h = md
    .replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');

  // Headings
  h = h.replace(/^#{3}\s+(.+)$/gm, '<h3>$1</h3>');
  h = h.replace(/^#{2}\s+(.+)$/gm, '<h3>$1</h3>');
  h = h.replace(/^#{1}\s+(.+)$/gm, '<h3>$1</h3>');

  // Bold / italic / code
  h = h.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
  h = h.replace(/\*(.+?)\*/g,     '<em>$1</em>');
  h = h.replace(/`(.+?)`/g,       '<code>$1</code>');

  // Simple markdown table → HTML table
  h = h.replace(/(\|.+\|\n\|[-| :]+\|\n(?:\|.+\|\n?)+)/g, mdTable => {
    const rows = mdTable.trim().split('\n');
    const head = rows[0].split('|').filter(c => c.trim()).map(c => `<th>${c.trim()}</th>`).join('');
    const body = rows.slice(2).map(r =>
      '<tr>' + r.split('|').filter(c => c.trim()).map(c => `<td>${c.trim()}</td>`).join('') + '</tr>'
    ).join('');
    return `<table class="med-table"><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table>`;
  });

  // Lists
  h = h.replace(/^[-*]\s+(.+)$/gm, '<li>$1</li>');
  h = h.replace(/^\d+\.\s+(.+)$/gm, '<li>$1</li>');
  h = h.replace(/<\/li>\n<li>/g, '</li><li>');

  // HR
  h = h.replace(/^---+$/gm, '<hr>');

  // Paragraphs
  h = h.replace(/\n\n/g, '</p><p>');
  h = h.replace(/\n/g, '<br>');

  return h;
}

/* ── Utils ───────────────────────────────────────── */
function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

function escHtml(s) {
  const d = document.createElement('div');
  d.appendChild(document.createTextNode(String(s ?? '')));
  return d.innerHTML;
}

function toast(msg, type = 'info') {
  let c = document.getElementById('toast-container');
  if (!c) { c = document.createElement('div'); c.id = 'toast-container'; c.className = 'toast-container'; document.body.appendChild(c); }
  const t = document.createElement('div');
  t.className = `toast ${type}`;
  t.innerHTML = `<span>${type === 'success' ? '✓' : type === 'error' ? '✕' : 'ℹ'}</span><span>${escHtml(msg)}</span>`;
  c.appendChild(t);
  setTimeout(() => t.remove(), 3500);
}

/* ── Boot ────────────────────────────────────────── */
document.addEventListener('DOMContentLoaded', init);
