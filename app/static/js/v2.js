// helpers for 2.0 pages. All dynamic text uses textContent (no innerHTML) to avoid injection.
function el(tag, text, cls, attrs) {
  const e = document.createElement(tag);
  if (text != null) e.textContent = text;
  if (cls) e.className = cls;
  if (attrs) for (const k in attrs) e.setAttribute(k, attrs[k]);
  return e;
}
function clear(n) { while (n.firstChild) n.removeChild(n.firstChild); return n; }
function showErr(id, msg) { const b = document.getElementById(id); b.textContent = msg || ''; b.classList.toggle('d-none', !msg); }
function table(cols, rows, id) {
  const t = el('table', null, 'table table-sm sticky-head mb-0'); if (id) t.id = id;
  const h = el('tr'); cols.forEach(c => h.appendChild(el('th', c.label))); t.appendChild(el('thead')).appendChild(h);
  const tb = el('tbody'); rows.forEach(r => { const tr = el('tr'); cols.forEach(c => tr.appendChild(td(c.f ? c.f(r) : r[c.k]))); tb.appendChild(tr); });
  t.appendChild(tb); const w = el('div', null, 'tbl-wrap'); w.appendChild(t); return w;
}
function badge(text, kind) { return el('span', text, 'badge bg-' + (kind || 'secondary') + ' me-1'); }
function confBadge(c) { return badge(c || '-', c === 'High' ? 'success' : c === 'Medium' ? 'warning text-dark' : 'secondary'); }
function bandBadge(b) { return badge(b, b === 'High' ? 'danger' : b === 'Medium' ? 'warning text-dark' : 'success'); }
function num(v, d = 3) { return v == null ? '-' : Number(v).toFixed(d); }
