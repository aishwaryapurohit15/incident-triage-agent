/* Incident Triage Agent: page logic, charts and the in-browser triage model. */
(function () {
  "use strict";
  var D = window.SITE_DATA;
  var M = window.WEB_MODEL;
  if (!D) return;

  // ---------------------------------------------------------------------
  // Helpers
  // ---------------------------------------------------------------------
  function get(path, obj) {
    return path.split(".").reduce(function (o, k) { return o == null ? undefined : o[k]; }, obj || D);
  }
  function css(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }
  var MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
  function monthLabel(ym) { var p = ym.split("-"); return MONTHS[+p[1] - 1] + " " + p[0].slice(2); }
  var fmt = {
    int: function (v) { return Math.round(v).toLocaleString("en-US"); },
    pct0: function (v) { return Math.round(v * 100) + "%"; },
    pct1: function (v) { return (v * 100).toFixed(1) + "%"; },
    pct2: function (v) { return (v * 100).toFixed(2) + "%"; },
    dec2: function (v) { return Number(v).toFixed(2); },
    dec3: function (v) { return Number(v).toFixed(3); },
    usdm: function (v) { return "$" + (v / 1e6).toFixed(1) + "M"; },
    usdk: function (v) { return v >= 1e6 ? "$" + (v / 1e6).toFixed(1) + "M" : "$" + Math.round(v / 1e3) + "K"; },
    date: function (v) { var d = new Date(v + "T00:00:00"); return MONTHS[d.getMonth()] + " " + d.getDate() + ", " + d.getFullYear(); },
    len: function (v) { return String(v.length); }
  };
  function el(tag, attrs, html) {
    var e = document.createElement(tag);
    if (attrs) Object.keys(attrs).forEach(function (k) { e.setAttribute(k, attrs[k]); });
    if (html != null) e.innerHTML = html;
    return e;
  }
  function esc(s) { return String(s).replace(/[&<>"]/g, function (c) { return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]; }); }
  var SEVS = ["SEV-1", "SEV-2", "SEV-3", "SEV-4"];
  function sevColor(s) { return css("--sev" + s.slice(-1)); }

  // Bind every data-k element to its value.
  document.querySelectorAll("[data-k]").forEach(function (n) {
    var v = get(n.getAttribute("data-k"));
    if (v === undefined || v === null) return;
    var f = n.getAttribute("data-f");
    n.textContent = f && fmt[f] ? fmt[f](v) : v;
  });

  // ---------------------------------------------------------------------
  // Static tables
  // ---------------------------------------------------------------------
  var methods = [
    ["majority_class", "Always predict the most common severity", "No skill; sets the floor"],
    ["keyword_rules", "Keyword rules", "Transparent and free, but misses every SEV-1 and drifts as wording changes"],
    ["tfidf_baseline", "TF-IDF + logistic regression (champion)", "Fast, cheap, explainable by word weights; chosen on validation cost"],
    ["tfidf_challenger", "TF-IDF + character n-grams + service metadata", "More features, more complexity; no gain in cost or SEV-1 recall"]
  ];
  var mt = document.querySelector("#method-table tbody");
  methods.forEach(function (m) {
    var r = D.test.severity[m[0]];
    var tr = el("tr", m[0] === D.validation.champion ? { "class": "champion" } : null);
    tr.innerHTML = "<td><b>" + m[1] + "</b></td><td class='r'>" + fmt.pct1(r.accuracy) + "</td><td class='r'>" + fmt.dec2(r.macro_f1) +
      "</td><td class='r'>" + fmt.pct0(r.sev1_recall) + "</td><td class='r'>" + fmt.pct1(r.under_triage_rate) + "</td><td class='r'><b>" +
      fmt.dec2(r.cost_per_ticket) + "</b></td><td class='small'>" + m[2] + "</td>";
    mt.appendChild(tr);
  });

  function renderCM() {
    var cm = document.getElementById("cm");
    cm.innerHTML = "";
    var m = D.confusion_matrix.matrix;
    var max = Math.max.apply(null, m.map(function (r) { return Math.max.apply(null, r); }));
    cm.appendChild(el("div", { "class": "h" }, "Rec. / Pred."));
    SEVS.forEach(function (s) { cm.appendChild(el("div", { "class": "h" }, s)); });
    m.forEach(function (row, i) {
      cm.appendChild(el("div", { "class": "h", style: "text-align:left" }, SEVS[i]));
      row.forEach(function (v, j) {
        var a = 0.08 + 0.8 * (v / max);
        var bg = i === j ? css("--ok") : (j > i ? css("--sev1") : css("--sev2"));
        var c = el("div", { "class": "c", title: SEVS[i] + " recorded, " + SEVS[j] + " suggested: " + v }, String(v));
        c.style.background = "color-mix(in srgb, " + bg + " " + Math.round(a * 100) + "%, transparent)";
        c.style.color = a > 0.55 ? css("--surface") : css("--ink");
        cm.appendChild(c);
      });
    });
  }

  function bars(container, rows, wide) {
    var c = document.getElementById(container);
    c.innerHTML = "";
    rows.forEach(function (r) {
      var b = el("div", { "class": "bar" + (wide ? " wide" : "") });
      b.innerHTML = "<span class='lab' title='" + esc(r[0]) + "'>" + esc(r[0]) + "</span><span class='track'><span class='fill' style='width:" +
        Math.max(0.5, r[1] * 100).toFixed(1) + "%;background:" + (r[2] || "var(--accent)") + "'></span></span><span class='pv'>" + fmt.pct0(r[1]) + "</span>";
      c.appendChild(b);
    });
  }
  bars("team-methods", [
    ["Keyword rules", D.test.team.keyword_rules.accuracy, "var(--faint)"],
    ["TF-IDF baseline", D.test.team.tfidf_baseline.accuracy],
    ["TF-IDF challenger", D.test.team.tfidf_challenger.accuracy, "var(--faint)"]
  ], true);
  bars("route-bars", [
    ["Fast path", D.routing.share.fast_path, "var(--ok)"],
    ["Full review", D.routing.share.full_review, "var(--watch)"],
    ["Escalate now", D.routing.share.escalate_now, "var(--alert)"]
  ], true);

  var st = document.querySelector("#sla-table tbody");
  D.risk.sla_by_severity.forEach(function (r) {
    st.appendChild(el("tr", null, "<td class='mono'>" + r.severity + "</td><td class='r'>" + r.median_hours.toFixed(1) +
      "</td><td class='r'>" + r.p90_hours.toFixed(1) + "</td><td class='r'>" + r.sla_hours + " h</td><td class='r'><b>" + fmt.pct1(r.breach_rate) + "</b></td>"));
  });
  document.getElementById("worst-team").textContent = D.risk.sla_by_team[0].team + " (" + fmt.pct1(D.risk.sla_by_team[0].breach_rate) + ")";

  function renderHeat() {
    var h = document.getElementById("heat");
    h.innerHTML = "";
    var hm = D.risk.heatmap;
    var order = hm.services.map(function (s, i) { return [s, hm.counts[i]]; })
      .sort(function (a, b) { return (b[1][0] * 4 + b[1][1]) - (a[1][0] * 4 + a[1][1]); });
    var max = Math.max.apply(null, hm.counts.map(function (r) { return Math.max.apply(null, r); }));
    h.appendChild(el("div", { "class": "h", style: "text-align:left" }, "Service"));
    SEVS.forEach(function (s) { h.appendChild(el("div", { "class": "h" }, s)); });
    order.forEach(function (row) {
      h.appendChild(el("div", { "class": "rl", title: row[0] }, esc(row[0])));
      row[1].forEach(function (v) {
        var a = Math.sqrt(v / max);
        var c = el("div", { "class": "c" }, fmt.int(v));
        c.style.background = "color-mix(in srgb, " + css("--accent") + " " + Math.round(6 + a * 80) + "%, transparent)";
        c.style.color = a > 0.6 ? css("--surface") : css("--ink");
        h.appendChild(c);
      });
    });
  }

  var ct = document.querySelector("#cluster-table tbody");
  D.clusters.slice(0, 10).forEach(function (c) {
    var terms = c.top_terms.filter(function (t) { return !/\bnum\b|changeid/.test(t); }).slice(0, 4).join(", ");
    var trend = c.trend_ratio_last6_vs_first6;
    ct.appendChild(el("tr", null, "<td><b>" + esc(c.label) + "</b><br><span class='small'>" + esc(terms) + "</span></td><td>" + esc(c.top_service) + "</td><td class='r'>" + fmt.int(c.incidents) + "</td><td class='r'>" +
      fmt.pct0(c.sev1_sev2_share) + "</td><td class='r'>" + fmt.pct0(c.repeat_within_7_days) + "</td><td class='r'" + (trend > 1.2 ? " style='color:var(--alert);font-weight:600'" : "") + ">" +
      trend.toFixed(2) + "</td><td class='r'>" + fmt.usdk(c.total_loss_usd) + "</td>"));
  });

  var tt = document.querySelector("#tool-table tbody");
  D.agent.tools.forEach(function (t) {
    tt.appendChild(el("tr", null, "<td class='mono'>" + t.name + "</td><td><span class='tag'>" + t.permission + "</span></td><td><span class='tag " +
      (t.granted ? "granted'>Allowed" : "denied'>Refused") + "</span></td>"));
  });
  document.getElementById("refusal").textContent = "PermissionDenied: " + D.agent.refusal_example;

  var routeName = { fast_path: "Fast path", full_review: "Full review", escalate_now: "Escalate now" };
  var et = document.querySelector("#example-table tbody");
  D.agent.examples.forEach(function (e) {
    var sim = e.similar_incidents[0];
    et.appendChild(el("tr", null, "<td><span class='mono small'>" + e.incident_id + "</span><br>" + esc(e.ticket.short_description) + "</td><td>" + esc(e.ticket.service) +
      "</td><td><span class='mono'>" + e.severity + "</span> <span class='small'>" + fmt.pct0(e.severity_confidence) + "</span><br><span class='small'>" + esc(e.team) +
      "</span></td><td><span class='mono'>" + e.human_recorded_severity + "</span><br><span class='small'>" + esc(e.human_recorded_team || "Unassigned") +
      "</span></td><td><span class='route " + e.review_route + "'>" + routeName[e.review_route] + "</span></td><td class='small'>" +
      (sim ? esc(sim.short_description) + " <span class='mono'>(" + sim.similarity.toFixed(2) + ")</span>" : "None") + "</td>"));
  });

  var mo = document.querySelector("#mon-table tbody");
  D.monitoring.months.forEach(function (m) {
    mo.appendChild(el("tr", { title: m.flags.join("; ") }, "<td class='mono'>" + monthLabel(m.month) + "</td><td class='r'>" + fmt.int(m.tickets) + "</td><td class='r'>" +
      fmt.pct1(m.override_rate) + "</td><td class='r'>" + fmt.pct1(m.team_override_rate) + "</td><td class='r'>" + fmt.dec3(m.psi_category_mix) + "</td><td class='r'>" +
      fmt.dec3(m.psi_predicted_severity) + "</td><td class='r'>" + fmt.pct0(m.fast_path_share) + "</td><td><span class='status " + m.status + "'>" + { ok: "OK", watch: "Watch", alert: "Alert" }[m.status] +
      "</span>" + (m.flags.length ? "<br><span class='small'>" + esc(m.flags.join("; ")) + "</span>" : "") + "</td>"));
  });

  // ---------------------------------------------------------------------
  // Charts: small dependency-free SVG renderer, themed from CSS tokens
  // ---------------------------------------------------------------------
  var NS = "http://www.w3.org/2000/svg";
  function S(tag, attrs, parent, text) {
    var e = document.createElementNS(NS, tag);
    Object.keys(attrs || {}).forEach(function (k) { e.setAttribute(k, attrs[k]); });
    if (text != null) e.textContent = text;
    if (parent) parent.appendChild(e);
    return e;
  }
  function niceMax(v) {
    if (v <= 0) return 1;
    var p = Math.pow(10, Math.floor(Math.log10(v))), n = v / p;
    return (n <= 1 ? 1 : n <= 2 ? 2 : n <= 2.5 ? 2.5 : n <= 5 ? 5 : 10) * p;
  }

  // Vertical chart: lines and/or stacked bars over categorical x positions.
  function chart(id, o) {
    var host = document.getElementById(id);
    host.innerHTML = "";
    var W = Math.max(280, host.clientWidth), H = host.clientHeight || 280;
    var m = { t: 18, r: o.r || 14, b: o.xTitle ? 46 : 30, l: o.l || 48 };
    var iw = W - m.l - m.r, ih = H - m.t - m.b, n = o.labels.length;
    var svg = S("svg", { viewBox: "0 0 " + W + " " + H, width: W, height: H, role: "img", "aria-label": o.aria || "" }, host);
    var muted = css("--muted"), line = css("--line");
    var maxV = o.yMax;
    if (maxV == null) {
      var mv = 0;
      (o.bars || []).length && o.labels.forEach(function (_, i) { var t = 0; o.bars.forEach(function (bs) { t += bs.values[i] || 0; }); mv = Math.max(mv, t); });
      (o.lines || []).forEach(function (ls) { ls.values.forEach(function (v) { if (v != null) mv = Math.max(mv, v); }); });
      maxV = o.yLog ? Math.pow(10, Math.ceil(Math.log10(mv))) : niceMax(mv * 1.05);
    }
    var y = o.yLog
      ? function (v) { return m.t + ih - (Math.log10(Math.max(v, 1)) / Math.log10(maxV)) * ih; }
      : function (v) { return m.t + ih - (v / maxV) * ih; };
    var band = iw / n;
    var x = function (i) { return m.l + band * (i + 0.5); };
    // Grid and y ticks
    var ticks = o.yLog ? [1, 10, 100, 1000, 10000, 100000].filter(function (v) { return v <= maxV; }) : [0, 0.25, 0.5, 0.75, 1].map(function (f) { return f * maxV; });
    ticks.forEach(function (v) {
      S("line", { x1: m.l, x2: W - m.r, y1: y(v), y2: y(v), stroke: line, "stroke-width": 1 }, svg);
      S("text", { x: m.l - 8, y: y(v) + 4, "text-anchor": "end", "font-size": 11, fill: muted }, svg, (o.yFmt || fmt.int)(v));
    });
    if (o.yTitle) S("text", { x: 12, y: m.t + ih / 2, transform: "rotate(-90 12 " + (m.t + ih / 2) + ")", "text-anchor": "middle", "font-size": 11, fill: muted }, svg, o.yTitle);
    // X labels
    var step = Math.max(1, Math.ceil(n / Math.max(2, Math.floor(iw / (o.xSpace || 70)))));
    for (var i = 0; i < n; i += step) {
      S("text", { x: x(i), y: H - m.b + 18, "text-anchor": "middle", "font-size": 11, fill: muted }, svg, o.xFmt ? o.xFmt(o.labels[i], i) : o.labels[i]);
    }
    if (o.xTitle) S("text", { x: m.l + iw / 2, y: H - 6, "text-anchor": "middle", "font-size": 11, fill: muted }, svg, o.xTitle);
    // Stacked bars
    if (o.bars) {
      var bw = Math.max(1, band * (o.barFill || 0.72));
      o.labels.forEach(function (lab, i) {
        var base = 0;
        o.bars.forEach(function (bs) {
          var v = bs.values[i] || 0;
          if (!v) return;
          var y0 = y(base), y1 = y(base + v);
          var r = S("rect", { x: x(i) - bw / 2, y: y1, width: bw, height: Math.max(0, y0 - y1), fill: bs.color }, svg);
          S("title", {}, r, (o.tipFmt ? o.tipFmt(lab, bs, v) : lab + " · " + (bs.name ? bs.name + ": " : "") + fmt.int(v)));
          base += v;
        });
      });
    }
    // Reference lines (horizontal)
    (o.hlines || []).forEach(function (h) {
      S("line", { x1: m.l, x2: W - m.r, y1: y(h.value), y2: y(h.value), stroke: h.color, "stroke-width": 1.2, "stroke-dasharray": "5 4" }, svg);
      S("text", { x: W - m.r, y: y(h.value) - 5, "text-anchor": "end", "font-size": 11, "font-weight": 600, fill: h.color }, svg, h.label);
    });
    // Lines
    (o.lines || []).forEach(function (ls) {
      var d = "", pen = false;
      ls.values.forEach(function (v, i) {
        if (v == null) { pen = false; return; }
        d += (pen ? "L" : "M") + x(i).toFixed(1) + " " + y(v).toFixed(1) + " "; pen = true;
      });
      if (ls.area) S("path", { d: d + "L" + x(n - 1) + " " + y(0) + " L" + x(0) + " " + y(0) + " Z", fill: ls.color, "fill-opacity": 0.1, stroke: "none" }, svg);
      S("path", { d: d, fill: "none", stroke: ls.color, "stroke-width": ls.width || 1.5, "stroke-dasharray": ls.dash || "", "stroke-linejoin": "round" }, svg);
      if (ls.points) ls.values.forEach(function (v, i) {
        if (v == null) return;
        var c = S("circle", { cx: x(i), cy: y(v), r: ls.points, fill: ls.color }, svg);
        S("title", {}, c, (o.xFmt ? o.xFmt(o.labels[i], i) : o.labels[i]) + ": " + (o.yFmt || fmt.int)(v));
      });
    });
    // Point markers (anomalies)
    (o.dots || []).forEach(function (dt) {
      var c = S("circle", { cx: x(dt.i), cy: y(dt.v), r: 4, fill: dt.color, stroke: css("--surface"), "stroke-width": 1.5 }, svg);
      S("title", {}, c, dt.tip);
    });
    // Vertical markers
    (o.vlines || []).forEach(function (vl) {
      S("line", { x1: x(vl.i), x2: x(vl.i), y1: m.t, y2: m.t + ih, stroke: vl.color, "stroke-width": 1.5, "stroke-dasharray": "4 4" }, svg);
      var right = vl.align === "right" || (vl.align == null && x(vl.i) > m.l + iw * 0.62);
      S("text", { x: x(vl.i) + (right ? -6 : 6), y: m.t + 12 + (vl.offset || 0), "text-anchor": right ? "end" : "start", "font-size": 11, "font-weight": 600, fill: vl.color }, svg, vl.label);
    });
    S("line", { x1: m.l, x2: W - m.r, y1: m.t + ih, y2: m.t + ih, stroke: css("--faint"), "stroke-width": 1 }, svg);
  }

  // Horizontal bar chart with value labels.
  function hbars(id, rows, valFmt) {
    var host = document.getElementById(id);
    host.innerHTML = "";
    var W = Math.max(280, host.clientWidth), rowH = 26, labW = Math.min(170, W * 0.4), valW = 58;
    var H = rows.length * rowH + 8;
    host.style.height = H + "px";
    var svg = S("svg", { viewBox: "0 0 " + W + " " + H, width: W, height: H, role: "img" }, host);
    var max = Math.max.apply(null, rows.map(function (r) { return r.v; }));
    rows.forEach(function (r, i) {
      var yy = i * rowH + 4, bw = (W - labW - valW - 8) * (r.v / max);
      S("text", { x: labW - 8, y: yy + 16, "text-anchor": "end", "font-size": 12, fill: css("--ink") }, svg, r.label);
      var rc = S("rect", { x: labW, y: yy + 5, width: Math.max(1, bw), height: 14, rx: 2, fill: r.color }, svg);
      S("title", {}, rc, r.label + ": " + valFmt(r.v));
      S("text", { x: labW + bw + 6, y: yy + 16, "font-size": 11.5, fill: css("--muted"), "font-family": css("--font-mono") }, svg, valFmt(r.v));
    });
  }

  function drawCharts() {
    var faint = css("--faint"), accent = css("--accent"), sev1 = css("--sev1");

    var dd = D.risk.daily;
    chart("ch-daily", {
      labels: dd.dates, aria: "Daily incident counts with control limit", xSpace: 90,
      xFmt: function (l) { return monthLabel(l.slice(0, 7)); },
      lines: [{ values: dd.limit, color: css("--line"), width: 1 }, { values: dd.incidents, color: accent, width: 1 }],
      dots: dd.incidents.map(function (v, i) { return dd.anomaly[i] ? { i: i, v: v, color: sev1, tip: fmt.date(dd.dates[i]) + ": " + v + " incidents (limit " + dd.limit[i] + ")" } : null; }).filter(Boolean)
    });

    chart("ch-monthly", {
      labels: D.risk.months_list, xFmt: function (l) { return monthLabel(l); }, xSpace: 60,
      bars: SEVS.map(function (s) { return { name: s, values: D.risk.monthly_by_severity[s], color: sevColor(s) }; }),
      tipFmt: function (l, bs, v) { return monthLabel(l) + " · " + bs.name + ": " + fmt.int(v); }
    });

    var sr = D.risk.service_risk.slice().sort(function (a, b) { return b.total_loss_usd - a.total_loss_usd; });
    hbars("ch-service-loss", sr.map(function (r) { return { label: r.service, v: r.total_loss_usd, color: r.criticality === 3 ? accent : faint }; }), fmt.usdm);

    var hs = D.risk.loss_distribution.histogram, ld = D.risk.loss_distribution;
    var mids = hs.counts.map(function (_, i) { return (hs.edges[i] + hs.edges[i + 1]) / 2; });
    function idx(v) { var best = 0; mids.forEach(function (mm, i) { if (Math.abs(mm - v) < Math.abs(mids[best] - v)) best = i; }); return best; }
    chart("ch-loss", {
      labels: mids, xFmt: function (v) { return fmt.usdm(v); }, xSpace: 64, yLog: true, l: 54, barFill: 0.92,
      xTitle: "Annual loss", yTitle: "Simulated years (log)",
      bars: [{ values: hs.counts, color: faint }],
      tipFmt: function (l, bs, v) { return "Around " + fmt.usdm(l) + " a year: " + fmt.int(v) + " simulated years"; },
      vlines: [{ i: idx(ld.expected_annual_loss), color: accent, label: "Expected " + fmt.usdm(ld.expected_annual_loss) },
               { i: idx(ld.var_99_9), color: sev1, label: "VaR 99.9% " + fmt.usdm(ld.var_99_9), offset: 16 }]
    });

    var vs = D.risk.category_shift.third_party_share_by_month, vm = Object.keys(vs).sort();
    chart("ch-vendor", {
      labels: vm, xFmt: monthLabel, xSpace: 60, yFmt: function (v) { return Math.round(v * 100) + "%"; },
      lines: [{ values: vm.map(function (k) { return vs[k]; }), color: accent, width: 2, points: 2.5, area: true }],
      vlines: [{ i: vm.indexOf("2026-07"), color: sev1, label: "NovaPay live", align: "right" }]
    });

    var ps = D.policy_search;
    var pcap = D.policy.fast_path_max_high_severity_probability;
    chart("ch-policy", {
      labels: ps.map(function (p) { return p.cap; }), xFmt: function (v) { return (v * 100).toFixed(1).replace(".0", "") + "%"; }, xSpace: 56,
      xTitle: "Cap on chance of SEV-1 or SEV-2", yMax: 0.6, yFmt: function (v) { return Math.round(v * 100) + "%"; },
      lines: [{ values: ps.map(function (p) { return p.fast_path_share; }), color: css("--ok"), width: 2, points: 2.5 },
              { values: ps.map(function (p) { return p.high_severity_leakage; }), color: sev1, width: 2, points: 2.5 }],
      hlines: [{ value: 0.03, color: sev1, label: "3% leakage limit" }],
      vlines: [{ i: ps.map(function (p) { return p.cap; }).indexOf(pcap), color: accent, label: "Chosen cap", offset: 0 }]
    });
  }
  function renderThemed() { renderCM(); renderHeat(); drawCharts(); }
  renderThemed();
  var rt;
  window.addEventListener("resize", function () { clearTimeout(rt); rt = setTimeout(drawCharts, 150); });
  var mq = window.matchMedia("(prefers-color-scheme: dark)");
  if (mq.addEventListener) mq.addEventListener("change", renderThemed);
  new MutationObserver(renderThemed).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });

  // Active section in the navigation.
  var links = Array.prototype.slice.call(document.querySelectorAll(".nav a"));
  if ("IntersectionObserver" in window) {
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (en) {
        if (en.isIntersecting) links.forEach(function (a) { a.classList.toggle("active", a.getAttribute("href") === "#" + en.target.id); });
      });
    }, { rootMargin: "-45% 0px -50% 0px" });
    document.querySelectorAll("section.block").forEach(function (s) { io.observe(s); });
  }

  // ---------------------------------------------------------------------
  // In-browser triage model (mirrors src/triage/quality.py and the TF-IDF baseline)
  // ---------------------------------------------------------------------
  if (!M) { document.getElementById("demo-status").textContent = "The demo model did not load."; return; }
  var vocab = {}; M.terms.forEach(function (t, i) { vocab[t] = i; });
  var simVocab = {}; M.sim_terms.forEach(function (t, i) { simVocab[t] = i; });
  var stop = {}; M.stop_words.forEach(function (w) { stop[w] = 1; });
  var ABBR = [[/\bdb\b/g, "database"], [/\bprod\b/g, "production"], [/\bpls\b/g, "please"], [/\bapp\b/g, "application"], [/\basap\b/g, "as soon as possible"], [/\bdc\b/g, "data center"]];

  function normalize(text) {
    var t = String(text).toLowerCase();
    ABBR.forEach(function (a) { t = t.replace(a[0], a[1]); });
    t = t.replace(/\b(chg|inc)\d+\b/g, " changeid ");
    t = t.replace(/\b\d+(\.\d+)*\b/g, " num ");
    t = t.replace(/[^a-z0-9_\-\s']/g, " ");
    return t.replace(/\s+/g, " ").trim();
  }
  function ngrams(tokens) {
    var out = tokens.slice();
    for (var i = 0; i < tokens.length - 1; i++) out.push(tokens[i] + " " + tokens[i + 1]);
    return out;
  }
  function tfidf(grams, map, idf) {
    var counts = {};
    grams.forEach(function (g) { if (map[g] !== undefined) counts[map[g]] = (counts[map[g]] || 0) + 1; });
    var vec = {}, norm = 0;
    Object.keys(counts).forEach(function (k) { var v = (1 + Math.log(counts[k])) * idf[k]; vec[k] = v; norm += v * v; });
    norm = Math.sqrt(norm) || 1;
    Object.keys(vec).forEach(function (k) { vec[k] /= norm; });
    return vec;
  }
  function softmax(z) { var m = Math.max.apply(null, z); var e = z.map(function (v) { return Math.exp(v - m); }); var s = e.reduce(function (a, b) { return a + b; }, 0); return e.map(function (v) { return v / s; }); }
  function score(model, vec) {
    return softmax(model.coef.map(function (row, c) {
      var z = model.intercept[c];
      Object.keys(vec).forEach(function (k) { z += row[k] * vec[k]; });
      return z;
    }));
  }

  function triage(t) {
    var text = normalize(t.short + ". " + t.desc);
    var tokens = text.match(/\b\w\w+\b/g) || [];
    var vec = tfidf(ngrams(tokens), vocab, M.idf);
    var sp = score(M.severity, vec), tp = score(M.team, vec);
    var sev = {}; M.severity.classes.forEach(function (c, i) { sev[c] = sp[i]; });
    var team = M.team.classes.map(function (c, i) { return [c, tp[i]]; }).sort(function (a, b) { return b[1] - a[1]; });
    var top = SEVS.slice().sort(function (a, b) { return sev[b] - sev[a]; })[0];
    var P = M.policy, route;
    if (sev["SEV-1"] >= P.escalate_sev1_probability) route = "escalate_now";
    else if (P.always_review.indexOf(top) >= 0) route = "full_review";
    else if (sev["SEV-1"] + sev["SEV-2"] <= P.fast_path_max_high_severity_probability && team[0][1] >= P.fast_path_team_confidence) route = "fast_path";
    else route = "full_review";

    // Similar incidents with the similarity vectorizer (stop words removed before n-grams).
    var stoks = (text.match(/\b[a-z][a-z0-9_]+\b/g) || []).filter(function (w) { return !stop[w]; });
    var svec = tfidf(ngrams(stoks), simVocab, M.sim_idf);
    var sims = M.history.map(function (h) {
      var s = 0;
      for (var i = 0; i < h.k.length; i++) { var v = svec[h.k[i]]; if (v) s += v * h.w[i]; }
      if (h.s === t.service) s += 0.05;
      return [h, s];
    }).sort(function (a, b) { return b[1] - a[1]; }).slice(0, 3);
    return { sev: sev, top: top, team: team, route: route, sims: sims, empty: Object.keys(vec).length === 0 };
  }

  var PRESETS = [
    { name: "Failed release", service: "Payments Gateway", channel: "Phone", short: "HTTP 500 errors on Payments Gateway after last night's release",
      desc: "Since release 7.2.0 was deployed, health checks are failing and no transactions are going through. Complete outage, all customers are affected." },
    { name: "Expired certificate", service: "Online Banking Portal", channel: "Monitoring Alert", short: "TLS handshake failures on Online Banking Portal",
      desc: "Customers see a certificate warning when logging in. The SSL certificate on web-fe-04 appears to have expired overnight. Many users are affected." },
    { name: "Vendor slowdown", service: "Card Authorization", channel: "Email", short: "NovaPay degraded performance",
      desc: "Latency from NovaPay increased to 12 seconds and card authorizations are timing out. About 15% of transactions are failing." },
    { name: "Disk space alert", service: "Enterprise Data Warehouse", channel: "Monitoring Alert", short: "Filesystem /data at 100% on edw-etl-11",
      desc: "Nightly loads are failing with no space left on device. Internal users only, no customer impact so far." },
    { name: "Password reset", service: "Email and Collaboration", channel: "Service Portal", short: "Password reset request",
      desc: "I am locked out of my account and need a password reset. Not urgent." }
  ];
  var services = Array.from(new Set(M.history.map(function (h) { return h.s; }))).sort();
  var sel = document.getElementById("f-service");
  services.forEach(function (s) { sel.appendChild(el("option", null, esc(s))); });
  var pr = document.getElementById("presets");
  PRESETS.forEach(function (p, i) {
    var b = el("button", { type: "button", "class": "chip" }, esc(p.name));
    b.addEventListener("click", function () { load(i); run(); });
    pr.appendChild(b);
  });
  function load(i) {
    var p = PRESETS[i];
    sel.value = p.service; document.getElementById("f-channel").value = p.channel;
    document.getElementById("f-short").value = p.short; document.getElementById("f-desc").value = p.desc;
  }

  function run() {
    var t = { service: sel.value, channel: document.getElementById("f-channel").value,
      short: document.getElementById("f-short").value.trim(), desc: document.getElementById("f-desc").value.trim() };
    var status = document.getElementById("demo-status");
    if (!t.short && !t.desc) { status.textContent = "Add a short description or a description first."; return; }
    var r = triage(t);
    status.textContent = r.empty ? "Few known words in this ticket, so the suggestion is uncertain." : "";
    var badge = document.getElementById("o-sev");
    badge.textContent = r.top; badge.style.background = sevColor(r.top);
    var route = document.getElementById("o-route");
    route.className = "route " + r.route; route.textContent = routeName[r.route];
    document.getElementById("o-team").textContent = "Owner: " + r.team[0][0];
    bars("o-sevbars", SEVS.map(function (s) { return [s, r.sev[s], sevColor(s)]; }));
    bars("o-teambars", r.team.slice(0, 3).map(function (x, i) { return [x[0], x[1], i ? "var(--faint)" : "var(--accent)"]; }), true);
    var ul = document.getElementById("o-sim"); ul.innerHTML = "";
    r.sims.forEach(function (s) {
      ul.appendChild(el("li", null, "<span class='id'>" + s[0].id + "</span><span class='d' title='" + esc(s[0].d) + "'>" + esc(s[0].d) + "</span><span class='s'>" + s[1].toFixed(2) + "</span>"));
    });
    var reason = "Most likely " + r.top + " (" + fmt.pct0(r.sev[r.top]) + ") for " + t.service + "; owner " + r.team[0][0] + " (" + fmt.pct0(r.team[0][1]) + "). ";
    if (r.route === "escalate_now") reason += "The chance of SEV-1 is " + fmt.pct0(r.sev["SEV-1"]) + ", so a coordinator is paged now.";
    else if (r.route === "fast_path") reason += "A serious outcome is unlikely and the owner is clear, so a coordinator only needs to confirm.";
    else reason += "A coordinator reviews the suggestion before anything changes.";
    document.getElementById("o-rationale").textContent = reason;
    var rec = {
      incident_id: "INC0000000", severity: r.top, severity_confidence: +r.sev[r.top].toFixed(4), team: r.team[0][0], team_confidence: +r.team[0][1].toFixed(4),
      similar_incidents: r.sims.map(function (s) { return { incident_id: s[0].id, similarity: +s[1].toFixed(3), short_description: s[0].d }; }),
      possible_duplicate_of: null, review_route: r.route, rationale: reason, model_version: "tfidf_baseline-web"
    };
    document.getElementById("o-json").textContent = JSON.stringify(rec, null, 2);
  }
  document.getElementById("ticket-form").addEventListener("submit", function (e) { e.preventDefault(); run(); });
  load(0); run();
})();
