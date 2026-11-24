/* Triage Console walkthrough: a simulated coordinator and risk-manager console
   driven by the project's in-browser model (docs/data/web_model.js) and
   test-period examples (docs/data/site_data.js). */
(function () {
  "use strict";
  var D = window.SITE_DATA, M = window.WEB_MODEL;
  var app = document.getElementById("app");
  if (!D || !M) { app.innerHTML = "<p class='gutter empty'>The model data did not load. Open this page from the project site.</p>"; return; }

  var SEVS = ["SEV-1", "SEV-2", "SEV-3", "SEV-4"];
  var SEV_NAME = { "SEV-1": "Critical", "SEV-2": "High", "SEV-3": "Medium", "SEV-4": "Low" };
  var SLA = { "SEV-1": "4 hours", "SEV-2": "8 hours", "SEV-3": "24 hours", "SEV-4": "72 hours" };
  var ROUTE = { escalate_now: "Escalate now", full_review: "Full review", fast_path: "Fast path" };
  var ROUTE_ORDER = { escalate_now: 0, full_review: 1, fast_path: 2 };
  var P = M.policy;

  function esc(s) { return String(s == null ? "" : s).replace(/[&<>"']/g, function (c) { return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]; }); }
  function pct(v) { return Math.round(v * 100) + "%"; }
  function pct1(v) { return (v * 100).toFixed(1) + "%"; }
  function sevVar(s) { return "var(--sev" + s.slice(-1) + ")"; }

  // ---------------------------------------------------------------------
  // Model engine (mirrors src/triage/quality.py and the TF-IDF baseline)
  // ---------------------------------------------------------------------
  var vocab = {}; M.terms.forEach(function (t, i) { vocab[t] = i; });
  var simVocab = {}; M.sim_terms.forEach(function (t, i) { simVocab[t] = i; });
  var stop = {}; M.stop_words.forEach(function (w) { stop[w] = 1; });
  var ABBR = [[/\bdb\b/g, "database"], [/\bprod\b/g, "production"], [/\bpls\b/g, "please"], [/\bapp\b/g, "application"], [/\basap\b/g, "as soon as possible"], [/\bdc\b/g, "data center"]];
  function normalize(text) {
    var t = String(text).toLowerCase();
    ABBR.forEach(function (a) { t = t.replace(a[0], a[1]); });
    t = t.replace(/\b(chg|inc)\d+\b/g, " changeid ").replace(/\b\d+(\.\d+)*\b/g, " num ").replace(/[^a-z0-9_\-\s']/g, " ");
    return t.replace(/\s+/g, " ").trim();
  }
  function ngrams(tok) { var o = tok.slice(); for (var i = 0; i < tok.length - 1; i++) o.push(tok[i] + " " + tok[i + 1]); return o; }
  function tfidf(grams, map, idf) {
    var c = {}, v = {}, n = 0;
    grams.forEach(function (g) { if (map[g] !== undefined) c[map[g]] = (c[map[g]] || 0) + 1; });
    Object.keys(c).forEach(function (k) { var x = (1 + Math.log(c[k])) * idf[k]; v[k] = x; n += x * x; });
    n = Math.sqrt(n) || 1; Object.keys(v).forEach(function (k) { v[k] /= n; });
    return v;
  }
  function softmax(z) { var m = Math.max.apply(null, z), e = z.map(function (x) { return Math.exp(x - m); }), s = e.reduce(function (a, b) { return a + b; }, 0); return e.map(function (x) { return x / s; }); }
  function score(model, v) { return softmax(model.coef.map(function (row, c) { var z = model.intercept[c]; Object.keys(v).forEach(function (k) { z += row[k] * v[k]; }); return z; })); }
  function simVec(text) {
    var tok = (normalize(text).match(/\b[a-z][a-z0-9_]+\b/g) || []).filter(function (w) { return !stop[w]; });
    return tfidf(ngrams(tok), simVocab, M.sim_idf);
  }
  function dot(a, b) { var s = 0; Object.keys(a).forEach(function (k) { if (b[k]) s += a[k] * b[k]; }); return s; }

  function suggest(t) {
    var text = normalize(t.short + ". " + t.desc);
    var v = tfidf(ngrams(text.match(/\b\w\w+\b/g) || []), vocab, M.idf);
    var sp = score(M.severity, v), tp = score(M.team, v);
    var sev = {}; M.severity.classes.forEach(function (c, i) { sev[c] = sp[i]; });
    var teams = M.team.classes.map(function (c, i) { return [c, tp[i]]; }).sort(function (a, b) { return b[1] - a[1]; });
    var top = SEVS.slice().sort(function (a, b) { return sev[b] - sev[a]; })[0];
    var route;
    if (sev["SEV-1"] >= P.escalate_sev1_probability) route = "escalate_now";
    else if (P.always_review.indexOf(top) >= 0) route = "full_review";
    else if (sev["SEV-1"] + sev["SEV-2"] <= P.fast_path_max_high_severity_probability && teams[0][1] >= P.fast_path_team_confidence) route = "fast_path";
    else route = "full_review";
    var sv = simVec(t.short + ". " + t.desc);
    var sims = M.history.map(function (h) {
      var s = 0; for (var i = 0; i < h.k.length; i++) { var x = sv[h.k[i]]; if (x) s += x * h.w[i]; }
      if (h.s === t.service) s += 0.05;
      return { id: h.id, d: h.d, sev: h.v, team: h.t, s: s };
    }).sort(function (a, b) { return b.s - a.s; }).slice(0, 3);
    var reason = "Most likely " + top + " (" + pct(sev[top]) + ") for " + t.service + "; owner " + teams[0][0] + " (" + pct(teams[0][1]) + "). ";
    if (route === "escalate_now") reason += "The chance of SEV-1 is " + pct(sev["SEV-1"]) + ", which is above the " + pct(P.escalate_sev1_probability) + " escalation line.";
    else if (route === "fast_path") reason += "A SEV-1 or SEV-2 outcome is unlikely (" + pct(sev["SEV-1"] + sev["SEV-2"]) + ") and the owner is clear.";
    else if (P.always_review.indexOf(top) >= 0) reason += top + " suggestions always get a full review.";
    else reason += "The model is not confident enough for the fast path.";
    return { severity: top, probs: sev, teams: teams.slice(0, 3), team: teams[0][0], teamConf: teams[0][1], route: route, sims: sims, reason: reason, vec: sv, empty: Object.keys(v).length === 0 };
  }

  // ---------------------------------------------------------------------
  // Queue: real test-period tickets, replayed as one morning shift
  // ---------------------------------------------------------------------
  var TIMES = ["05:48", "06:05", "06:22", "06:40", "07:02", "07:15", "07:31", "07:50", "08:12", "08:27", "08:44", "09:03"];
  var tickets = D.agent.examples.map(function (e, i) {
    return { id: e.incident_id, time: TIMES[i], service: e.ticket.service, channel: e.ticket.channel, short: e.ticket.short_description, desc: e.ticket.description, reporter: reporterFor(e.ticket.channel, i) };
  });
  // A second report of the Customer Contact Center outage, phoned in 22 minutes later.
  var parent = tickets.filter(function (t) { return t.service === "Customer Contact Center"; })[0];
  var DUP = { id: "INC0017491", time: addMin(parent.time, 22), service: parent.service, channel: "Phone",
    short: "Customer Contact Center errors after the configuration change",
    desc: "Health checks for Customer Contact Center are failing after last night's configuration change. Agents cannot load customer records. Many users are affected.",
    reporter: "Floor supervisor, contact center" };
  tickets.push(DUP);
  var LIVE = { id: "INC0017512", time: "09:38", service: "Card Authorization", channel: "Monitoring Alert",
    short: "NovaPay timeouts on card authorizations",
    desc: "Latency from NovaPay increased to 14 seconds. Card Authorization timeouts are rising and about 20% of transactions are failing. Many users are affected.",
    reporter: "Monitoring alert" };

  function reporterFor(ch, i) {
    if (ch === "Monitoring Alert") return "Monitoring alert";
    var names = ["A. Okafor", "M. Lindqvist", "S. Patel", "D. Moreau", "K. Tanaka", "L. Fernandes", "R. Haddad", "E. Novak", "T. Brennan", "Y. Chen", "P. Duarte", "H. Weber"];
    return names[i % names.length] + " · " + ch;
  }
  function addMin(hhmm, m) { var p = hhmm.split(":"), t = +p[0] * 60 + +p[1] + m; return String(Math.floor(t / 60)).padStart(2, "0") + ":" + String(t % 60).padStart(2, "0"); }
  function mins(hhmm) { var p = hhmm.split(":"); return +p[0] * 60 + +p[1]; }

  function enrich(t) {
    t.s = suggest(t);
    // Duplicate check: an earlier open ticket on the same service within the policy window.
    t.dup = null;
    tickets.forEach(function (o) {
      if (o === t || !o.s || o.service !== t.service) return;
      var gap = mins(t.time) - mins(o.time);
      if (gap <= 0 || gap > P.duplicate_window_hours * 60) return;
      var s = dot(t.s.vec, o.s.vec);
      if (s >= P.duplicate_threshold && (!t.dup || s > t.dup.s)) t.dup = { id: o.id, s: s, short: o.short };
    });
  }
  tickets.forEach(enrich);

  // ---------------------------------------------------------------------
  // State and audit trail
  // ---------------------------------------------------------------------
  var S = { mobileDetail: false, view: "coord", filter: "all", sel: tickets[0].id, overrideFor: null, compose: false, killed: false, liveArrived: false };
  var clock = "09:40";
  var audit = [];
  function log(who, text, id) { audit.unshift({ tm: clock, who: who, text: text, id: id }); }
  tickets.forEach(function (t) {
    log("agent", "Suggested " + t.s.severity + " and " + t.s.team + "; routed to " + ROUTE[t.s.route].toLowerCase() + (t.dup ? "; flagged possible duplicate of " + t.dup.id : "") + ".", t.id);
  });

  function byId(id) { return tickets.filter(function (t) { return t.id === id; })[0]; }
  function sorted() {
    return tickets.slice().sort(function (a, b) {
      var da = a.decision ? 1 : 0, db = b.decision ? 1 : 0;
      if (da !== db) return da - db;
      if (!S.killed && ROUTE_ORDER[a.s.route] !== ROUTE_ORDER[b.s.route]) return ROUTE_ORDER[a.s.route] - ROUTE_ORDER[b.s.route];
      return mins(b.time) - mins(a.time);
    });
  }
  function firstWhere(f) { return sorted().filter(f)[0]; }

  // ---------------------------------------------------------------------
  // Rendering
  // ---------------------------------------------------------------------
  function render() {
    document.getElementById("v-coord").setAttribute("aria-pressed", S.view === "coord");
    document.getElementById("v-mgr").setAttribute("aria-pressed", S.view === "mgr");
    document.getElementById("user-name").textContent = S.view === "coord" ? "J. Rivera, triage coordinator" : "N. Adeyemi, operational risk manager";
    document.getElementById("avatar").textContent = S.view === "coord" ? "JR" : "NA";
    var pill = document.getElementById("mode-pill");
    pill.className = "mode-pill " + (S.killed ? "off" : "on");
    pill.textContent = S.killed ? "Manual triage mode" : "Suggestions on";
    var banner = document.getElementById("banner");
    banner.className = "banner" + (S.killed ? " alert" : "");
    banner.querySelector("span").innerHTML = S.killed
      ? "<b>Suggestions are switched off.</b> The kill switch is on, so the agent's suggestions are hidden and every ticket is triaged by hand. Nothing else changed."
      : "<b>Product walkthrough.</b> A simulated console built on the project's model and real test-period tickets. Ticket text is shown as reporters wrote it, typos included. Names are fictional and nothing here changes a real system.";
    app.innerHTML = S.view === "coord" ? coordHTML() : mgrHTML();
    bind();
  }

  function sevBadge(s, ghost) { return ghost ? "<span class='sev ghost'>Unset</span>" : "<span class='sev' style='background:" + sevVar(s) + "'>" + s + "</span>"; }
  function routePill(r) { return "<span class='route " + r + "'>" + ROUTE[r] + "</span>"; }

  function coordHTML() {
    var list = sorted();
    var counts = { all: 0, escalate_now: 0, full_review: 0, fast_path: 0 };
    list.forEach(function (t) { if (!t.decision) { counts.all++; counts[t.s.route]++; } });
    var shown = list.filter(function (t) { return S.filter === "all" || S.killed || t.s.route === S.filter; });
    var tabs = S.killed ? "" : ["all", "escalate_now", "full_review", "fast_path"].map(function (f) {
      return "<button type='button' class='tab' data-filter='" + f + "' aria-pressed='" + (S.filter === f) + "'>" + (f === "all" ? "All open" : ROUTE[f]) + " <span class='n'>" + counts[f] + "</span></button>";
    }).join("");
    var items = shown.map(function (t) {
      var d = t.decision, stripe = S.killed ? "var(--line)" : (d ? "var(--line)" : sevVar(t.s.severity));
      var right = d ? "<span class='status done'>" + (d.type === "duplicate" ? "Linked" : "Triaged") + "</span>" + sevBadge(d.severity)
        : (S.killed ? "<span class='status'>Needs triage</span>" : sevBadge(t.s.severity) + routePill(t.s.route));
      return "<li><button type='button' class='qitem" + (d ? " done" : "") + (t.isNew ? " new" : "") + "' data-id='" + t.id + "' aria-current='" + (S.sel === t.id) + "'>" +
        "<span class='stripe' style='background:" + stripe + "'></span>" +
        "<span><span class='l1'><span class='mono'>" + t.id + "</span><span>" + t.time + "</span></span>" +
        "<span class='t'>" + esc(t.short) + "</span>" +
        "<span class='l3'><span class='chip'>" + esc(t.service) + "</span>" + (!S.killed && t.dup && !d ? "<span class='chip dup'>Possible duplicate</span>" : "") + "</span></span>" +
        "<span class='right'>" + right + "</span></button></li>";
    }).join("");
    return "<div class='workspace" + (S.mobileDetail || S.compose ? " show-detail" : "") + "'>" +
      "<section class='queue' aria-label='Ticket queue' id='queue'><div class='queue-head gutter'><h2>Incoming queue <span class='small'>Shift started 05:30 · " + counts.all + " open</span></h2>" +
      "<div class='tabs' id='tabs'>" + tabs + "</div>" +
      "<div><button type='button' class='btn' id='new-ticket' style='padding:6px 10px;font-size:13px'>Log a new ticket</button></div></div>" +
      "<ul class='qlist'>" + (items || "<li class='empty'>Nothing in this view.</li>") + "</ul></section>" +
      "<section class='detail gutter' aria-label='Ticket detail' id='detail'><button type='button' class='btn back-queue' id='back-queue'>Back to queue</button>" + (S.compose ? composeHTML() : detailHTML(byId(S.sel))) + "</section></div>";
  }

  function composeHTML() {
    var opts = Array.from(new Set(M.history.map(function (h) { return h.s; }))).sort().map(function (s) { return "<option>" + esc(s) + "</option>"; }).join("");
    return "<div class='card compose' id='compose'><div class='card-h'><h3>Log a new ticket</h3><span class='ai-tag'>Triaged as soon as it is logged</span></div>" +
      "<div class='override row' style='border:0;padding:0'><div class='row'><div class='field'><label for='c-service'>Service</label><select id='c-service'>" + opts + "</select></div>" +
      "<div class='field'><label for='c-channel'>Reported through</label><select id='c-channel'><option>Phone</option><option>Email</option><option>Service Portal</option><option>Monitoring Alert</option></select></div></div></div>" +
      "<div class='field'><label for='c-short'>Short description</label><input type='text' id='c-short' maxlength='200' placeholder='For example: Mobile app login failing for customers'></div>" +
      "<div class='field'><label for='c-desc'>Description</label><textarea id='c-desc' maxlength='2000' placeholder='What is happening, who is affected and since when'></textarea></div>" +
      "<div class='actions'><button type='button' class='btn primary' id='c-submit'>Log ticket</button><button type='button' class='btn' id='c-cancel'>Cancel</button><span class='hint' id='c-msg'></span></div></div>";
  }

  function bars(rows, short) {
    return "<div class='bars'>" + rows.map(function (r) {
      return "<div class='bar" + (short ? " short" : "") + "'><span class='k' title='" + esc(r[0]) + "'>" + esc(r[0]) + "</span><span class='tr'><span class='f' style='width:" + Math.max(0.5, r[1] * 100).toFixed(1) + "%;background:" + r[2] + "'></span></span><span class='v'>" + pct(r[1]) + "</span></div>";
    }).join("") + "</div>";
  }

  function detailHTML(t) {
    if (!t) return "<p class='empty'>Select a ticket.</p>";
    var s = t.s, d = t.decision;
    var head = "<div class='d-head'><div class='d-meta'><span class='mono'><b>" + t.id + "</b></span><span>Opened <b>" + t.time + "</b></span><span>Service <b>" + esc(t.service) + "</b></span><span>Channel <b>" + esc(t.channel) + "</b></span></div>" +
      "<h1 class='d-title'>" + esc(t.short) + "</h1></div>" +
      "<div class='reporter'><div class='who'>" + esc(t.reporter) + "</div>" + esc(t.desc) + "</div>";

    var dupBox = (!S.killed && t.dup && !d) ? "<div class='dupbox' id='dupbox'><span>This looks like a second report of <b>" + t.dup.id + "</b>, opened on the same service " + (mins(t.time) - mins(byId(t.dup.id).time)) + " minutes earlier (similarity " + t.dup.s.toFixed(2) + ").</span>" +
      "<span class='actions'><button type='button' class='btn' id='link-dup'>Link as duplicate</button><button type='button' class='btn' id='open-parent'>Open " + t.dup.id + "</button></span></div>" : "";

    var sugg = S.killed ? "<div class='card'><div class='card-h'><h3>Agent suggestion</h3><span class='ai-tag'>Hidden</span></div><p class='why'>Suggestions are switched off by the risk manager. Set the severity and owner below by hand.</p></div>" :
      "<div class='card' id='suggestion'><div class='card-h'><h3>Agent suggestion</h3><span class='ai-tag'>Model " + esc(D.validation.champion.replace("_", " ")) + " · " + esc(P.version) + "</span></div>" +
      "<div class='verdict' id='verdict'>" + sevBadge(s.severity) + "<span><b>" + SEV_NAME[s.severity] + "</b> <span class='small'>· resolve within " + SLA[s.severity] + "</span></span>" + routePill(s.route) + "<span class='small'>Owner: <b style='color:var(--ink)'>" + esc(s.team) + "</b></span></div>" +
      "<p class='why' id='why'>" + esc(s.reason) + (s.empty ? " Few words in this ticket are known to the model, so treat it with care." : "") + "</p>" +
      "<div class='cols'><div id='probs'><div class='lbl'>Severity probabilities</div>" + bars(SEVS.map(function (k) { return [k, s.probs[k], sevVar(k)]; }), true) + "</div>" +
      "<div><div class='lbl'>Owner</div>" + bars(s.teams.map(function (x, i) { return [x[0], x[1], i ? "var(--faint)" : "var(--accent)"]; })) + "</div></div>" +
      "<div id='similar'><div class='lbl'>Similar past incidents</div><ul class='sims'>" + s.sims.map(function (x) {
        return "<li><span class='id'>" + x.id + "</span><span class='d' title='" + esc(x.d) + "'>" + esc(x.d) + "</span><span class='s'>" + x.sev + " · " + x.s.toFixed(2) + "</span></li>";
      }).join("") + "</ul></div></div>";

    var act;
    if (d) {
      act = "<div class='card' id='decided'><div class='card-h'><h3>Your decision</h3><span class='ai-tag'>Recorded " + d.time + "</span></div><div class='decision" + (d.type === "override" ? " ovr" : "") + "'>" + esc(d.text) + "</div>" +
        "<div class='actions'><button type='button' class='btn' id='undo'>Reopen decision</button><button type='button' class='btn danger' id='agent-close'>Ask the agent to close this ticket</button></div><p class='hint'>The second button shows what happens when anyone asks the agent to make a change.</p></div>";
    } else {
      var teamOpts = D.agent.tools ? M.team.classes.map(function (c) { return "<option" + (c === s.team ? " selected" : "") + ">" + esc(c) + "</option>"; }).join("") : "";
      var sevOpts = SEVS.map(function (k) { return "<option value='" + k + "'" + (k === (t.ovrSev || s.severity) ? " selected" : "") + ">" + k + " · " + SEV_NAME[k] + "</option>"; }).join("");
      var primary = S.killed ? "Save triage" : (s.route === "escalate_now" ? "Confirm and page " + shortTeam(s.team) : (s.route === "fast_path" ? "Confirm" : "Accept suggestion"));
      var showForm = S.killed || S.overrideFor === t.id;
      act = "<div class='card' id='actions'><div class='card-h'><h3>Your decision</h3><span class='ai-tag'>Only people can change the ticket</span></div>" +
        (showForm ? "" : "<div class='actions'><button type='button' class='btn primary' id='accept'>" + esc(primary) + "</button><button type='button' class='btn' id='override'>Override</button></div>" +
          "<p class='hint'>" + (s.route === "fast_path" ? "Fast path: one click to confirm. You can still override." : s.route === "escalate_now" ? "Escalations page the owning team as soon as you confirm." : "Read the evidence above, then accept or override.") + "</p>") +
        (showForm ? "<div class='override' id='override-form' style='" + (S.killed ? "border:0;padding:0" : "") + "'><div class='row'><div class='field'><label for='o-sev'>Severity</label><select id='o-sev'>" + sevOpts + "</select></div>" +
          "<div class='field'><label for='o-team'>Owner</label><select id='o-team'>" + teamOpts + "</select></div></div>" +
          (S.killed ? "" : "<div class='field'><label for='o-reason'>Reason for the override (required)</label><textarea id='o-reason' placeholder='What did you see that the agent missed?'>" + esc(t.ovrReason || "") + "</textarea></div>") +
          "<div class='actions'><button type='button' class='btn primary' id='save-override'>" + (S.killed ? "Save triage" : "Save override") + "</button>" + (S.killed ? "" : "<button type='button' class='btn' id='cancel-override'>Cancel</button>") + "<span class='hint' id='o-msg'></span></div></div>" : "") +
        "</div>";
    }
    var trail = audit.filter(function (a) { return a.id === t.id; });
    var auditHTML = "<div class='card' id='audit'><div class='card-h'><h3>Audit trail</h3><span class='ai-tag'>Every action is logged</span></div><ul class='audit'>" + trail.map(function (a) {
      return "<li><span class='tm'>" + a.tm + "</span><span class='who " + a.who + "'>" + (a.who === "denied" ? "Refused" : a.who === "human" ? "J. Rivera" : "Agent") + "</span><span>" + esc(a.text) + "</span></li>";
    }).join("") + "</ul></div>";
    return head + dupBox + sugg + act + auditHTML;
  }
  function shortTeam(t) { return t.replace(" Support", "").replace(" Administration", "").replace(" Operations", " Ops").replace(" Engineering", ""); }

  function mgrHTML() {
    var mon = D.monitoring.months, last = mon[mon.length - 1];
    var decided = tickets.filter(function (t) { return t.decision; });
    var ovr = decided.filter(function (t) { return t.decision.type === "override"; }).length;
    var vs = D.risk.category_shift.third_party_share_by_month, keys = Object.keys(vs).sort();
    var top = D.clusters.slice().sort(function (a, b) { return b.trend_ratio_last6_vs_first6 - a.trend_ratio_last6_vs_first6; })[0];
    var max = Math.max.apply(null, keys.map(function (k) { return vs[k]; }));
    var pts = keys.map(function (k, i) { return (i / (keys.length - 1) * 300).toFixed(1) + "," + (52 - vs[k] / max * 46).toFixed(1); }).join(" ");
    var shift = keys.indexOf("2026-07") / (keys.length - 1) * 300;
    var rows = mon.map(function (m) {
      return "<tr><td>" + monthName(m.month) + "</td><td class='r'>" + m.tickets + "</td><td class='r'>" + pct1(m.override_rate) + "</td><td class='r'>" + m.psi_category_mix.toFixed(3) + "</td><td class='r'>" + pct(m.fast_path_share) + "</td><td><span class='st " + m.status + "'>" + { ok: "OK", watch: "Watch", alert: "Alert" }[m.status] + "</span></td><td class='small'>" + esc(m.flags.join("; ") || "No flags") + "</td></tr>";
    }).join("");
    var hum = audit.filter(function (a) { return a.who !== "agent"; });
    return "<div class='mgr gutter'>" +
      "<div class='kpis' id='m-kpis'>" +
      kpi(pct1(last.override_rate), "Severity override rate, " + monthName(last.month)) +
      kpi(pct(D.routing.share.fast_path), "Tickets on the fast path, test period") +
      kpi(pct1(D.routing.fast_path_high_severity_leakage), "Fast-path tickets that were SEV-1 or SEV-2") +
      kpi(pct1(D.routing.sev1_caught_by_escalation_or_review), "SEV-1 incidents kept off the fast path") +
      kpi(decided.length + " of " + tickets.length, "Tickets triaged this shift, " + ovr + " overridden") + "</div>" +
      "<div class='card' id='m-alert'><div class='card-h'><h3>Recurring issue alert</h3><span class='st alert'>Rising</span></div>" +
      "<p class='why'><b style='color:var(--ink)'>" + esc(top.label) + "</b> on " + esc(top.top_service) + " are running at " + top.trend_ratio_last6_vs_first6.toFixed(1) + " times their early volume, and " + pct(top.repeat_within_7_days) + " repeat within a week. The rise starts when the NovaPay integration went live in July. Suggested action: raise it with Third-Party Vendor Management and add NovaPay to the vendor scorecard review.</p>" +
      "<div><div class='lbl'>Third-party share of incidents by month</div><svg class='spark' viewBox='0 0 300 56' preserveAspectRatio='none' role='img' aria-label='Third-party share rising after July 2026'><line x1='" + shift.toFixed(1) + "' x2='" + shift.toFixed(1) + "' y1='0' y2='56' stroke='var(--sev1)' stroke-dasharray='3 3' vector-effect='non-scaling-stroke'/><polyline points='" + pts + "' fill='none' stroke='var(--accent)' stroke-width='2' vector-effect='non-scaling-stroke'/></svg>" +
      "<div class='small' style='display:flex;justify-content:space-between'><span>" + monthName(keys[0]) + "</span><span style='color:var(--sev1)'>NovaPay live, Jul 2026</span><span>" + monthName(keys[keys.length - 1]) + "</span></div></div></div>" +
      "<div class='card' id='m-monitor'><div class='card-h'><h3>Monthly model monitoring</h3><span class='ai-tag'>Watch above " + pct1(D.monitoring.thresholds.override_watch) + " overrides or PSI 0.10</span></div><div class='table-wrap'><table><thead><tr><th>Month</th><th class='r'>Tickets</th><th class='r'>Overrides</th><th class='r'>PSI, mix</th><th class='r'>Fast path</th><th>Status</th><th>Flags</th></tr></thead><tbody>" + rows + "</tbody></table></div></div>" +
      "<div class='card' id='m-kill'><div class='card-h'><h3>Kill switch</h3><span class='ai-tag'>models/registry.json</span></div><div class='switch'><button type='button' class='toggle' id='kill' role='switch' aria-checked='" + S.killed + "' aria-label='Turn off agent suggestions'></button>" +
      "<span><b>" + (S.killed ? "Suggestions are off." : "Suggestions are on.") + "</b> <span class='small'>" + (S.killed ? "Coordinators triage by hand. Turn it back on when the issue is fixed." : "Turning this on hides every suggestion at once and returns the desk to manual triage. The agent never wrote to tickets, so nothing needs repairing.") + "</span></span></div></div>" +
      "<div class='card' id='m-log'><div class='card-h'><h3>Human decisions this shift</h3><span class='ai-tag'>From the audit trail</span></div>" + (hum.length ? "<ul class='audit'>" + hum.map(function (a) {
        return "<li><span class='tm'>" + a.tm + "</span><span class='who " + a.who + "'>" + (a.who === "denied" ? "Refused" : "Coordinator") + "</span><span><span class='mono small'>" + a.id + "</span> " + esc(a.text) + "</span></li>";
      }).join("") + "</ul>" : "<p class='why'>No decisions yet. Triage a few tickets in the coordinator view and they will appear here.</p>") + "</div></div>";
  }
  function monthName(ym) { var p = ym.split("-"); return ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][+p[1] - 1] + " " + p[0]; }
  function kpi(v, l) { return "<div class='kpi'><span class='v'>" + v + "</span><span class='l'>" + esc(l) + "</span></div>"; }

  // ---------------------------------------------------------------------
  // Actions
  // ---------------------------------------------------------------------
  function tick() { clock = addMin(clock, 1); }
  function toast(msg, warn) {
    var box = document.getElementById("toasts");
    while (box.children.length > 1) box.removeChild(box.firstChild);
    var t = document.createElement("div"); t.className = "toast" + (warn ? " warn" : ""); t.textContent = msg;
    box.appendChild(t); setTimeout(function () { t.remove(); }, 3200);
  }
  function accept(t) {
    tick();
    var s = t.s, text = "Confirmed " + s.severity + " and assigned " + s.team + (s.route === "escalate_now" ? "; paged the on-call engineer." : ".");
    t.decision = { type: "confirm", severity: s.severity, team: s.team, time: clock, text: text };
    log("human", text, t.id);
    toast(t.id + ": " + text);
  }
  function override(t, sev, team, reason) {
    tick();
    var text = "Overrode the suggestion: set " + sev + " (agent said " + t.s.severity + ")" + (team !== t.s.team ? " and owner " + team : "") + ". Reason: " + reason;
    t.decision = { type: "override", severity: sev, team: team, time: clock, text: text };
    log("human", text, t.id);
    toast(t.id + ": override saved and sent to the monitoring report.");
  }
  function manual(t, sev, team) {
    tick();
    var text = "Triaged by hand: " + sev + ", " + team + ".";
    t.decision = { type: "manual", severity: sev, team: team, time: clock, text: text };
    log("human", text, t.id);
    toast(t.id + ": saved.");
  }
  function linkDup(t) {
    tick();
    var p = byId(t.dup.id);
    var sev = p.decision ? p.decision.severity : p.s.severity;
    var text = "Linked as a duplicate of " + p.id + "; it follows that incident's severity (" + sev + ") and owner, and the caller gets its updates.";
    t.decision = { type: "duplicate", severity: sev, team: p.decision ? p.decision.team : p.s.team, time: clock, text: text };
    log("human", text, t.id);
    toast(t.id + " linked to " + p.id + ".");
  }
  function agentClose(t) {
    tick();
    log("agent", "Asked to close the ticket.", t.id);
    log("denied", "close_ticket refused: the agent has read and suggest permission only. A person must close it.", t.id);
    toast("The agent refused: it is not allowed to close tickets.", true);
  }
  function arriveLive() {
    if (S.liveArrived) return;
    S.liveArrived = true; clock = "09:41";
    var t = Object.assign({}, LIVE, { isNew: true }); tickets.push(t); enrich(t);
    log("agent", "Suggested " + t.s.severity + " and " + t.s.team + "; routed to " + ROUTE[t.s.route].toLowerCase() + ".", t.id);
    toast("New ticket " + t.id + " arrived and was triaged in under a second.");
  }

  function bind() {
    function on(id, ev, fn) { var e = document.getElementById(id); if (e) e.addEventListener(ev, fn); }
    on("tabs", "click", function (e) { var b = e.target.closest("[data-filter]"); if (b) { S.filter = b.getAttribute("data-filter"); render(); } });
    app.querySelectorAll(".qitem").forEach(function (b) { b.addEventListener("click", function () { S.sel = b.getAttribute("data-id"); S.overrideFor = null; S.compose = false; S.mobileDetail = true; render(); window.scrollTo(0, 0); }); });
    var t = byId(S.sel);
    on("back-queue", "click", function () { S.mobileDetail = false; S.compose = false; render(); window.scrollTo(0, 0); });
    on("accept", "click", function () { accept(t); render(); });
    on("override", "click", function () { S.overrideFor = t.id; render(); var r = document.getElementById("o-reason"); if (r) r.focus(); });
    on("cancel-override", "click", function () { S.overrideFor = null; render(); });
    on("save-override", "click", function () {
      var sev = document.getElementById("o-sev").value, team = document.getElementById("o-team").value;
      if (S.killed) { manual(t, sev, team); render(); return; }
      var reason = document.getElementById("o-reason").value.trim();
      if (reason.length < 5) { document.getElementById("o-msg").textContent = "Add a short reason so reviewers can learn from the override."; return; }
      if (sev === t.s.severity && team === t.s.team) { document.getElementById("o-msg").textContent = "This matches the suggestion. Use Accept instead, or change a value."; return; }
      S.overrideFor = null; override(t, sev, team, reason); render();
    });
    on("link-dup", "click", function () { linkDup(t); render(); });
    on("open-parent", "click", function () { S.sel = t.dup.id; render(); });
    on("undo", "click", function () { tick(); log("human", "Reopened the decision.", t.id); t.decision = null; render(); });
    on("agent-close", "click", function () { agentClose(t); render(); });
    on("new-ticket", "click", function () { S.compose = true; render(); document.getElementById("c-short").focus(); });
    on("c-cancel", "click", function () { S.compose = false; render(); });
    on("c-submit", "click", function () {
      var sh = document.getElementById("c-short").value.trim(), de = document.getElementById("c-desc").value.trim();
      if (!sh || !de) { document.getElementById("c-msg").textContent = "Add a short description and a description."; return; }
      tick();
      var nt = { id: "INC00175" + String(20 + tickets.length).padStart(2, "0"), time: clock, service: document.getElementById("c-service").value, channel: document.getElementById("c-channel").value, short: sh, desc: de, reporter: "J. Rivera · logged on behalf of caller", isNew: true };
      tickets.push(nt); enrich(nt);
      log("agent", "Suggested " + nt.s.severity + " and " + nt.s.team + "; routed to " + ROUTE[nt.s.route].toLowerCase() + (nt.dup ? "; flagged possible duplicate of " + nt.dup.id : "") + ".", nt.id);
      S.compose = false; S.sel = nt.id; S.filter = "all"; S.mobileDetail = true; render();
    });
    on("kill", "click", function () {
      S.killed = !S.killed; tick();
      log("human", S.killed ? "Kill switch turned on: suggestions hidden, manual triage." : "Kill switch turned off: suggestions restored.", "registry");
      toast(S.killed ? "Suggestions are off. The desk is in manual triage mode." : "Suggestions are back on.", S.killed);
      render();
    });
  }

  document.getElementById("v-coord").addEventListener("click", function () { S.view = "coord"; render(); });
  document.getElementById("v-mgr").addEventListener("click", function () { S.view = "mgr"; render(); });

  // ---------------------------------------------------------------------
  // Guided tour
  // ---------------------------------------------------------------------
  var esc1 = firstWhere(function (t) { return t.s.route === "escalate_now" && !t.dup; });
  var fast = firstWhere(function (t) { return t.s.route === "fast_path"; });
  var ovr = byId("INC0012957") || firstWhere(function (t) { return t.s.route === "full_review"; });
  var vague = firstWhere(function (t) { return t.s.empty || /^not working/i.test(t.desc); });

  var STEPS = [
    { title: "A coordinator starts the morning shift", body: "Twelve tickets came in overnight, and the agent has already suggested a severity, an owner and a review route for each one. Escalations sit at the top, then full reviews, then the fast path. The colored stripe shows the suggested severity.",
      go: function () { S.view = "coord"; S.filter = "all"; S.sel = sorted()[0].id; }, hl: "#queue" },
    { title: "Escalations come first", body: "This ticket's chance of being SEV-1 is above 30%, so the agent routes it to Escalate now, whatever its top guess is. The probabilities show how sure the model is.",
      go: function () { S.sel = esc1.id; }, hl: "#verdict" },
    { title: "Check the evidence before acting", body: "The agent explains its suggestion in plain words and shows the closest past incidents, so the coordinator can see what it is matching against.",
      go: function () { S.sel = esc1.id; }, hl: "#similar" },
    { title: "Confirm and page the owner", body: "One click records the decision and pages the owning team. The decision is written to the ticket's audit trail under the coordinator's name, not the agent's.",
      go: function () { S.sel = esc1.id; if (!esc1.decision) { accept(esc1); } }, hl: "#decided" },
    { title: "Fast path: confirm in one click", body: "Here a SEV-1 or SEV-2 outcome is unlikely and the owner is clear, so the ticket takes the fast path. The coordinator still confirms it, which takes seconds instead of minutes.",
      go: function () { S.sel = fast.id; }, hl: "#actions" },
    { title: "Override when you disagree", body: "The agent suggested " + ovr.s.severity + " for this ledger outage, but the coordinator knows the service is critical and sets SEV-2. An override needs a reason, and every override feeds the monthly monitoring report.",
      go: function () { S.sel = ovr.id; S.overrideFor = ovr.id; ovr.ovrSev = "SEV-2"; ovr.ovrReason = "Core Banking Ledger is a critical service and the pods are crash-looping, so customer impact is likely."; }, hl: "#override-form" },
    { title: "Duplicates get linked, not reworked", body: "A floor supervisor phoned in the same contact-center outage 22 minutes later. The agent spotted the match, so the coordinator links it instead of triaging it twice.",
      go: function () { S.overrideFor = null; S.sel = DUP.id; }, hl: "#dupbox" },
    { title: "Uncertain tickets go to a person", body: "This reporter only wrote \"Not working.\", so the model leans on the short title, and it is not sure who owns the ticket. Low confidence sends it to full review, where a coordinator can call the user back instead of trusting a guess.",
      go: function () { S.sel = vague.id; }, hl: "#suggestion" },
    { title: "Only people change tickets", body: "Ask the agent to close a ticket and it refuses: it can read and suggest, but closing, reassigning and changing severity belong to people. The refusal is logged too.",
      go: function () { S.sel = esc1.id; var t = byId(esc1.id); if (!audit.some(function (a) { return a.id === t.id && a.who === "denied"; })) agentClose(t); }, hl: "#audit" },
    { title: "A new ticket arrives", body: "A monitoring alert about NovaPay timeouts lands at 09:41. The model triages it in the browser the moment it arrives. Try it yourself with \"Log a new ticket\".",
      go: function () { arriveLive(); S.sel = LIVE.id; S.filter = "all"; }, hl: "#suggestion" },
    { title: "The risk manager's view", body: "The operational risk manager watches the model, not individual tickets: override rates, drift and the safety of the fast path. A recurring-issue alert shows vendor API outages rising since NovaPay went live.",
      go: function () { S.view = "mgr"; }, hl: "#m-alert" },
    { title: "Monitoring and the kill switch", body: "Each month gets a status from override rates and population stability. If the model misbehaves, one switch hides every suggestion and returns the desk to manual triage. Try it, then switch back to the coordinator view.",
      go: function () { S.view = "mgr"; }, hl: "#m-kill" },
    { title: "Explore on your own", body: "You have seen the whole flow. Click any ticket, accept or override suggestions, log your own tickets, or flip the kill switch. Your decisions show up in the risk manager's log.",
      go: function () { S.view = "coord"; }, hl: null }
  ];
  var cur = 0;
  var tour = document.getElementById("tour");
  var dots = document.getElementById("t-dots");
  dots.innerHTML = STEPS.map(function () { return "<i></i>"; }).join("");
  function show(i) {
    cur = Math.max(0, Math.min(STEPS.length - 1, i));
    var st = STEPS[cur];
    if (S.killed && cur < 11) { S.killed = false; }
    S.compose = false;
    S.mobileDetail = cur > 0;
    st.go(); render();
    document.getElementById("t-step").textContent = "Step " + (cur + 1) + " of " + STEPS.length;
    document.getElementById("t-title").textContent = st.title;
    document.getElementById("t-body").textContent = st.body;
    document.getElementById("t-back").disabled = cur === 0;
    document.getElementById("t-next").textContent = cur === STEPS.length - 1 ? "Finish" : "Next";
    Array.prototype.forEach.call(dots.children, function (d, k) { d.className = k === cur ? "on" : ""; });
    document.querySelectorAll(".hl").forEach(function (e) { e.classList.remove("hl"); });
    if (st.hl) {
      var el = document.querySelector(st.hl);
      if (el) {
        el.classList.add("hl");
        var r = el.getBoundingClientRect();
        if (window.innerWidth < 980) { window.scrollTo(0, Math.max(0, window.scrollY + r.top - 130)); }
        else if (r.top < 70 || r.bottom > window.innerHeight - 40) el.scrollIntoView({ behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth", block: "center" });
      }
      var item = document.querySelector(".qitem[aria-current='true']");
      if (item && window.innerWidth >= 980) item.scrollIntoView({ block: "nearest" });
    }
  }
  document.getElementById("t-next").addEventListener("click", function () { if (cur === STEPS.length - 1) { tour.classList.add("min"); document.querySelectorAll(".hl").forEach(function (e) { e.classList.remove("hl"); }); } else show(cur + 1); });
  document.getElementById("t-back").addEventListener("click", function () { show(cur - 1); });
  document.getElementById("t-min").addEventListener("click", function () { tour.classList.add("min"); document.querySelectorAll(".hl").forEach(function (e) { e.classList.remove("hl"); }); });
  document.getElementById("t-show").addEventListener("click", function () { tour.classList.remove("min"); show(cur); });
  document.addEventListener("keydown", function (e) {
    if (tour.classList.contains("min") || /INPUT|TEXTAREA|SELECT/.test(document.activeElement.tagName)) return;
    if (e.key === "ArrowRight") document.getElementById("t-next").click();
    if (e.key === "ArrowLeft" && cur > 0) show(cur - 1);
  });

  show(0);
})();
