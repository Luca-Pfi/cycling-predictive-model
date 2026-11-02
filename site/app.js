// Breakaway — static site. Routes: #calendar, #race-<id>[-s<stage>], #accuracy, #method.
const view = document.getElementById("view");
const cache = {};
const get = url => cache[url] ||= fetch(url).then(r => { if (!r.ok) throw new Error(url); return r.json(); });
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const pct = (p, d) => p == null ? "–" : p === 0 ? "0%" : (100 * p).toFixed(d ?? (p < 0.01 ? 1 : 0)) + "%";
const odds = p => { if (p == null) return "–"; const o = 1 / Math.max(p, 1e-4); return o >= 100 ? Math.round(o) : o.toFixed(2); };
const MON = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
const MONTH = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"];
const DAY = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];
const d = s => new Date(s + "T12:00:00");
const day = s => `${d(s).getDate()} ${MON[d(s).getMonth()]}`;
const longDay = s => `${DAY[d(s).getDay()]} ${day(s)}`;
const span = (a, b) => a === b ? day(a) : d(a).getMonth() === d(b).getMonth() ? `${d(a).getDate()}–${day(b)}` : `${day(a)} – ${day(b)}`;
const mmss = s => `${Math.floor(s / 60)}:${String(Math.round(s % 60)).padStart(2, "0")}`;
const ICON = { p1: "flat", p2: "hills, flat finish", p3: "hills, uphill finish", p4: "mountains, flat finish", p5: "summit finish" };
const SHAPE = { p1: "M1 15 L63 15", p2: "M1 15 C9 11 13 7 20 12 S33 6 41 11 S55 13 63 15",
  p3: "M1 15 C9 11 13 7 20 12 S35 10 44 11 S57 6 63 3", p4: "M1 16 L11 7 L18 13 L30 2 L40 12 L50 6 L63 16",
  p5: "M1 16 L11 8 L18 13 L29 5 L38 12 L50 8 L63 2" };
const profile = icon => `<svg class="profile" viewBox="0 0 64 18" aria-hidden="true"><path d="${SHAPE[icon] || SHAPE.p2}"/>${icon === "p3" || icon === "p5" ? '<circle cx="62" cy="3" r="2.2"/>' : ""}</svg>`;
let INDEX = null;

// ------------------------------------------------------------ live model (same as cyclepredict/live.py)
function phi(z) { const t = 1 / (1 + 0.3275911 * Math.abs(z) / Math.SQRT2);
  const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-z * z / 2);
  return z >= 0 ? (1 + y) / 2 : (1 - y) / 2; }
function closing(L, s) { let c = L.chase[s.chase] ?? 6.5; c *= 1 - L.size_effect * (Math.max(s.n, 1) - 1);
  if (s.km <= 10) c *= 1.25; if (s.hard) c *= 1.2; return Math.max(c, 0.5); }
function pPhys(L, s) { if (s.km <= 0) return s.gap > 0 ? 1 : 0;
  return phi((s.gap - closing(L, s) * s.km - L.margin) / (L.sd * Math.pow(s.km, 0.75) + 5)); }
const logit = p => { p = Math.min(Math.max(p, 1e-4), 1 - 1e-4); return Math.log(p / (1 - p)); };
function pSurvive(L, s, pre, dist) { const w = Math.min(Math.max(s.km / dist, 0), 1);
  return 1 / (1 + Math.exp(-((1 - w) * logit(pPhys(L, s)) + w * logit(pre)))); }
function gapNeeded(L, s, km) { let lo = 0, hi = 1800;
  for (let i = 0; i < 40; i++) { const m = (lo + hi) / 2; if (pPhys(L, { ...s, km, gap: m }) >= 0.5) hi = m; else lo = m; } return hi; }

// ------------------------------------------------------------ routing
async function route() {
  const h = location.hash.slice(1) || "calendar";
  INDEX ||= await get("data/index.json");
  document.getElementById("notice").textContent = INDEX.demo
    ? `Demo data: fictional riders on the real calendar, demo date ${longDay(INDEX.today)} ${INDEX.season}. Real ProCyclingStats data replaces it on the next build.`
    : "";
  document.getElementById("built").textContent = `Built ${INDEX.built.replace("T", " ")}.`;
  const m = h.match(/^race-(.+?)(?:-s(\d+))?$/);
  const page = m ? "calendar" : h;
  document.querySelectorAll("[data-nav]").forEach(a => a.dataset.nav === page ? a.setAttribute("aria-current", "page") : a.removeAttribute("aria-current"));
  try {
    if (m) await raceView(m[1], m[2] ? +m[2] : null);
    else if (h === "accuracy") await accuracyView();
    else if (h === "method") methodView();
    else calendarView();
  } catch (e) { view.innerHTML = `<p>Could not load this page (${esc(e.message)}).</p>`; }
  if (!m || !m[2]) window.scrollTo(0, 0);
}
window.addEventListener("hashchange", route);
route();

// ------------------------------------------------------------ calendar
function calendarView() {
  document.title = `Breakaway · ${INDEX.season}`;
  const kind = r => r.n > 1 ? `${r.n} stages` : r.kind === "championship_itt" ? "time trial" : r.kind === "championship" ? "championship" : "one day";
  const status = r => r.status === "finished" ? (r.winner ? `won by ${esc(r.winner)}` : "result")
    : r.status === "running" ? `racing now${r.p_bw != null ? ` · next stage break ${pct(r.p_bw)}` : ""}`
    : `${r.early ? "early preview" : "preview"}${r.p_bw != null ? ` · break ${pct(r.p_bw)}` : ""}`;
  let html = `<h1>${INDEX.season} season</h1><p class="meta">${INDEX.races.length} races. Each preview is published as soon as the race is known and updated with the startlist.</p>`;
  let month = -1, todayDone = false;
  const rows = ['<div class="scroll"><table class="cal"><tbody>'];
  for (const r of INDEX.races) {
    const mo = d(r.start).getMonth();
    if (mo !== month) { rows.push(`<tr class="mo"><th colspan="4">${MONTH[mo]}</th></tr>`); month = mo; }
    if (!todayDone && r.end >= INDEX.today) { rows.push(`<tr class="dim"><td>${day(INDEX.today)}</td><td colspan="3">today</td></tr>`); todayDone = true; }
    const name = r.gender === "women" && !/women/i.test(r.name) ? `${r.name} (women)` : r.name;
    rows.push(`<tr${r.status === "finished" ? ' class="dim"' : ""}><td>${span(r.start, r.end)}</td><td class="w"><a href="#race-${r.id}">${esc(name)}</a></td>
      <td class="kind hide-s">${kind(r)}</td><td class="status">${status(r)}</td></tr>`);
  }
  rows.push("</tbody></table></div>");
  view.innerHTML = html + rows.join("");
}

// ------------------------------------------------------------ race
async function raceView(id, n) {
  const R = await get(`data/race/${id}.json`);
  const S = R.stages.find(s => s.n === n) || R.stages.find(s => s.status === "upcoming") || R.stages[R.stages.length - 1];
  const who = rid => R.riders[rid] || [rid, "", ""];
  document.title = `${R.name} ${R.year} · Breakaway`;
  const multi = R.stages.length > 1;
  let h = `<p class="small back"><a href="#calendar">All races</a></p><h1>${esc(R.name)} ${R.year}</h1>
    <p class="meta">${span(R.start, R.end)} · ${multi ? `${R.stages.length} stages` : R.kind === "championship_itt" ? "time trial" : "one-day race"}${R.gender === "women" ? " · women" : ""}</p>`;
  if (multi) h += `<nav class="stages" aria-label="Stages">${R.stages.map(s => `<a href="#race-${R.id}-s${s.n}" class="${s.status === "finished" ? "done" : ""}" ${s === S ? 'aria-current="page"' : ""}>${s.n}</a>`).join("")}</nav>`;
  const climbs = Object.entries(S.climbs || {}).map(([c, k]) => `${k} × ${c === "HC" ? "HC" : "cat. " + c}`).join(", ");
  h += `<div class="stagehead"><h3>${multi ? `Stage ${S.n} · ` : ""}${longDay(S.date)}</h3>${profile(S.icon)}
    <span class="meta">${S.km ?? "–"} km · ${S.vert != null ? S.vert.toLocaleString("en") + " m climbing · " : ""}${S.itt ? "time trial" : ICON[S.icon] || ""}${S.ps != null ? ` · profile score ${S.ps}` : ""}</span></div>
    ${climbs ? `<p class="meta">Climbs: ${climbs}${S.last_climb ? `; last: ${esc(S.last_climb)}` : ""}.</p>` : ""}`;
  view.innerHTML = h + (S.status === "finished" ? finished(R, S, who) : upcoming(R, S, who));
  if (S.status === "upcoming" && !S.itt && S.riders) liveSetup(R, S, who);
}

function winTable(rows, S, who, n = 12) {
  const mx = Math.max(...rows.slice(0, n).map(r => r[2]), 1e-9);
  return `<div class="scroll"><table><thead><tr><th>Rider</th><th class="hide-s"></th><th class="n">Win</th><th class="n">Odds</th></tr></thead><tbody>${
    rows.slice(0, n).map(r => { const [rid, pb, pw, qb, qc] = r, viaB = (S.p_bw ?? 0) * pb * qb, viaP = (1 - (S.p_bw ?? 0)) * qc, t = viaB + viaP || 1;
      return `<tr><td class="nm">${esc(who(rid)[0])} <span class="muted small">${esc((who(rid)[2] || "").split(" ·")[0])}</span></td>
        <td class="barcell hide-s"><div class="bar" title="from the bunch ${pct(pw * viaP / t)}, from the break ${pct(pw * viaB / t)}"><i class="p" style="width:${100 * pw * viaP / t / mx}%"></i><i class="b" style="width:${100 * pw * viaB / t / mx}%"></i></div></td>
        <td class="n">${pct(pw, 1)}</td><td class="n">${odds(pw)}</td></tr>`; }).join("")}</tbody></table></div>`;
}

function upcoming(R, S, who) {
  if (!S.riders) return `<p class="lead">${esc(S.note || "No preview yet.")}</p>`;
  let h = S.provisional ? `<p class="meta">Early preview. The startlist is not out yet, so last year's riders who are still racing stand in for the field. It is replaced when PCS publishes the startlist.</p>` : "";
  const fav = S.riders[0];
  if (S.itt) h += `<p class="lead">Time trial. Favourite <b>${esc(who(fav[0])[0])}</b>, ${pct(fav[2])} · odds ${odds(fav[2])}.</p>`;
  else h += `<p class="lead">The breakaway wins <b class="acc">${pct(S.p_bw)}</b> of the time.</p>
    <p class="meta">Fair odds: break wins ${odds(S.p_bw)} · break caught ${odds(1 - S.p_bw)}. Expected break: ${Math.round(S.size)} riders. Favourite: ${esc(who(fav[0])[0])}, ${pct(fav[2])} (${odds(fav[2])}).</p>`;
  const brk = [...S.riders].sort((a, b) => b[1] - a[1]).slice(0, 12);
  h += `<div class="cols"><div><h2>Who wins</h2>${winTable(S.riders, S, who)}
      ${S.itt ? "" : `<p class="small muted">Bar: chance of winning from the bunch (dark) and from the break (colour).</p>`}</div>`;
  if (!S.itt) h += `<div><h2>Who goes in the break</h2><div class="scroll"><table><thead><tr><th>Rider</th><th class="hide-s">Team</th><th class="n">In break</th><th class="n">Odds</th></tr></thead><tbody>${
    brk.map(r => `<tr><td>${esc(who(r[0])[0])}</td><td class="muted small hide-s">${esc(who(r[0])[1])}</td><td class="n">${pct(r[1])}</td><td class="n">${odds(r[1])}</td></tr>`).join("")}</tbody></table></div></div>`;
  h += `</div><h2>Key paths to victory</h2>` + keyPaths(S, who);
  const P = S.pattern || {};
  if (P.n) h += `<h2>How it has been won</h2><p>${P.n} past ${R.stages.length > 1 ? "stages of this kind in this race" : "editions"}: bunch sprint ${pct(P.sprint_share)}, small group ${pct(P.small_group_share)}, solo ${pct(P.solo_share)}.
    ${P.solo_km_q ? `Winning solos went ${P.solo_km_q[0]}–${P.solo_km_q[1]} km from the line. ` : ""}${P.break_win_rate != null ? `The day's break won ${pct(P.break_win_rate)} of the time` : ""}${P.winning_break_size ? `, with a median of ${P.winning_break_size} riders when it did` : ""}.</p>`;
  if (!S.itt) h += `<h2>Live</h2><p class="meta">Set the race situation from the TV or the PCS live ticker.</p><div id="live"></div>`;
  if (S.versions?.length) h += `<h2>Preview history</h2><div class="scroll"><table><thead><tr><th>Date</th><th>Field</th><th class="n">Break wins</th><th>Favourite</th></tr></thead><tbody>${
    S.versions.map(v => `<tr><td>${day(v.at)}</td><td class="muted">${v.prov ? "last year's riders" : "startlist"}</td><td class="n">${pct(v.p_bw)}</td><td>${esc(who(v.fav)[0])}</td></tr>`).join("")}</tbody></table></div>`;
  return h;
}

// Advice shared by a whole group (favourites, break candidates) is shown once, then what differs per rider.
function keyPaths(S, who) {
  const favs = new Set(S.favs || []), entries = Object.entries(S.paths || {});
  const groups = [["Favourites", entries.filter(([r]) => favs.has(r))], ["Break candidates", entries.filter(([r]) => !favs.has(r))]];
  return groups.filter(([, g]) => g.length).map(([title, g]) => {
    const common = g.length > 1 ? g[0][1].filter(t => g.every(([, tips]) => tips.includes(t))) : [];
    const riders = g.map(([rid, tips]) => { const r = S.riders.find(x => x[0] === rid), own = tips.filter(t => !common.includes(t));
      return `<div class="path"><h3>${esc(who(rid)[0])} <span>${esc(who(rid)[2])} · win ${pct(r[2], 1)} · break ${pct(r[1])}</span></h3>
        ${own.length ? `<ul class="tips">${own.map(t => `<li>${esc(t)}</li>`).join("")}</ul>` : '<p class="small muted">Same plan as above.</p>'}</div>`; }).join("");
    return `<h3 style="margin:18px 0 6px">${title}</h3>${common.length ? `<ul class="tips">${common.map(t => `<li>${esc(t)}</li>`).join("")}</ul>` : ""}${riders}`;
  }).join("");
}

function finished(R, S, who) {
  const X = S.result, C = S.call;
  let h = `<p class="lead">Won by <b>${esc(who(X.winner)[0])}</b>${X.how ? `, ${esc(X.how.toLowerCase())}` : ""}.</p>`;
  if (X.brk) h += `<p class="meta">${X.brk.length ? `The break of ${X.brk.length} (${X.brk.map(r => esc(who(r)[0])).join(", ")}) ${X.break_won ? "won the stage" : "was caught"}.` : "No breakaway went clear for long."}</p>`;
  else h += `<p class="meta">PCS has no breakaway data for this race.</p>`;
  if (!C) return h + `<p class="muted">No prediction was stored for this ${R.stages.length > 1 ? "stage" : "race"}.</p>`;
  h += `<h2>Our call</h2><ul class="tips">
    <li>${C.retro ? "Computed after the race (no preview was stored in time)." : `Made ${day(C.at)}${C.backtest ? ", in the season backtest (model trained only on earlier seasons)" : ""}.`}</li>
    ${S.itt ? "" : `<li>Breakaway wins: we said ${pct(C.p_bw)}; ${X.break_won == null ? "no breakaway data" : X.break_won ? "it won" : "it was caught"}.</li>`}
    ${X.winner_rank ? `<li>Winner ${esc(who(X.winner)[0])}: number ${X.winner_rank} in our preview, at ${pct(X.winner_p, 1)} (odds ${odds(X.winner_p)}).</li>` : ""}
    ${X.brk?.length && X.break_hits != null ? `<li>Break: ${X.break_hits} of the ${X.brk.length} riders were in our top ${X.brk.length} candidates.</li>` : ""}</ul>`;
  const pos = Object.fromEntries((X.top10 || []).map((r, i) => [r, i + 1])), inb = new Set(X.brk || []);
  h += `<div class="scroll"><table><thead><tr><th>Our ranking</th><th class="n">Win</th><th class="n">Odds</th><th class="n">Result</th><th class="n hide-s">In break</th></tr></thead><tbody>${
    S.riders.slice(0, 15).map(r => `<tr${pos[r[0]] ? "" : ' class="dim"'}><td>${esc(who(r[0])[0])}</td><td class="n">${pct(r[2], 1)}</td><td class="n">${odds(r[2])}</td>
      <td class="n">${pos[r[0]] ? pos[r[0]] : "–"}</td><td class="n hide-s">${inb.has(r[0]) ? "yes" : ""}</td></tr>`).join("")}</tbody></table></div>`;
  return h;
}

// ------------------------------------------------------------ live calculator
function liveSetup(R, S, who) {
  const L = INDEX.live, box = document.getElementById("live");
  const cand = [...S.riders].sort((a, b) => b[1] - a[1]).slice(0, 40), k = Math.max(1, Math.round(S.size || 1));
  box.innerHTML = `<div class="live">
    <div><span class="lab">Riders in the break</span><div class="picker">${cand.map((r, i) =>
      `<label><input type="checkbox" value="${esc(r[0])}" ${i < k ? "checked" : ""}> ${esc(who(r[0])[0])}<span>${pct(r[1])}</span></label>`).join("")}</div></div>
    <div><label for="km">Km to go: <b id="kmv"></b></label><input type="range" id="km" min="0" max="${S.km}" step="0.5" value="${Math.min(60, S.km)}">
      <span class="lab" style="margin-top:14px">Gap</span><input type="number" id="gm" min="0" max="30" value="2" aria-label="minutes"> min
      <input type="number" id="gs" min="0" max="59" value="30" aria-label="seconds"> s
      <span class="lab" style="margin-top:14px">Chase</span><div class="chase" id="chase">${Object.keys(L.chase).map(c =>
        `<button type="button" data-c="${c}" aria-pressed="${c === "several teams"}">${c}</button>`).join("")}</div>
      <label style="margin-top:14px;color:var(--fg);font-size:14px"><input type="checkbox" id="hard" ${S.icon === "p3" || S.icon === "p5" ? "checked" : ""}> a climb or hard finish is still to come</label></div></div>
    <div id="liveOut"></div>`;
  const run = () => liveRender(R, S, who);
  box.querySelectorAll("input").forEach(x => x.addEventListener("input", run));
  box.querySelectorAll("#chase button").forEach(b => b.addEventListener("click", () => {
    box.querySelectorAll("#chase button").forEach(x => x.setAttribute("aria-pressed", x === b)); run(); }));
  run();
}

function liveRender(R, S, who) {
  const L = INDEX.live, q = s => document.querySelector(s);
  const brk = [...document.querySelectorAll(".picker input:checked")].map(x => x.value);
  const st = { n: brk.length, km: +q("#km").value, gap: 60 * (+q("#gm").value || 0) + (+q("#gs").value || 0),
    chase: q("#chase [aria-pressed=true]").dataset.c, hard: q("#hard").checked };
  q("#kmv").textContent = `${st.km} km`;
  const ps = pSurvive(L, st, S.p_bw, S.km), need = gapNeeded(L, st, st.km), inb = new Set(brk);
  const fo = S.riders.map(r => inb.has(r[0]) ? r[5] : 0), fv = S.riders.map(r => inb.has(r[0]) ? 0 : r[6]);
  const so = fo.reduce((a, b) => a + b, 0) || 1, sv = fv.reduce((a, b) => a + b, 0) || 1;
  const live = S.riders.map((r, i) => ({ id: r[0], in: inb.has(r[0]), q: fo[i] / so, p: ps * fo[i] / so + (1 - ps) * fv[i] / sv, fo: r[5] }))
    .sort((a, b) => b.p - a.p);
  const fastest = live.filter(x => x.in).reduce((a, b) => (b.fo > (a?.fo ?? -1) ? b : a), null);
  const P = S.pattern || {}, win = P.solo_km_q ? `${P.solo_km_q[0]}–${P.solo_km_q[1]} km` : null;
  const checkpoints = [30, 20, 10].filter(x => x <= S.km).map(x => `${mmss(gapNeeded(L, st, x))} at ${x} km`).join(", ");
  q("#liveOut").innerHTML = `<p class="lead">The break survives <b class="acc">${pct(ps)}</b> · odds ${odds(ps)} <span class="muted">· caught ${odds(1 - ps)}</span></p>
    <p class="meta">Gap ${mmss(st.gap)}; a ${st.n}-rider break needs ${mmss(need)} here for even chances (${checkpoints} to go). Pre-race call ${pct(S.p_bw)}, which counts less as the finish gets closer.</p>
    <div class="cols"><div><h2>Gap needed</h2>${needChart(L, st, S.km)}</div>
    <div><h2>Live odds</h2><div class="scroll"><table><thead><tr><th>Rider</th><th></th><th class="n">Win</th><th class="n">Odds</th></tr></thead><tbody>${
      live.slice(0, 14).map(x => `<tr><td>${esc(who(x.id)[0])}</td><td class="small ${x.in ? "acc" : "muted"}">${x.in ? "break" : "bunch"}</td><td class="n">${pct(x.p, 1)}</td><td class="n">${odds(x.p)}</td></tr>`).join("")}</tbody></table></div></div></div>
    <h2>Live key paths</h2>${live.filter(x => x.in).map(x => `<div class="path"><h3>${esc(who(x.id)[0])} <span>in the break · ${pct(x.p, 1)} · odds ${odds(x.p)}</span></h3><ul class="tips">
      <li>${ps < 0.6 ? "Keep working with the break: the gap is not safe yet." : "The gap looks safe: the win is now decided inside the break."} If it stays away: ${pct(x.q)} to win.</li>
      <li>${fastest && x.id === fastest.id ? "Fastest finisher of the group: follow the attacks and wait for the sprint." :
        `Not the fastest finisher here: attack before the last kilometre${win ? `; winning solos here went ${win} out` : ""}${S.last_climb ? `, ${esc(S.last_climb)} is the place` : ""}.`}</li></ul></div>`).join("")}`;
}

function needChart(L, st, dist) {
  const W = 460, H = 230, l = 40, r = 10, t = 10, b = 30, max = Math.min(Math.max(100, st.km), dist);
  const ks = []; for (let k = max; k > 0; k -= Math.max(1, max / 40)) ks.push(k); ks.push(0);
  const g = ks.map(k => gapNeeded(L, st, k) / 60), top = Math.ceil(Math.max(...g, st.gap / 60, 1) + .5);
  const x = k => l + (W - l - r) * (1 - k / max), y = m => H - b - (H - t - b) * Math.min(m, top) / top;
  const line = ks.map((k, i) => `${i ? "L" : "M"}${x(k).toFixed(1)} ${y(g[i]).toFixed(1)}`).join(" ");
  let s = `<path class="zone" d="${line} L${x(0)} ${y(top)} L${x(max)} ${y(top)}Z"/>`;
  for (let m = 0; m <= top; m += Math.max(1, Math.round(top / 5))) s += `<line class="grid" x1="${l}" x2="${W - r}" y1="${y(m)}" y2="${y(m)}"/><text x="${l - 6}" y="${y(m) + 4}" text-anchor="end">${m}:00</text>`;
  for (let k = 0; k <= max; k += max > 60 ? 20 : 10) s += `<text x="${x(k)}" y="${H - b + 15}" text-anchor="middle">${k}</text>`;
  s += `<text x="${W - r}" y="${H - 2}" text-anchor="end">km to go</text><path class="line" d="${line}"/>
    <circle class="now" cx="${x(Math.min(st.km, max))}" cy="${y(st.gap / 60)}" r="5"/>`;
  return `<svg class="chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="Gap the break needs at each distance">${s}</svg>
    <p class="small muted">Dashed line: even chances. Above it (shaded) the break is favoured. Dot: now.</p>`;
}

// ------------------------------------------------------------ accuracy
async function accuracyView() {
  document.title = "Accuracy · Breakaway";
  const A = await get("data/accuracy.json"), M = A.metrics, o = M.overall;
  const row = (label, a, b) => `<tr><td>${label}</td><td class="n">${a}</td><td class="n muted">${b}</td></tr>`;
  let h = `<h1>Accuracy</h1><p class="meta">Every season from ${M.test_years[0]} to ${M.test_years[M.test_years.length - 1]} is predicted by models trained only on earlier seasons, then compared with what happened. From now on, each preview is also scored after its race.</p>
    <div class="scroll"><table><thead><tr><th>Measure</th><th class="n">Model</th><th class="n">Baseline</th></tr></thead><tbody>
    ${row("Does the break win? AUC (1 = perfect, 0.5 = coin toss)", o.bw_auc.toFixed(2), `${o.bw_auc_baseline.toFixed(2)} race history`)}
    ${row("Does the break win? Brier score (lower is better)", o.bw_brier.toFixed(3), o.bw_brier_baseline.toFixed(3))}
    ${row("Who is in the break? AUC", o.member_auc.toFixed(2), `${o.member_auc_baseline.toFixed(2)} rider's break rate`)}
    ${row("Break riders found among our top picks", pct(o.member_precision_at_k), "")}
    ${row("Winner was our favourite", pct(o.winner_top1), "")}
    ${row("Winner in our top 3", pct(o.winner_top3), `${pct(o.winner_top3_baseline)} form`)}
    ${row("Winner in our top 10", pct(o.winner_top10), "")}</tbody></table></div>`;
  h += `<h2>By season</h2><div class="scroll"><table><thead><tr><th>Season</th><th class="n">Break AUC</th><th class="n">Break riders AUC</th><th class="n">Winner top 3</th></tr></thead><tbody>${
    Object.entries(M.per_year).map(([y, v]) => `<tr><td>${y}</td><td class="n">${v.bw_auc?.toFixed(2) ?? "–"}</td><td class="n">${v.member_auc?.toFixed(2) ?? "–"}</td><td class="n">${pct(v.winner_top3)}</td></tr>`).join("")}</tbody></table></div>`;
  h += `<div class="cols"><div><h2>Calibration of the break call</h2>${calibChart(A.calib)}</div>
    <div><h2>Biggest misses</h2><div class="scroll"><table><tbody>${A.misses.map(x => `<tr><td class="muted small">${day(x.date)}</td><td><a href="#race-${x.id}">${esc(x.race)}</a><br><span class="small muted">${esc(x.note)}</span></td></tr>`).join("")}</tbody></table></div></div></div>`;
  h += `<h2>By race</h2><p class="small muted">Sorted from least to most accurate break call.</p><div class="scroll"><table><thead><tr><th>Race</th><th class="n">Stages</th><th class="n">Break Brier</th><th class="n hide-s">Break riders found</th><th class="n">Winner top 3</th></tr></thead><tbody>${
    [...A.per_race].sort((a, b) => (b.brier ?? -1) - (a.brier ?? -1)).map(x => `<tr><td>${esc(x.race)}</td><td class="n">${x.stages}</td><td class="n">${x.brier ?? "–"}</td><td class="n hide-s">${pct(x.hits)}</td><td class="n">${pct(x.top3)}</td></tr>`).join("")}</tbody></table></div>`;
  view.innerHTML = h;
}

function calibChart(c) {
  const W = 380, H = 300, l = 40, r = 10, t = 10, b = 36, x = p => l + (W - l - r) * p, y = p => H - b - (H - t - b) * p;
  let s = "";
  [0, .25, .5, .75, 1].forEach(p => { s += `<line class="grid" x1="${l}" x2="${W - r}" y1="${y(p)}" y2="${y(p)}"/><text x="${l - 6}" y="${y(p) + 4}" text-anchor="end">${p * 100}%</text><text x="${x(p)}" y="${H - b + 15}" text-anchor="middle">${p * 100}%</text>`; });
  const mx = Math.max(...c.map(v => v.n), 1);
  s += `<line class="diag" x1="${x(0)}" y1="${y(0)}" x2="${x(1)}" y2="${y(1)}"/>` + c.map(v => `<circle class="dot" cx="${x(v.pred)}" cy="${y(v.obs)}" r="${3 + 9 * Math.sqrt(v.n / mx)}"><title>${pct(v.pred)} predicted, ${pct(v.obs)} happened (${v.n} stages)</title></circle>`).join("");
  s += `<text x="${W - r}" y="${H - 2}" text-anchor="end">predicted</text>`;
  return `<svg class="chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="Predicted versus observed breakaway wins">${s}</svg>
    <p class="small muted">Stages grouped by predicted chance. On the dotted line, the predictions match what happened.</p>`;
}

// ------------------------------------------------------------ method
function methodView() {
  document.title = "Method · Breakaway";
  view.innerHTML = `<h1>Method</h1>
  <p>Breakaway covers the road races on the French betting regulator's list (ANJ decision 2026-159): the WorldTour and other pro races, the Tour de France Femmes, and the World, European and Olympic road races and time trials. National championships are left out.</p>
  <h2>Data</h2><p>Results come from ProCyclingStats. PCS records how many kilometres each rider spent ahead of the peloton. A rider was in the day's breakaway when that is at least 20 km and a quarter of the race; the breakaway won when the winner was one of them. Races without breakaway data are used only for the winner models.</p>
  <h2>Before the race</h2><p>Five gradient-boosting models: does the break win, how many riders are in it, who is in it, who wins if it survives, and who wins if it is caught. They combine into each rider's chance of winning: the chance the break wins × his chance of being in it × his chance of winning it, plus the chance it is caught × his chance of winning from the bunch. The fair odds are 1 ÷ that chance.</p>
  <p>Every input is computed from races before the one being predicted: break habits by terrain and over the last year, results on similar terrain, form, record in this race, the team's habits, the GC position before the stage, the profile and climbs, the strength of the field, and each race's own history, so a small race is not judged by Grand Tour standards.</p>
  <p>A preview appears as soon as the race is on PCS. Until the startlist is out, last year's riders who are still racing stand in for the field. Each new version is kept.</p>
  <h2>Live</h2><p>A peloton chasing with several teams closes about one minute every 10 km, faster in the final 10 km and against a small break. The break survives if its gap beats what the peloton can close before the line, with a margin for uncertainty. Early in the race the pre-race call counts more. Every live estimate is saved and, once enough races have finished, the live model is refitted on them.</p>
  <h2>Checking ourselves</h2><p>After each race the stored previews are scored against the result and the models are retrained with the new results. The accuracy page shows the season tests, how well the probabilities match reality, race by race, and the biggest misses.</p>`;
}
