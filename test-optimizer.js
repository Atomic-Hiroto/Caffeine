#!/usr/bin/env node
/*
  Caffeine optimizer benchmark harness.
  Run:  node test-optimizer.js

  Compares three strategies on a battery of scenarios (varying locked-dose counts,
  peak windows, "current-time" constraints, sleep tightness):

    OLD    — greedy water-filling mg allocator + 30-min grid + simulated annealing
             (pre-upgrade baseline)
    NEW    — exact 2-constraint LP mg allocator + 15-min grid + coordinate descent
             (current implementation in index.html)
    TRUTH  — exhaustive 5-min grid over time configs + exact LP per config
             (ground truth reference, slow)

  For each scenario we print peak avg, sleep level, total mg, % gap vs TRUTH,
  and wall time. A final summary ranks the strategies.
*/

const HL = 5.5;
const DAILY_LIMIT = 400;
const SLEEP_THRESHOLD = 50;

/* ---------- Shared primitives (copied verbatim from index.html) ---------- */
const caffeineAt = (t, doses) => {
  let s = 0;
  for (const d of doses) {
    if (t >= d.time) {
      const el = t - d.time;
      s += d.mg * Math.min(1, el / 0.75) * Math.pow(0.5, el / HL);
    }
  }
  return s;
};

const peakAverage = (doses, pS, pE, step = 0.1) => {
  let sum = 0, n = 0;
  for (let t = pS; t <= pE; t += step) { sum += caffeineAt(t, doses); n++; }
  return n ? sum / n : 0;
};

const peakCoefAt = (doseTime, pS, pE) => {
  if (doseTime >= pE) return 0;
  let sum = 0, n = 0;
  for (let t = pS; t <= pE; t += 0.1) {
    if (t >= doseTime) {
      const el = t - doseTime;
      sum += Math.min(1, el / 0.75) * Math.pow(0.5, el / HL);
    }
    n++;
  }
  return n ? sum / n : 0;
};

const sleepCoefAt = (doseTime, sT) => {
  if (doseTime > sT) return 0;
  const el = sT - doseTime;
  return Math.min(1, el / 0.75) * Math.pow(0.5, el / HL);
};

const clamp = (v, mn, mx) => Math.max(mn, Math.min(mx, v));
const snapT = t => Math.round(t * 12) / 12;
const snapM = m => Math.round(m / 5) * 5;

/* ---------- OLD: greedy water-filling mg allocator ---------- */
const allocateMg_greedy = (times, pS, pE, sT, mgMin, mgMax, totalBudget, sleepBudget, respectSleep) => {
  const N = times.length;
  if (N === 0) return [];
  const c = times.map(t => peakCoefAt(t, pS, pE));
  const s = times.map(t => sleepCoefAt(t, sT));
  const mg = new Array(N).fill(mgMin);
  let totalUsed = mgMin * N;
  let sleepUsed = mg.reduce((a, v, i) => a + v * s[i], 0);
  if (totalUsed > totalBudget) return mg;
  const order = Array.from({ length: N }, (_, i) => i);
  const rank = i => {
    if (!respectSleep) return c[i];
    if (s[i] <= 1e-6) return c[i] * 1e6;
    return c[i] / Math.max(s[i], 0.01);
  };
  order.sort((a, b) => rank(b) - rank(a));
  for (const i of order) {
    const maxAddByTotal = totalBudget - totalUsed;
    const maxAddBySleep = respectSleep
      ? (s[i] > 1e-6 ? Math.max(0, (sleepBudget - sleepUsed) / s[i]) : Infinity)
      : Infinity;
    const maxAddByBox = mgMax - mg[i];
    const add = Math.max(0, Math.min(maxAddByTotal, maxAddBySleep, maxAddByBox));
    mg[i] += add;
    totalUsed += add;
    sleepUsed += add * s[i];
    if (totalUsed >= totalBudget - 1e-6) break;
  }
  return mg;
};

/* ---------- NEW: exact LP mg allocator via vertex enumeration ---------- */
const allocateMg_LP = (times, pS, pE, sT, mgMin, mgMax, totalBudget, sleepBudget, respectSleep) => {
  const N = times.length;
  if (N === 0) return [];
  const c = times.map(t => peakCoefAt(t, pS, pE));
  const s = times.map(t => sleepCoefAt(t, sT));
  const EPS = 1e-6;
  const feasible = (mg) => {
    let tot = 0, sl = 0;
    for (let i = 0; i < N; i++) {
      if (mg[i] < mgMin - EPS || mg[i] > mgMax + EPS) return false;
      tot += mg[i]; sl += s[i] * mg[i];
    }
    if (tot > totalBudget + EPS) return false;
    if (respectSleep && sl > sleepBudget + EPS) return false;
    return true;
  };
  const objective = (mg) => { let v = 0; for (let i = 0; i < N; i++) v += c[i] * mg[i]; return v; };
  let bestMg = new Array(N).fill(mgMin), bestObj = -Infinity;
  if (feasible(bestMg)) bestObj = objective(bestMg);
  const tryVertex = (mg) => {
    if (!feasible(mg)) return;
    const obj = objective(mg);
    if (obj > bestObj) { bestObj = obj; bestMg = mg.slice(); }
  };
  const maxK = 1 + (respectSleep ? 1 : 0);
  for (let basicMask = 0; basicMask < (1 << N); basicMask++) {
    const basicIdx = [];
    for (let i = 0; i < N; i++) if (basicMask & (1 << i)) basicIdx.push(i);
    const k = basicIdx.length;
    if (k > maxK) continue;
    const nonBasic = [];
    for (let i = 0; i < N; i++) if (!(basicMask & (1 << i))) nonBasic.push(i);
    const nb = nonBasic.length;
    for (let bm = 0; bm < (1 << nb); bm++) {
      const mg = new Array(N);
      for (let j = 0; j < nb; j++) mg[nonBasic[j]] = (bm & (1 << j)) ? mgMax : mgMin;
      if (k === 0) {
        for (const i of basicIdx) mg[i] = mgMin;
        tryVertex(mg); continue;
      }
      const active = respectSleep ? (k === 2 ? [["T", "S"]] : [["T"], ["S"]]) : [["T"]];
      for (const combo of active) {
        if (k === 1) {
          const a = combo[0] === "T" ? 1 : s[basicIdx[0]];
          if (Math.abs(a) < 1e-9) continue;
          let rhs = combo[0] === "T" ? totalBudget : sleepBudget;
          for (const j of nonBasic) rhs -= (combo[0] === "T" ? 1 : s[j]) * mg[j];
          mg[basicIdx[0]] = rhs / a;
        } else {
          const i1 = basicIdx[0], i2 = basicIdx[1];
          const a = 1, b = 1, c2 = s[i1], d = s[i2];
          let r0 = totalBudget, r1 = sleepBudget;
          for (const j of nonBasic) { r0 -= mg[j]; r1 -= s[j] * mg[j]; }
          const det = a * d - b * c2;
          if (Math.abs(det) < 1e-9) continue;
          mg[i1] = (d * r0 - b * r1) / det;
          mg[i2] = (a * r1 - c2 * r0) / det;
        }
        tryVertex(mg);
      }
    }
  }
  return bestMg;
};

/* ---------- Scenario → optimizer context ---------- */
const buildContext = (sc) => {
  const locked = sc.locked || [];
  const capTotal = sc.capTotal ?? true;
  const respectSleep = sc.respectSleep ?? true;
  const TOTAL_CAP_RAW = capTotal ? DAILY_LIMIT : 800;
  const lockedTotal = locked.reduce((a, d) => a + d.mg, 0);
  const lockedSleepLevel = locked.length ? caffeineAt(sc.sT, locked) : 0;
  const totalBudget = Math.max(0, TOTAL_CAP_RAW - lockedTotal);
  const sleepBudget = Math.max(0, SLEEP_THRESHOLD - lockedSleepLevel);
  const tMin = Math.max(4, sc.pS - 2, sc.earliestTime ?? 0);
  const tMax = Math.min(sc.pE, sc.sT - 0.5);
  return {
    ...sc, locked, respectSleep, capTotal, lockedSleepLevel, totalBudget, sleepBudget,
    tMin, tMax, mgMin: 10, mgMax: 300,
  };
};

const scoreCandidate = (ctx, cand) => {
  const all = ctx.locked.concat(cand);
  const peakAvg = peakAverage(all, ctx.pS, ctx.pE, 0.1);
  let penalty = 0;
  if (ctx.respectSleep) {
    const candSleep = caffeineAt(ctx.sT, cand);
    const remaining = Math.max(0, SLEEP_THRESHOLD - ctx.lockedSleepLevel);
    const excess = Math.max(0, candSleep - remaining);
    penalty += excess ** 2 * 4;
  }
  const tot = all.reduce((a, d) => a + d.mg, 0);
  if (ctx.capTotal && tot > DAILY_LIMIT) penalty += (tot - DAILY_LIMIT) ** 2 * 1.5;
  return peakAvg - penalty;
};

/* ---------- OLD optimizer: 30-min grid + greedy + SA ---------- */
function optimizer_OLD(sc) {
  const ctx = buildContext(sc);
  const { nNew, pS, pE, sT, respectSleep, locked, tMin, tMax, mgMin, mgMax, totalBudget, sleepBudget } = ctx;
  const N = Math.max(1, Math.min(8, nNew | 0));
  if (tMin >= tMax - 0.1) {
    return Array.from({ length: N }, (_, i) => ({
      time: snapT(clamp(tMin + i * 0.25, tMin, tMin + N)), mg: mgMin, type: "custom", locked: false,
    }));
  }
  const buildFromTimes = (times) => {
    const mgs = allocateMg_greedy(times, pS, pE, sT, mgMin, mgMax, totalBudget, sleepBudget, respectSleep);
    return times.map((t, i) => ({ time: snapT(t), mg: snapM(mgs[i]) }));
  };
  const gridLo = Math.max(tMin, pS - 1.5);
  const gridHi = Math.min(tMax, pE + 0.25);
  const slots = [];
  for (let t = gridLo; t <= gridHi + 1e-9; t += 0.5) slots.push(t);
  if (slots[0] > tMin) slots.unshift(tMin);
  if (slots[slots.length - 1] < tMax) slots.push(tMax);
  let best = null, bestScore = -Infinity;
  const combos = [];
  const MAX_COMBOS = 4000;
  const rec = (start, acc) => {
    if (combos.length >= MAX_COMBOS) return;
    if (acc.length === N) { combos.push(acc.slice()); return; }
    for (let i = start; i < slots.length; i++) {
      if (combos.length >= MAX_COMBOS) return;
      acc.push(slots[i]); rec(i + 1, acc); acc.pop();
    }
  };
  if (N <= 4) rec(0, []);
  else {
    for (let k = 0; k < MAX_COMBOS; k++) {
      const picked = [], used = new Set();
      while (picked.length < N) {
        const j = Math.floor(Math.random() * slots.length);
        if (!used.has(j)) { used.add(j); picked.push(slots[j]); }
      }
      picked.sort((a, b) => a - b); combos.push(picked);
    }
  }
  for (const times of combos) {
    const cand = buildFromTimes(times);
    const s = scoreCandidate(ctx, cand);
    if (s > bestScore) { best = cand.map(x => ({ ...x })); bestScore = s; }
  }
  // SA refinement (seeded for some determinism across runs — but still stochastic)
  let current = best.map(x => ({ ...x })), curScore = bestScore;
  const iters = 1500, T0 = 15;
  for (let k = 0; k < iters; k++) {
    const T = T0 * Math.pow(0.9975, k);
    const cand = current.map(d => ({ ...d }));
    const idx = Math.floor(Math.random() * N);
    const roll = Math.random();
    if (roll < 0.55) {
      const span = 1.2 * (T / T0) + 0.15;
      cand[idx].time = snapT(clamp(cand[idx].time + (Math.random() - 0.5) * 2 * span, tMin, tMax));
      cand.sort((a, b) => a.time - b.time);
      const mgs = allocateMg_greedy(cand.map(d => d.time), pS, pE, sT, mgMin, mgMax, totalBudget, sleepBudget, respectSleep);
      cand.forEach((d, i) => { d.mg = snapM(mgs[i]); });
    } else {
      const span = 50 * (T / T0) + 8;
      cand[idx].mg = snapM(clamp(cand[idx].mg + (Math.random() - 0.5) * 2 * span, mgMin, mgMax));
    }
    const s = scoreCandidate(ctx, cand);
    const dS = s - curScore;
    if (dS > 0 || Math.random() < Math.exp(dS / Math.max(T, 0.01))) {
      current = cand; curScore = s;
      if (s > bestScore) { best = cand.map(x => ({ ...x })); bestScore = s; }
    }
  }
  return best.map(d => ({ time: d.time, mg: d.mg, type: "custom", locked: false }));
}

/* ---------- NEW optimizer: 15-min grid + LP + coordinate descent ---------- */
function optimizer_NEW(sc) {
  const ctx = buildContext(sc);
  const { nNew, pS, pE, sT, respectSleep, locked, tMin, tMax, mgMin, mgMax, totalBudget, sleepBudget } = ctx;
  const N = Math.max(1, Math.min(8, nNew | 0));
  if (tMin >= tMax - 0.1) {
    return Array.from({ length: N }, (_, i) => ({
      time: snapT(clamp(tMin + i * 0.25, tMin, tMin + N)), mg: mgMin, type: "custom", locked: false,
    }));
  }
  const buildFromTimes = (times) => {
    const mgs = allocateMg_LP(times, pS, pE, sT, mgMin, mgMax, totalBudget, sleepBudget, respectSleep);
    return times.map((t, i) => ({ time: snapT(t), mg: snapM(mgs[i]) }));
  };
  const gridLo = Math.max(tMin, pS - 1.5);
  const gridHi = Math.min(tMax, pE + 0.25);
  const slots = [];
  for (let t = gridLo; t <= gridHi + 1e-9; t += 0.25) slots.push(Math.round(t * 12) / 12);
  if (slots.length === 0 || slots[0] > tMin + 1e-9) slots.unshift(tMin);
  if (slots[slots.length - 1] < tMax - 1e-9) slots.push(tMax);
  let best = null, bestScore = -Infinity;
  const MAX_COMBOS = 10000;
  if (N <= 4) {
    let count = 0; const acc = [];
    const rec = (start) => {
      if (count >= MAX_COMBOS) return;
      if (acc.length === N) {
        count++;
        const cand = buildFromTimes(acc);
        const s = scoreCandidate(ctx, cand);
        if (s > bestScore) { best = cand.map(x => ({ ...x })); bestScore = s; }
        return;
      }
      for (let i = start; i < slots.length; i++) {
        if (count >= MAX_COMBOS) return;
        acc.push(slots[i]); rec(i + 1); acc.pop();
      }
    };
    rec(0);
  } else {
    const stride = Math.max(1, Math.ceil(Math.pow(slots.length, N / 4) / MAX_COMBOS));
    const acc = []; let count = 0;
    const rec = (start) => {
      if (count >= MAX_COMBOS) return;
      if (acc.length === N) {
        count++;
        const cand = buildFromTimes(acc);
        const s = scoreCandidate(ctx, cand);
        if (s > bestScore) { best = cand.map(x => ({ ...x })); bestScore = s; }
        return;
      }
      for (let i = start; i < slots.length; i += stride) {
        if (count >= MAX_COMBOS) return;
        acc.push(slots[i]); rec(i + 1); acc.pop();
      }
    };
    rec(0);
  }
  if (!best) {
    best = Array.from({ length: N }, (_, i) => ({
      time: snapT(clamp(tMin + i * 0.25, tMin, tMax)), mg: mgMin,
    }));
    bestScore = scoreCandidate(ctx, best);
  }
  const fiveMin = 1 / 12;
  const deltas = [-3, -2, -1, 1, 2, 3];
  let improved = true, guard = 0;
  while (improved && guard++ < 50) {
    improved = false;
    for (let i = 0; i < N; i++) {
      const origT = best[i].time;
      for (const d of deltas) {
        const newT = snapT(clamp(origT + d * fiveMin, tMin, tMax));
        if (newT === origT) continue;
        const times = best.map((x, j) => j === i ? newT : x.time).sort((a, b) => a - b);
        const cand = buildFromTimes(times);
        const sc2 = scoreCandidate(ctx, cand);
        if (sc2 > bestScore + 1e-5) {
          best = cand.map(x => ({ ...x }));
          bestScore = sc2; improved = true;
        }
      }
    }
  }
  return best.map(d => ({ time: d.time, mg: d.mg, type: "custom", locked: false }));
}

/* ---------- TRUTH: exhaustive 5-min grid + LP per config ---------- */
function optimizer_TRUTH(sc) {
  const ctx = buildContext(sc);
  const { nNew, pS, pE, sT, respectSleep, locked, tMin, tMax, mgMin, mgMax, totalBudget, sleepBudget } = ctx;
  const N = Math.max(1, Math.min(8, nNew | 0));
  if (tMin >= tMax - 0.1) {
    return Array.from({ length: N }, (_, i) => ({
      time: snapT(clamp(tMin + i * 0.25, tMin, tMin + N)), mg: mgMin, type: "custom", locked: false,
    }));
  }
  const buildFromTimes = (times) => {
    const mgs = allocateMg_LP(times, pS, pE, sT, mgMin, mgMax, totalBudget, sleepBudget, respectSleep);
    return times.map((t, i) => ({ time: snapT(t), mg: snapM(mgs[i]) }));
  };
  // Fine 5-min grid. We still cap tMax at peak end (doses past that contribute 0 to peak avg).
  const slots = [];
  for (let t = tMin; t <= tMax + 1e-9; t += 1 / 12) slots.push(Math.round(t * 12) / 12);
  let best = null, bestScore = -Infinity;
  const acc = [];
  const rec = (start) => {
    if (acc.length === N) {
      const cand = buildFromTimes(acc);
      const s = scoreCandidate(ctx, cand);
      if (s > bestScore) { best = cand.map(x => ({ ...x })); bestScore = s; }
      return;
    }
    for (let i = start; i < slots.length; i++) {
      acc.push(slots[i]); rec(i + 1); acc.pop();
    }
  };
  // Guard against combinatorial explosion for large N — TRUTH is meaningful only for N ≤ 3
  // at 5-min resolution over typical windows (~50 slots → C(50,3)=19600 which is fine).
  if (N > 3) {
    // Fall back to 15-min grid so truth remains tractable for N=4
    slots.length = 0;
    for (let t = tMin; t <= tMax + 1e-9; t += 0.25) slots.push(Math.round(t * 12) / 12);
  }
  rec(0);
  if (!best) best = Array.from({ length: N }, (_, i) => ({ time: snapT(clamp(tMin + i * 0.25, tMin, tMax)), mg: mgMin }));
  return best.map(d => ({ time: d.time, mg: d.mg, type: "custom", locked: false }));
}

/* ---------- Scenarios ---------- */
const scenarios = [
  // 0 locked
  { name: "Morning plan, 2 new, standard", nNew: 2, pS: 9, pE: 13, sT: 23, locked: [] },
  { name: "Morning plan, 3 new, standard", nNew: 3, pS: 9, pE: 13, sT: 23, locked: [] },
  { name: "Morning plan, 4 new, standard", nNew: 4, pS: 9, pE: 13, sT: 23, locked: [] },
  { name: "Single dose, wide peak window", nNew: 1, pS: 10, pE: 15, sT: 23, locked: [] },
  { name: "Single dose, narrow peak window", nNew: 1, pS: 13, pE: 14.5, sT: 23, locked: [] },
  { name: "Late bedtime (early)", nNew: 2, pS: 14, pE: 18, sT: 24.5, locked: [] },
  { name: "Early bedtime, tight", nNew: 2, pS: 9, pE: 12, sT: 21, locked: [] },

  // 1 locked (morning)
  {
    name: "1 locked @ 8am/120mg, need 2 more for afternoon peak",
    nNew: 2, pS: 13, pE: 17, sT: 23, locked: [{ time: 8, mg: 120, locked: true }],
  },
  {
    name: "1 locked @ 9:30am/80mg, 3 new",
    nNew: 3, pS: 10, pE: 15, sT: 23, locked: [{ time: 9.5, mg: 80, locked: true }],
  },

  // 2 locked (the bug scenario)
  {
    name: "The 6pm bug: 2 locked, need 1 more",
    nNew: 1, pS: 17, pE: 20, sT: 24, earliestTime: 18.25,
    locked: [
      { time: 11.5, mg: 105, locked: true },
      { time: 15.5, mg: 110, locked: true },
    ],
  },
  {
    name: "2 locked near sleep budget cap, evening peak",
    nNew: 2, pS: 16, pE: 20, sT: 23, earliestTime: 15,
    locked: [
      { time: 9, mg: 120, locked: true },
      { time: 13, mg: 100, locked: true },
    ],
  },

  // Sleep-dominated
  {
    name: "Very late peak window, sleep dominates",
    nNew: 2, pS: 18, pE: 22, sT: 23.5, locked: [],
  },
  {
    name: "respectSleep = false, late peak",
    nNew: 2, pS: 18, pE: 22, sT: 23.5, locked: [], respectSleep: false,
  },

  // Over-locked total budget
  {
    name: "Heavy locked dose, tight budget",
    nNew: 2, pS: 14, pE: 18, sT: 23,
    locked: [{ time: 8, mg: 250, locked: true }],
  },

  // Planning from "now"
  {
    name: "Afternoon start (planToday, now=15:00)",
    nNew: 2, pS: 15, pE: 19, sT: 23, earliestTime: 15.25, locked: [],
  },
];

/* ---------- Runner ---------- */
function evaluate(label, fn, sc, runs = 1) {
  let t0 = process.hrtime.bigint();
  let best = null, bestScore = -Infinity;
  const ctx = buildContext(sc);
  const results = [];
  for (let r = 0; r < runs; r++) {
    const out = fn(sc);
    const sc2 = scoreCandidate(ctx, out);
    results.push({ out, sc2 });
    if (sc2 > bestScore) { best = out; bestScore = sc2; }
  }
  const t1 = process.hrtime.bigint();
  const ms = Number(t1 - t0) / 1e6 / runs;
  const merged = (sc.locked || []).concat(best).sort((a, b) => a.time - b.time);
  const avg = peakAverage(merged, sc.pS, sc.pE, 0.05);
  const sleep = caffeineAt(sc.sT, merged);
  const total = merged.reduce((a, d) => a + d.mg, 0);
  const variance = runs > 1 ? (() => {
    const scores = results.map(r => r.sc2);
    const m = scores.reduce((a, b) => a + b, 0) / scores.length;
    const v = scores.reduce((a, b) => a + (b - m) ** 2, 0) / scores.length;
    return Math.sqrt(v);
  })() : 0;
  return { label, best, avg, sleep, total, ms, variance, bestScore };
}

function fmtDoses(doses) {
  return doses.map(d => {
    const h = Math.floor(d.time), m = Math.round((d.time - h) * 60);
    return `${h}:${String(m).padStart(2, "0")}/${d.mg}mg`;
  }).join(", ");
}

function runScenario(sc) {
  console.log("\n" + "═".repeat(86));
  console.log("▶ " + sc.name);
  console.log("   peak=" + sc.pS + "-" + sc.pE + "  sleep=" + sc.sT +
              "  nNew=" + sc.nNew + "  locked=" + (sc.locked || []).length +
              (sc.respectSleep === false ? "  (no sleep cap)" : "") +
              (sc.earliestTime ? "  earliest=" + sc.earliestTime : ""));
  if ((sc.locked || []).length) console.log("   locked: " + fmtDoses(sc.locked));

  // NEW and TRUTH are deterministic → one run. OLD is stochastic → 5 runs.
  const rOld = evaluate("OLD  ", optimizer_OLD, sc, 5);
  const rNew = evaluate("NEW  ", optimizer_NEW, sc, 1);
  const rTruth = evaluate("TRUTH", optimizer_TRUTH, sc, 1);

  const gap = (r) => {
    if (rTruth.avg <= 1e-6) return "0.00";
    return (((rTruth.avg - r.avg) / rTruth.avg) * 100).toFixed(2);
  };

  const row = (r) =>
    r.label + " | " +
    ("peakAvg " + r.avg.toFixed(2) + "mg").padEnd(18) +
    ("sleep " + r.sleep.toFixed(1)).padEnd(13) +
    ("tot " + r.total.toFixed(0)).padEnd(10) +
    ("gap " + gap(r) + "%").padEnd(12) +
    ("t " + r.ms.toFixed(1) + "ms").padEnd(12) +
    (r.variance > 0 ? "σ=" + r.variance.toFixed(3) : "") +
    "\n       doses: " + fmtDoses(r.best);

  console.log(row(rOld));
  console.log(row(rNew));
  console.log(row(rTruth));
  return { sc, rOld, rNew, rTruth };
}

/* ---------- Main ---------- */
console.log("Caffeine optimizer benchmark");
console.log("OLD=greedy+SA (baseline, 5 runs)  NEW=LP+coord-descent  TRUTH=5-min brute+LP");

const results = scenarios.map(runScenario);

console.log("\n" + "═".repeat(86));
console.log("SUMMARY (lower gap = better; gap is % short of TRUTH peak avg)");
console.log("─".repeat(86));
const gapOf = (r, truth) => truth.avg <= 1e-6 ? 0 : ((truth.avg - r.avg) / truth.avg) * 100;
let sumOld = 0, sumNew = 0, worstOld = 0, worstNew = 0, winsOld = 0, winsNew = 0, ties = 0;
for (const { rOld, rNew, rTruth } of results) {
  const gO = gapOf(rOld, rTruth);
  const gN = gapOf(rNew, rTruth);
  sumOld += gO; sumNew += gN;
  worstOld = Math.max(worstOld, gO); worstNew = Math.max(worstNew, gN);
  if (rNew.avg > rOld.avg + 0.05) winsNew++;
  else if (rOld.avg > rNew.avg + 0.05) winsOld++;
  else ties++;
}
const n = results.length;
console.log(`  OLD avg gap: ${(sumOld / n).toFixed(3)}%   worst: ${worstOld.toFixed(2)}%`);
console.log(`  NEW avg gap: ${(sumNew / n).toFixed(3)}%   worst: ${worstNew.toFixed(2)}%`);
console.log(`  Head-to-head (NEW vs OLD): NEW wins ${winsNew}, OLD wins ${winsOld}, ties ${ties}`);
console.log(`  Determinism: NEW is deterministic (σ=0), OLD is stochastic`);
