/** Terrain statistics and navigation grid, ported line for line from ml/inference/terrain.py. */
import type { ClassKey, PerClass, SpatialEntry, TerrainGrid, TerrainStats } from '../types/api';

const KEYS: ClassKey[] = ['soil', 'bedrock', 'sand', 'big_rock'];
const LABELS: Record<ClassKey, string> = { soil: 'Soil', bedrock: 'Bedrock', sand: 'Sand', big_rock: 'Big rock' };
const ROWS = ['upper', 'middle', 'lower'];
const COLS = ['left', 'centre', 'right'];
const IGNORE = 255;

const pct = (n: number, d: number) => (d ? Math.round((1e6 * n) / d) / 1e4 : 0);
const r4 = (v: number) => Math.round(v * 1e4) / 1e4;
const r6 = (v: number) => Math.round(v * 1e6) / 1e6;
const edges = (n: number, parts: number) => Array.from({ length: parts + 1 }, (_, i) => Math.floor((i * n) / parts));
const per = <T,>(f: (k: ClassKey, i: number) => T): PerClass<T> =>
  Object.fromEntries(KEYS.map((k, i) => [k, f(k, i)])) as PerClass<T>;

/** 8-connected components of a boolean mask; returns the number of regions and the largest region's size. */
function components(mask: Uint8Array, w: number, h: number): [number, number] {
  const seen = new Uint8Array(mask.length);
  const queue = new Int32Array(mask.length);
  let regions = 0, largest = 0;
  for (let s = 0; s < mask.length; s++) {
    if (!mask[s] || seen[s]) continue;
    regions++;
    let head = 0, tail = 0, size = 0;
    queue[tail++] = s;
    seen[s] = 1;
    while (head < tail) {
      const p = queue[head++];
      size++;
      const y = (p / w) | 0, x = p - y * w;
      for (let dy = -1; dy <= 1; dy++) {
        const ny = y + dy;
        if (ny < 0 || ny >= h) continue;
        for (let dx = -1; dx <= 1; dx++) {
          const nx = x + dx;
          if ((dx || dy) && nx >= 0 && nx < w) {
            const q = ny * w + nx;
            if (mask[q] && !seen[q]) { seen[q] = 1; queue[tail++] = q; }
          }
        }
      }
    }
    largest = Math.max(largest, size);
  }
  return [regions, largest];
}

export function terrainStatistics(cls: Uint8Array, w: number, h: number, conf?: Float32Array, unc?: Float32Array, bands = 12): TerrainStats {
  const counts = [0, 0, 0, 0];
  let valid = 0;
  for (let p = 0; p < cls.length; p++) if (cls[p] !== IGNORE) { counts[cls[p]]++; valid++; }
  const percent = per((_, i) => pct(counts[i], valid));
  const dominant = valid ? KEYS[counts.indexOf(Math.max(...counts))] : null;
  const re = edges(h, 3), ce = edges(w, 3), be = edges(h, bands);

  const spatial = per((key, i): SpatialEntry => {
    const n = counts[i];
    const zones: Record<string, number> = {};
    const zoneCount = new Array(9).fill(0);
    let sy = 0, sx = 0, minY = h, maxY = -1, minX = w, maxX = -1;
    const mask = new Uint8Array(cls.length);
    for (let y = 0; y < h; y++) {
      const zr = y < re[1] ? 0 : y < re[2] ? 1 : 2;
      for (let x = 0; x < w; x++) {
        if (cls[y * w + x] !== i) continue;
        mask[y * w + x] = 1;
        sy += y; sx += x;
        if (y < minY) minY = y; if (y > maxY) maxY = y; if (x < minX) minX = x; if (x > maxX) maxX = x;
        zoneCount[zr * 3 + (x < ce[1] ? 0 : x < ce[2] ? 1 : 2)]++;
      }
    }
    ROWS.forEach((r, a) => COLS.forEach((c, b) => { zones[`${r}_${c}`] = pct(zoneCount[a * 3 + b], n); }));
    if (!n) {
      return { pixels: 0, centroid: null, bbox: null, regions: 0, largest_region_percent_of_class: 0, largest_region_percent_of_image: 0, zone_share_percent: zones };
    }
    const [regions, largest] = components(mask, w, h);
    void key;
    return {
      pixels: n,
      centroid: [r4(sy / n / h), r4(sx / n / w)],
      bbox: [r4(minY / h), r4(minX / w), r4((maxY + 1) / h), r4((maxX + 1) / w)],
      regions,
      largest_region_percent_of_class: pct(largest, n),
      largest_region_percent_of_image: pct(largest, valid),
      zone_share_percent: zones,
    };
  });

  const zone_composition: TerrainStats['zone_composition'] = {};
  ROWS.forEach((r, a) => COLS.forEach((c, b) => {
    const cc = [0, 0, 0, 0];
    let size = 0;
    for (let y = re[a]; y < re[a + 1]; y++) for (let x = ce[b]; x < ce[b + 1]; x++) {
      const v = cls[y * w + x];
      if (v !== IGNORE) { cc[v]++; size++; }
    }
    zone_composition[`${r}_${c}`] = { percent: per((_, i) => pct(cc[i], size)), dominant: size ? KEYS[cc.indexOf(Math.max(...cc))] : null };
  }));
  const band_profile: TerrainStats['band_profile'] = [];
  for (let bi = 0; bi < bands; bi++) {
    const bc = [0, 0, 0, 0];
    let size = 0;
    for (let y = be[bi]; y < be[bi + 1]; y++) for (let x = 0; x < w; x++) {
      const v = cls[y * w + x];
      if (v !== IGNORE) { bc[v]++; size++; }
    }
    band_profile.push({ band: bi + 1, from_top_percent: Math.round((1000 * be[bi]) / h) / 10, ...per((_, i) => pct(bc[i], size)) });
  }

  const stats: TerrainStats = {
    height: h, width: w, valid_pixels: valid, unlabeled_pixels: h * w - valid,
    class_pixels: per((_, i) => counts[i]), class_percent: percent, dominant_class: dominant,
    spatial, zone_composition, band_profile, summary: '',
  };

  if (conf) {
    const hist = new Array(10).fill(0);
    let sum = 0, min = 1, low = 0, high = 0;
    const perSum = [0, 0, 0, 0];
    const vals: number[] = [];
    for (let p = 0; p < cls.length; p++) {
      if (cls[p] === IGNORE) continue;
      const v = conf[p];
      sum += v; if (v < min) min = v; if (v < 0.5) low++; if (v >= 0.9) high++;
      hist[Math.min(9, Math.floor(v * 10))]++;
      perSum[cls[p]] += v;
      vals.push(v);
    }
    vals.sort((a, b) => a - b);
    const median = vals.length ? (vals.length % 2 ? vals[(vals.length - 1) / 2] : (vals[vals.length / 2 - 1] + vals[vals.length / 2]) / 2) : null;
    stats.confidence = {
      mean: valid ? r6(sum / valid) : null, median: median === null ? null : r6(median), min: valid ? r6(min) : null,
      per_class_mean: per((_, i) => (counts[i] ? r6(perSum[i] / counts[i]) : null)),
      histogram: hist.map((n, j) => ({ from: j / 10, to: (j + 1) / 10, pixels: n, percent: pct(n, valid) })),
      low_confidence_percent: pct(low, valid), high_confidence_percent: pct(high, valid), calibrated: false,
      note: 'Softmax confidence of the predicted class. Not calibrated; it is not a measure of accuracy.',
    };
  }
  if (unc) {
    let sum = 0, high = 0;
    const perSum = [0, 0, 0, 0];
    for (let p = 0; p < cls.length; p++) {
      if (cls[p] === IGNORE) continue;
      sum += unc[p]; if (unc[p] > 0.5) high++; perSum[cls[p]] += unc[p];
    }
    stats.uncertainty = {
      mean_entropy: valid ? r6(sum / valid) : null, high_uncertainty_percent: pct(high, valid),
      per_class_mean_entropy: per((_, i) => (counts[i] ? r6(perSum[i] / counts[i]) : null)),
      definition: 'Normalised predictive entropy of the softmax distribution (0 = one class certain, 1 = all four classes equally likely).',
    };
  }
  stats.summary = describe(stats);
  return stats;
}

export function describe(s: TerrainStats): string {
  if (!s.valid_pixels || !s.dominant_class) return 'No classified pixels.';
  const p = s.class_percent;
  const ranked = [...KEYS].sort((a, b) => p[b] - p[a]);
  const parts = ranked.filter((k) => p[k] >= 0.05).map((k) => `${LABELS[k].toLowerCase()} ${p[k].toFixed(1)}%`);
  const dom = s.dominant_class;
  let text = `The predicted terrain is dominated by ${LABELS[dom].toLowerCase()} (${p[dom].toFixed(1)}% of classified pixels). `;
  text += `Composition: ${parts.join(', ')}. `;
  const absent = KEYS.filter((k) => p[k] < 0.05).map((k) => LABELS[k].toLowerCase());
  if (absent.length) text += `Essentially absent (<0.05%): ${absent.join(', ')}. `;
  const zc = s.spatial[dom].zone_share_percent;
  const top = Object.keys(zc).reduce((a, b) => (zc[b] > zc[a] ? b : a));
  text += `Most ${LABELS[dom].toLowerCase()} pixels lie in the ${top.replace('_', ' ')} part of the frame. `;
  const c = s.confidence;
  if (c && c.mean !== null) {
    text += `Mean softmax confidence is ${(c.mean * 100).toFixed(1)}%, with ${c.low_confidence_percent.toFixed(1)}% of pixels below 50% confidence. Confidence is uncalibrated and does not measure accuracy.`;
  }
  return text;
}

/** Majority-vote grid whose long side has `size` cells (InferenceService.grid + class_grid). */
export function classGrid(cls: Uint8Array, w: number, h: number, size: number, unc?: Float32Array): TerrainGrid {
  const scale = size / Math.max(h, w);
  let rows = Math.max(2, Math.round(h * scale)), cols = Math.max(2, Math.round(w * scale));
  rows = Math.min(rows, h);
  cols = Math.min(cols, w);
  const re = edges(h, rows), ce = edges(w, cols);
  const cells: number[][] = [], u: number[][] = [];
  for (let r = 0; r < rows; r++) {
    const row: number[] = [], urow: number[] = [];
    for (let c = 0; c < cols; c++) {
      const cnt = [0, 0, 0, 0];
      let any = 0, us = 0, n = 0;
      for (let y = re[r]; y < re[r + 1]; y++) for (let x = ce[c]; x < ce[c + 1]; x++) {
        const v = cls[y * w + x];
        if (v !== IGNORE) { cnt[v]++; any++; }
        if (unc) { us += unc[y * w + x]; n++; }
      }
      row.push(any ? cnt.indexOf(Math.max(...cnt)) : -1);
      urow.push(n ? r4(us / n) : 0);
    }
    cells.push(row);
    u.push(urow);
  }
  return { rows, cols, cells, uncertainty: unc ? u : null };
}
